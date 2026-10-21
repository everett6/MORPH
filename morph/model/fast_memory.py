"""Causal, differentiable session memory; no persistent model mutation at inference."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class FastMemoryState:
    matrix: torch.Tensor
    version: int = 0

    def clone(self):
        return FastMemoryState(self.matrix.clone(), self.version)

    def snapshot(self):
        return self.clone()

    def rollback(self, snapshot):
        # A rollback is itself a mutation: never reuse an old cache epoch.
        self.matrix = snapshot.matrix.clone()
        self.version = max(self.version, snapshot.version) + 1
        return self

    def reset(self):
        self.matrix = torch.zeros_like(self.matrix)
        self.version += 1
        return self


class GraphMemory(nn.Module):
    def __init__(self, dim, memory_dim=32, projection_backend="reference"):
        super().__init__()
        if dim <= 0 or memory_dim <= 0:
            raise ValueError('memory dimensions must be positive')
        if projection_backend not in ("reference", "batched"):
            raise ValueError("unsupported memory projection backend")
        self.projection_backend = projection_backend
        self.dim, self.memory_dim = dim, memory_dim
        self.key = nn.Linear(dim, memory_dim)
        self.value = nn.Linear(dim, memory_dim)
        self.query = nn.Linear(dim, memory_dim)
        self.read = nn.Linear(memory_dim, dim, bias=False)
        self.gates = nn.Linear(dim, 2)
        self.graph_projection = nn.Linear(dim, dim, bias=False)

    def forward(self, hidden, state=None, graph_context=None):
        if hidden.ndim != 3 or hidden.shape[-1] != self.dim:
            raise ValueError('hidden must have shape [batch,time,dim]')
        batch, length, _ = hidden.shape
        if graph_context is not None and graph_context.shape != (batch, self.dim):
            raise ValueError('graph_context must have shape [batch,dim]')
        if state is None:
            state = FastMemoryState(torch.zeros(batch, self.memory_dim, self.memory_dim,
                                               device=hidden.device, dtype=torch.float32))
        if state.matrix.shape != (batch, self.memory_dim, self.memory_dim) or state.matrix.dtype != torch.float32 or state.matrix.device != hidden.device:
            raise ValueError('memory state has incompatible shape, device or precision')
        matrix = state.matrix
        outputs = []
        projected = None
        if self.projection_backend == 'batched' and length:
            tokens = hidden
            if graph_context is not None:
                tokens = tokens + self.graph_projection(graph_context.to(hidden.dtype))[:, None]
            # Preserve the reference gate linear's storage dtype and disabled
            # autocast context (FP16 slots compute FP16 gates before conversion).
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                gates_all = torch.sigmoid(self.gates(tokens).float())
            projected = (tokens, F.normalize(self.key(tokens).float(), dim=-1, eps=1e-6),
                         self.value(tokens).float(), F.normalize(self.query(tokens).float(), dim=-1, eps=1e-6), gates_all)
        # Outer training differentiates through every update. State is functional:
        # passing a branch never mutates its parent or the caller's checkpoint.
        for index in range(length):
            if projected is None:
                token = hidden[:, index]
                if graph_context is not None:
                    token = token + self.graph_projection(graph_context.to(token.dtype))
                key = F.normalize(self.key(token).float(), dim=-1, eps=1e-6)
                value = self.value(token).float()
                query = F.normalize(self.query(token).float(), dim=-1, eps=1e-6)
            else:
                token, key, value, query, _ = (tensor[:, index] for tensor in projected)
            # Disable autocast for state operations even under FP16 model training.
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                retrieved = torch.bmm(matrix, query.unsqueeze(-1)).squeeze(-1)
                error = value - torch.bmm(matrix, key.unsqueeze(-1)).squeeze(-1)
                gates = (torch.sigmoid(self.gates(token).float())
                         if projected is None else projected[4][:, index])
                eta, decay = gates[:, :1], gates[:, 1:] * 0.1
                correction = error.unsqueeze(-1) * key.unsqueeze(-2)
                correction = correction / (key.square().sum(-1)[:, None, None] + 1e-6)
                matrix = (1-decay.unsqueeze(-1)) * matrix + eta.unsqueeze(-1) * correction
            outputs.append(token + self.read(retrieved.to(token.dtype)))
        output = torch.stack(outputs, 1) if outputs else hidden.clone()
        return output, FastMemoryState(matrix, state.version + length)
