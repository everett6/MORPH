"""Native MORPH sequential backbone with explicit recurrent state boundaries."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .ssm import SelectiveSSM
from .graph import GraphExchange


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        # FP32 reduction protects BF16 while preserving FP64 gradient tests.
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        return (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MorphBlock(nn.Module):
    def __init__(self, cfg, with_graph=False):
        super().__init__()
        self.graph = GraphExchange(cfg) if with_graph else None
        self.norm_ssm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ssm = SelectiveSSM(cfg)
        self.norm_ffn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, state=None, graph_memory=None, return_graph=False):
        graph_state = None
        if self.graph is not None and state is not None:
            state, graph_state = state
        update, state = self.ssm(self.norm_ssm(x), state)
        x = x + update
        x = x + self.ffn(self.norm_ffn(x))
        if self.graph is not None:
            x, graph_state, graph_memory = self.graph(x, graph_state, graph_memory)
            state = (state, graph_state)
        return (x, state, graph_memory) if return_graph else (x, state)


@dataclass
class MorphOutput:
    logits: torch.Tensor
    states: tuple[torch.Tensor, ...]


class MorphModel(nn.Module):
    def __init__(self, cfg, activation_checkpointing=False):
        super().__init__()
        self.config = cfg
        self.activation_checkpointing = activation_checkpointing
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(MorphBlock(cfg, cfg.graph_enabled and (i + 1) % cfg.graph_interval == 0) for i in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids, states=None):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError('input_ids must be [batch, nonempty sequence]')
        if states is not None and len(states) != len(self.blocks):
            raise ValueError('one state per block is required')
        x = self.embedding(input_ids)
        result = []
        graph_memory = None
        for i, block in enumerate(self.blocks):
            state = None if states is None else states[i]
            if self.activation_checkpointing and self.training:
                x, state, graph_memory = checkpoint(block, x, state, graph_memory, True, use_reentrant=False)
            else:
                x, state, graph_memory = block(x, state, graph_memory, True)
            result.append(state)
        return MorphOutput(self.lm_head(self.norm(x)), tuple(result))

    def loss(self, tokens):
        if tokens.shape[1] < 2:
            raise ValueError('training requires at least two tokens')
        logits = self(tokens[:, :-1]).logits
        return F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size), tokens[:, 1:].reshape(-1))
