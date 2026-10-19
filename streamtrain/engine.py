"""Exact DLS-FB replay with CPU masters and optional CUDA ping-pong transfers.

Only layer input boundaries live on CPU between passes. No optimizer updates
occur here. Every microbatch completes backward before returning. Modules are
functional templates; authoritative parameters never leave the CPU.
"""
from contextlib import nullcontext
import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F


class Stage(nn.Module):
    def __init__(self, module, kind, embedding=None):
        super().__init__()
        self.module = module
        self.kind = kind
        if embedding is not None:
            self.weight = embedding.weight

    def forward(self, x, graph=None):
        if self.kind == 'embedding':
            return self.module(x), None
        if self.kind == 'block':
            x, _, graph = self.module(x, graph_memory=graph, return_graph=True)
            return x, graph
        return F.linear(self.module(x), self.weight)


class Slot:
    def __init__(self, size, device, dtype, asynchronous):
        self.device = device
        self.asynchronous = asynchronous
        self.weights = torch.empty(size, device=device, dtype=dtype)
        self.host = torch.empty(size, dtype=dtype, pin_memory=asynchronous)
        self.grad_host = torch.empty(size, pin_memory=asynchronous)
        self.grad_device = torch.empty(size, device=device)
        self.loaded = None
        self.done = None
        self.copied = None
        self.pending = None
        self.status = 'FREE'
        self.version = None

    def flush(self):
        if self.pending is not None:
            if self.copied is not None:
                self.copied.synchronize()
            offset = 0
            for p in self.pending:
                value = self.grad_host[offset:offset+p.numel()].view_as(p)
                if p.grad is None:
                    p.grad = value.clone()
                else:
                    p.grad.add_(value)
                offset += p.numel()
            self.pending = None
        self.status = 'FREE'


class StreamTrainEngine:
    def __init__(self, model, device='cpu', async_mode=False, dtype=torch.float32):
        if any(p.device.type != 'cpu' for p in model.parameters()):
            raise ValueError('streaming requires CPU master parameters')
        self.model, self.device, self.dtype = model, torch.device(device), dtype
        self.async_mode = async_mode
        if async_mode and self.device.type != 'cuda':
            raise ValueError('async mode requires CUDA')
        self.stages = [Stage(model.embedding, 'embedding')]
        self.stages += [Stage(block, 'block') for block in model.blocks]
        self.stages += [Stage(model.norm, 'head', model.embedding)]
        self.specs = [list(stage.named_parameters()) for stage in self.stages]
        size = max(sum(p.numel() for _, p in spec) for spec in self.specs)
        self.slots = [Slot(size, self.device, dtype, async_mode) for _ in range(2)]
        self.parameter_version = 0
        self.h2d = torch.cuda.Stream(device=self.device) if async_mode else None
        self.d2h = torch.cuda.Stream(device=self.device) if async_mode else None
        self.compute = torch.cuda.current_stream(self.device) if self.device.type == 'cuda' else None
        self.transfer_bytes = dict(h2d_weights=0, d2h_gradients=0)

    def load(self, index, slot):
        slot.flush()
        if self.async_mode and slot.loaded is not None:
            slot.loaded.synchronize()  # pinned staging must outlive previous DMA
        offset = 0
        for _, p in self.specs[index]:
            slot.host[offset:offset+p.numel()].copy_(p.detach().flatten())
            offset += p.numel()
        context = torch.cuda.stream(self.h2d) if self.async_mode else nullcontext()
        with context:
            if self.async_mode and slot.done is not None:
                self.h2d.wait_event(slot.done)
            slot.weights[:offset].copy_(slot.host[:offset], non_blocking=self.async_mode)
            if self.async_mode:
                slot.loaded = torch.cuda.Event()
                slot.loaded.record(self.h2d)
        slot.status, slot.version = 'LOADING', self.parameter_version
        self.transfer_bytes['h2d_weights'] += offset * slot.weights.element_size()

    def parameters(self, index, slot, grad):
        if slot.version != self.parameter_version:
            raise RuntimeError('parameter version changed before replay')
        if self.async_mode:
            self.compute.wait_event(slot.loaded)
        slot.status = 'IN_USE'
        result, offset = {}, 0
        for name, p in self.specs[index]:
            result[name] = slot.weights[offset:offset+p.numel()].view_as(p).detach().requires_grad_(grad)
            offset += p.numel()
        for name, buffer in self.stages[index].named_buffers():
            result[name] = buffer.to(self.device)
        return result

    def release(self, slot):
        if self.async_mode:
            slot.done = torch.cuda.Event()
            slot.done.record(self.compute)
        slot.status = 'FREE'

    def evacuate(self, index, slot, grads):
        offset = 0
        for (_, p), grad in zip(self.specs[index], grads):
            slot.grad_device[offset:offset+p.numel()].copy_(grad.flatten())
            offset += p.numel()
        self.release(slot)
        with torch.cuda.stream(self.d2h) if self.async_mode else nullcontext():
            if self.async_mode:
                self.d2h.wait_event(slot.done)
            slot.grad_host[:offset].copy_(slot.grad_device[:offset], non_blocking=self.async_mode)
            if self.async_mode:
                slot.copied = torch.cuda.Event()
                slot.copied.record(self.d2h)
        slot.pending = [p for _, p in self.specs[index]]
        slot.status = 'EVACUATING'
        self.transfer_bytes['d2h_gradients'] += offset * 4

    def backward(self, tokens, scale=1.0):
        """Accumulate both token and graph cotangents through exact replay."""
        inputs = []
        state = (tokens[:, :-1].to(self.device), None)
        self.load(0, self.slots[0])
        for index in range(len(self.stages) - 1):
            slot = self.slots[index % 2]
            inputs.append(tuple(v.detach().cpu() if v is not None else None for v in state))
            self.load(index + 1, self.slots[(index + 1) % 2])
            params = self.parameters(index, slot, False)
            with torch.no_grad(), torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype == torch.bfloat16):
                state = functional_call(self.stages[index], params, state)
            self.release(slot)
        inputs.append(tuple(v.detach().cpu() if v is not None else None for v in state))
        cotangents = None
        loss_value = None
        for pos, index in enumerate(reversed(range(len(self.stages)))):
            slot = self.slots[index % 2]
            if index:
                self.load(index - 1, self.slots[(index - 1) % 2])
            params = self.parameters(index, slot, True)
            state = tuple(v.to(self.device).detach().requires_grad_(bool(index)) if v is not None else None
                          for v in inputs[index])
            state_values = [v for v in state if v is not None] if index else []
            parameters = [params[name] for name, _ in self.specs[index]]
            values = state_values + parameters
            with torch.enable_grad(), torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype == torch.bfloat16):
                output = functional_call(self.stages[index], params, state)
                if cotangents is None:
                    loss = F.cross_entropy(output.float().flatten(0, 1), tokens[:, 1:].to(self.device).flatten())
                    loss_value = loss.detach().item()
                    grads = torch.autograd.grad(loss * scale, values, allow_unused=True)
                else:
                    outputs = [v for v in output if v is not None]
                    grads = torch.autograd.grad(outputs, values, cotangents, allow_unused=True)
            grads = [torch.zeros_like(v) if g is None else g for v, g in zip(values, grads)]
            cotangents = grads[:len(state_values)] if index else None
            self.evacuate(index, slot, grads[len(state_values):])
            del params, output, grads, parameters, values, state_values
        for slot in self.slots:
            slot.flush()
        return loss_value
