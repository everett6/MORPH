"""Causal headwise selective SSM, implemented with ordinary PyTorch autograd.

This is a slow correctness reference, not a Mamba-2 kernel implementation.
Each head has a scalar negative A; B, C, and positive dt depend on input.
Zero-order hold: s'=exp(dt*A)*s + expm1(dt*A)/A * B*x.
The returned state retains its graph; callers choose any truncation explicitly.
"""
import torch
from torch import nn
from torch.nn import functional as F


def selective_scan(x, dt, a, b, c, state):
    outputs = []
    for t in range(x.shape[1]):
        da = dt[:, t] * a
        decay = da.exp()[..., None, None]
        integral = (da.expm1() / a)[..., None, None]
        state = decay * state + integral * x[:, t, :, :, None] * b[:, t, :, None, :]
        outputs.append((state * c[:, t, :, None, :]).sum(-1))
    return torch.stack(outputs, dim=1), state


def chunked_scan(x, dt, a, b, c, state, chunk_size=32):
    """Parallel affine scan inside bounded chunks, stable for negative A.

    No inverse products are formed. Masked exponent differences are nonpositive,
    avoiding prefix-product division and its long-sequence underflow problem.
    """
    outputs = []
    for start in range(0, x.shape[1], chunk_size):
        end = min(start + chunk_size, x.shape[1])
        da = dt[:, start:end] * a
        prefix = da.cumsum(1)
        difference = prefix.transpose(1, 2)[..., :, None] - prefix.transpose(1, 2)[..., None, :]
        causal = torch.ones(end-start, end-start, device=x.device, dtype=torch.bool).tril()
        transition = difference.masked_fill(~causal, -torch.inf).exp()
        drive = (da.expm1() / a)[..., None, None] * x[:, start:end, :, :, None] * b[:, start:end, :, None, :]
        states = torch.einsum('bhts,bshpn->bthpn', transition, drive)
        states = states + prefix.exp()[..., None, None] * state[:, None]
        outputs.append((states * c[:, start:end, :, None, :]).sum(-1))
        state = states[:, -1]
    return torch.cat(outputs, 1), state


class SelectiveSSM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backend = cfg.ssm_backend
        self.chunk_size = cfg.scan_chunk_size
        self.inner = cfg.d_model * cfg.expand
        self.heads = self.inner // cfg.head_dim
        self.head_dim, self.d_state = cfg.head_dim, cfg.d_state
        self.in_proj = nn.Linear(cfg.d_model, 2 * self.inner, bias=False)
        self.select_proj = nn.Linear(self.inner, self.heads + 2 * cfg.d_state, bias=False)
        self.a_log = nn.Parameter(torch.zeros(self.heads))
        self.dt_bias = nn.Parameter(torch.full((self.heads,), -2.0))
        self.skip = nn.Parameter(torch.ones(self.heads))
        self.out_proj = nn.Linear(self.inner, cfg.d_model, bias=False)

    def forward(self, x, state=None):
        batch, length, _ = x.shape
        u, gate = self.in_proj(x).chunk(2, -1)
        u = F.silu(u)
        select = self.select_proj(u)
        dt = F.softplus(select[..., :self.heads] + self.dt_bias)
        b, c = select[..., self.heads:].chunk(2, -1)
        b = b[:, :, None, :].expand(-1, -1, self.heads, -1)
        c = c[:, :, None, :].expand(-1, -1, self.heads, -1)
        u = u.view(batch, length, self.heads, self.head_dim)
        # Keep persistent recurrence in FP32 during low-precision compute.
        if u.dtype in (torch.bfloat16, torch.float16):
            u, dt, b, c = u.float(), dt.float(), b.float(), c.float()
        expected = (batch, self.heads, self.head_dim, self.d_state)
        if state is None:
            state = u.new_zeros(expected)
        elif state.shape != expected or state.device != u.device or state.dtype != u.dtype:
            raise ValueError('SSM state shape, device, and dtype must match its input')
        arguments = (u, dt, -self.a_log.to(u.dtype).exp(), b, c, state)
        if self.backend=='triton' and u.is_cuda:
            from .triton_scan import triton_scan
            y,state=triton_scan(*arguments,self.chunk_size)
        elif self.backend in ('parallel','triton'):
            y,state=chunked_scan(*arguments,self.chunk_size)
        else:y,state=selective_scan(*arguments)
        y = y + u * self.skip[None, None, :, None]
        return self.out_proj(y.flatten(-2) * F.silu(gate)), state
