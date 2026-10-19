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


class SelectiveSSM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
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
        y, state = selective_scan(u, dt, -self.a_log.to(u.dtype).exp(), b, c, state)
        y = y + u * self.skip[None, None, :, None]
        return self.out_proj(y.flatten(-2) * F.silu(gate)), state
