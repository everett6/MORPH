"""Hybrid CPU-capable Muon/AdamW, following KellerJordan/Muon.

Muon: hidden Linear projection matrices only, FP32 Newton-Schulz quintic
iteration (3.4445,-4.7750,2.0315), 5 steps, Nesterov momentum 0.95,
sqrt(max(1, rows/cols)) scale. FP32 intentionally differs from upstream BF16
orthogonalization for a reproducible CPU reference. Auxiliary AdamW uses
PyTorch's reference optimizer. No reduced/quantized optimizer states.
"""
import torch
from torch import nn


def orthogonalize(gradient, steps=5):
    x = gradient.float()
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.T
        x = 3.4445 * x + (-4.775 * gram + 2.0315 * (gram @ gram)) @ x
    return x.T if transpose else x


def partition(model):
    hidden = {id(module.weight) for name, module in model.named_modules()
              if isinstance(module, nn.Linear) and name != 'lm_head'}
    muon, adam = [], []
    for param in model.parameters():
        (muon if id(param) in hidden else adam).append(param)
    return muon, adam


class HybridMuonAdamW:
    def __init__(self, model, lr=1e-3, muon_lr=None, weight_decay=.01):
        self.muon, adam = partition(model)
        self.lr = lr if muon_lr is None else muon_lr
        self.weight_decay = weight_decay
        self.momentum = {}
        self.adam = torch.optim.AdamW(adam, lr=lr, weight_decay=weight_decay, betas=(.9, .95), foreach=False)

    def zero_grad(self, set_to_none=True):
        for p in self.muon:
            p.grad = None if set_to_none else torch.zeros_like(p)
        self.adam.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        all_params = self.muon + [p for g in self.adam.param_groups for p in g['params']]
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in all_params):
            raise FloatingPointError('nonfinite gradient; optimizer not updated')
        for i, p in enumerate(self.muon):
            if p.grad is None:
                continue
            if i not in self.momentum:
                self.momentum[i] = torch.zeros_like(p)
            momentum = self.momentum[i]
            momentum.lerp_(p.grad, .05)
            update = orthogonalize(p.grad.lerp(momentum, .95))
            update *= max(1, p.shape[0] / p.shape[1]) ** .5
            p.mul_(1 - self.lr * self.weight_decay).add_(update, alpha=-self.lr)
        self.adam.step()

    def state_dict(self):
        return dict(momentum=self.momentum, adam=self.adam.state_dict(), lr=self.lr, weight_decay=self.weight_decay)

    def load_state_dict(self, state):
        self.momentum = {i: value.to(self.muon[i].device) for i, value in state['momentum'].items()}
        self.adam.load_state_dict(state['adam'])
        self.lr, self.weight_decay = state['lr'], state['weight_decay']
