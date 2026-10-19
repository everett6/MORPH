import pytest
import torch
from morph.model.ssm import selective_scan, chunked_scan


@pytest.mark.parametrize('length,chunk', [(1, 4), (7, 3), (65, 16)])
def test_parallel_scan_forward_and_all_gradients(length, chunk):
    torch.manual_seed(5)
    x = torch.randn(1, length, 2, 3, dtype=torch.double, requires_grad=True)
    dt = (torch.rand(1, length, 2, dtype=torch.double)+.01).requires_grad_()
    a = torch.tensor([-.2, -3.], dtype=torch.double, requires_grad=True)
    b = torch.randn(1, length, 2, 2, dtype=torch.double, requires_grad=True)
    c = torch.randn_like(b, requires_grad=True)
    state = torch.randn(1, 2, 3, 2, dtype=torch.double, requires_grad=True)
    args = x, dt, a, b, c, state
    expected = selective_scan(*args)
    got = chunked_scan(*args, chunk)
    for p, q in zip(expected, got):
        torch.testing.assert_close(p, q)
    ga = torch.autograd.grad(sum(t.square().sum() for t in expected), args)
    gb = torch.autograd.grad(sum(t.square().sum() for t in got), args)
    for p, q in zip(ga, gb):
        torch.testing.assert_close(p, q)
