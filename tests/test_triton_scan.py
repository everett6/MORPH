import pytest
import torch
from morph.model.ssm import selective_scan


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_fused_scan_forward_all_gradients_and_chunk_boundaries():
    pytest.importorskip('triton')
    from morph.model.triton_scan import triton_scan
    torch.manual_seed(10)
    for length in [7,33]:
        args=[torch.randn(1,length,2,3,device='cuda'),torch.rand(1,length,2,device='cuda')*.3+.1,
              torch.tensor([-.4,-1.2],device='cuda'),torch.randn(1,length,2,4,device='cuda'),
              torch.randn(1,length,2,4,device='cuda'),torch.randn(1,2,3,4,device='cuda')]
        args=[value.requires_grad_() for value in args]
        expected=selective_scan(*args);got=triton_scan(*args,chunk_size=8)
        for p,q in zip(expected,got):torch.testing.assert_close(p,q,rtol=5e-4,atol=3e-5)
        a=torch.autograd.grad(sum(value.square().sum() for value in expected),args)
        b=torch.autograd.grad(sum(value.square().sum() for value in got),args)
        for p,q in zip(a,b):torch.testing.assert_close(p,q,rtol=1e-3,atol=3e-4)
