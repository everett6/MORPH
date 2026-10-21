import math
import pytest
import torch
from benchmarks.audit_precision import compare_states,relative_error


def test_relative_update_error_uses_update_not_large_weight_norm():
    initial={'p':torch.tensor([1000.,1000.],dtype=torch.float64)}
    ref={'p':torch.tensor([[1001.,1000.],[1.,2.],[3.,4.]],dtype=torch.float64)}
    opt={'p':torch.tensor([[1001.,1000.2],[1.1,2.],[3.,4.2]],dtype=torch.float64)}
    result=compare_states(initial,ref,opt)
    assert result['update_relative_l2']==pytest.approx(.2)
    assert result['first_moment_relative_l2']==pytest.approx(.1/math.sqrt(5))
    assert result['second_moment_relative_l2']==pytest.approx(.2/5)
    assert result['finite']
