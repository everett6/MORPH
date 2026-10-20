import math
from benchmarks.delta_precision import acceptance


def test_gate_rejects_bad_updates_even_when_losses_match():
    reference=dict(loss=2.,grad_norm=1.)
    assert acceptance(reference,reference,.03)['passed']
    assert not acceptance(reference,reference,.11)['passed']
    assert not acceptance(reference,dict(loss=2.05,grad_norm=1.),.01)['passed']
    assert not acceptance(reference,dict(loss=2.,grad_norm=1.2),.01)['passed']
    assert not acceptance(reference,dict(loss=math.nan,grad_norm=1.),.01)['passed']
