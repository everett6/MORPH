import copy
import pytest
import torch
from morph import MorphConfig, MorphModel
from streamtrain.engine import StreamTrainEngine
from streamtrain.optimizer import HybridMuonAdamW


@pytest.mark.parametrize('device,async_mode', [('cpu', False), ('cuda', False), ('cuda', True)])
def test_streamed_gradients_updates_and_buffer_reuse(device, async_mode):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(8)
    cfg = MorphConfig(vocab_size=16, d_model=8, num_layers=3, d_ff=16, expand=2,
                      d_state=2, head_dim=4, graph_enabled=True, graph_interval=1,
                      d_graph=4, graph_nodes=3, reasoning_steps=4)
    master = MorphModel(cfg)
    resident = copy.deepcopy(master).to(device)
    engine = StreamTrainEngine(master, device=device, async_mode=async_mode)
    a = HybridMuonAdamW(master, lr=0.001)
    b = HybridMuonAdamW(resident, lr=0.001)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    for _ in range(3):
        a.zero_grad(); b.zero_grad()
        expected = resident.loss(ids.to(device)); expected.backward()
        got = engine.backward(ids)
        assert abs(got - expected.item()) < 1e-5
        for (name, p), (_, q) in zip(master.named_parameters(), resident.named_parameters()):
            torch.testing.assert_close(p.grad, q.grad.cpu(), rtol=3e-4, atol=2e-6, msg=name)
        a.step(); b.step(); engine.parameter_version += 1
        for p, q in zip(master.parameters(), resident.parameters()):
            torch.testing.assert_close(p, q.cpu(), rtol=5e-4, atol=1e-5)
    assert all(p.device.type == 'cpu' for p in master.parameters())


def test_bf16_streamed_cuda_and_transfer_bytes():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(12)
    cfg = MorphConfig(vocab_size=16, d_model=8, num_layers=2, d_ff=16,
                      d_state=2, head_dim=4, graph_enabled=True, graph_interval=1,
                      d_graph=4, graph_nodes=3)
    master = MorphModel(cfg)
    resident = copy.deepcopy(master).cuda().bfloat16()
    engine = StreamTrainEngine(master, 'cuda', True, torch.bfloat16)
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        expected = resident.loss(ids.cuda())
    expected.backward()
    got = engine.backward(ids)
    assert abs(got - expected.item()) < .02
    for (name, p), (_, q) in zip(master.named_parameters(), resident.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad.float().cpu(), rtol=.12, atol=.002, msg=name)
    total = sum(p.numel() for spec in engine.specs for _, p in spec)
    head = sum(p.numel() for _, p in engine.specs[-1])
    assert engine.transfer_bytes['h2d_weights'] == 4 * total - 2 * head
    assert engine.transfer_bytes['d2h_gradients'] == 4 * total
