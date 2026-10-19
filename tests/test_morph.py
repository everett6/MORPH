import copy
import torch
import pytest
from morph import MorphConfig, MorphModel
from morph.model.ssm import selective_scan
from morph.training.runner import train_run


def tiny():
    return MorphConfig(vocab_size=16, d_model=8, num_layers=2, d_ff=16, expand=2, d_state=2, head_dim=4)


def test_scan_analytic_and_gradcheck():
    x = torch.tensor([[[[2.]], [[4.]]]], dtype=torch.double, requires_grad=True)
    dt = torch.full((1, 2, 1), torch.log(torch.tensor(2.)).item(), dtype=torch.double, requires_grad=True)
    a = torch.tensor([-1.], dtype=torch.double, requires_grad=True)
    b = torch.ones(1, 2, 1, 1, dtype=torch.double, requires_grad=True)
    c = b.detach().clone().requires_grad_()
    s = torch.ones(1, 1, 1, 1, dtype=torch.double, requires_grad=True)
    y, state = selective_scan(x, dt, a, b, c, s)
    torch.testing.assert_close(y.flatten(), torch.tensor([1.5, 2.75], dtype=torch.double))
    torch.testing.assert_close(state.flatten(), torch.tensor([2.75], dtype=torch.double))
    assert torch.autograd.gradcheck(selective_scan, (x, dt, a, b, c, s))


def test_model_chunk_causality_and_gradients():
    torch.manual_seed(42)
    model = MorphModel(tiny()).double()
    ids = torch.randint(16, (2, 6))
    full = model(ids)
    first = model(ids[:, :3])
    second = model(ids[:, 3:], first.states)
    torch.testing.assert_close(full.logits, torch.cat((first.logits, second.logits), 1))
    altered = ids.clone(); altered[:, 4:] = 0
    torch.testing.assert_close(full.logits[:, :4], model(altered).logits[:, :4])
    full.logits.square().mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert model.lm_head.weight is model.embedding.weight
    other = copy.deepcopy(model); other.zero_grad()
    one = other(ids[:, :3]); two = other(ids[:, 3:], one.states)
    torch.cat((one.logits, two.logits), 1).square().mean().backward()
    for p, q in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_config_and_meta_count():
    with pytest.raises(ValueError):
        MorphConfig(d_model=7, head_dim=4)
    cfg = MorphConfig.from_file('configs/morph_7b.json')
    with torch.device('meta'):
        model = MorphModel(cfg)
    assert 6.9e9 < sum(p.numel() for p in model.parameters()) < 7.1e9


def test_overfit_and_exact_resume(tmp_path):
    cfg = tiny()
    ids = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4]])
    whole = train_run(cfg, ids, steps=60, seed=7, lr=0.02)
    assert whole['losses'][-1] < whole['losses'][0] * 0.2
    checkpoint = tmp_path / 'state.pt'
    train_run(cfg, ids, steps=25, seed=7, lr=0.02, checkpoint=checkpoint)
    resumed = train_run(cfg, ids, steps=60, seed=999, lr=0.02, resume=checkpoint)
    assert resumed['losses'] == whole['losses']
    for key, value in whole['model'].state_dict().items():
        torch.testing.assert_close(value, resumed['model'].state_dict()[key], rtol=0, atol=0)


def test_activation_checkpoint_matches_full_graph_gradients():
    cfg = MorphConfig(vocab_size=16, d_model=8, num_layers=2, d_ff=16, expand=2,
                      d_state=2, head_dim=4, graph_enabled=True, graph_interval=1,
                      d_graph=4, graph_nodes=3)
    torch.manual_seed(2)
    plain = MorphModel(cfg)
    checkpointed = copy.deepcopy(plain)
    checkpointed.activation_checkpointing = True
    ids = torch.tensor([[1, 2, 3, 4]])
    plain.loss(ids).backward()
    checkpointed.loss(ids).backward()
    for p, q in zip(plain.parameters(), checkpointed.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_bf16_cached_state_is_reusable():
    model = MorphModel(tiny())
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.autocast('cpu', dtype=torch.bfloat16):
        full = model(ids)
        first = model(ids[:, :2]); last = model(ids[:, 2:], first.states)
    torch.testing.assert_close(full.logits, torch.cat((first.logits, last.logits), 1), rtol=.02, atol=.002)
