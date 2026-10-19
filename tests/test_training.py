import copy
import pytest
import torch
from morph import MorphConfig, MorphModel
from morph.training.runner import train_run
from streamtrain.optimizer import HybridMuonAdamW, partition


def test_adam_auxiliary_matches_reference_and_nonfinite_is_atomic():
    model = MorphModel(MorphConfig(vocab_size=16, d_model=8, num_layers=1, d_ff=16, d_state=2, head_dim=4))
    _, auxiliary = partition(model)
    reference = [torch.nn.Parameter(p.detach().clone()) for p in auxiliary]
    hybrid = HybridMuonAdamW(model, lr=.003)
    adam = torch.optim.AdamW(reference, lr=.003, weight_decay=.01, betas=(.9, .95), foreach=False)
    for _ in range(3):
        for p, q in zip(auxiliary, reference):
            p.grad = torch.ones_like(p) * .2
            q.grad = p.grad.clone()
        hybrid.step(); adam.step()
    for p, q in zip(auxiliary, reference):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    before = copy.deepcopy(model.state_dict())
    next(model.parameters()).grad.fill_(float('nan'))
    with pytest.raises(FloatingPointError):
        hybrid.step()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize('engine', ['dls', 'async'])
def test_streaming_resume_accumulation_and_overfit(engine, tmp_path):
    if engine == 'async' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    device = 'cuda' if engine == 'async' else 'cpu'
    cfg = MorphConfig(vocab_size=8, d_model=8, num_layers=1, d_ff=16, d_state=2, head_dim=4)
    ids = torch.tensor([[1, 2, 3, 1, 2, 3]])
    kwargs = dict(cfg=cfg, tokens=ids, seed=8, lr=.02, engine=engine, device=device, accumulation=2)
    uninterrupted = train_run(**kwargs, steps=30)
    assert uninterrupted['losses'][-1] < uninterrupted['losses'][0] * .6
    checkpoint = tmp_path / 'resume.pt'
    train_run(**kwargs, steps=12, checkpoint=checkpoint)
    resumed = train_run(**kwargs, steps=30, resume=checkpoint)
    assert uninterrupted['losses'] == resumed['losses']
    for p, q in zip(uninterrupted['model'].parameters(), resumed['model'].parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    with pytest.raises(ValueError, match='settings differ'):
        train_run(**{**kwargs, 'tokens': ids + 1}, steps=30, resume=checkpoint)


def test_muon_diagonal_matches_scalar_newton_schulz_reference():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False), torch.nn.LayerNorm(2))
    optimizer = HybridMuonAdamW(model, lr=.02)
    weight = model[0].weight
    before = weight.detach().clone()
    weight.grad = torch.diag(torch.tensor([1., 2.]))
    # Nesterov first update = .0975 * grad; normalize, then evaluate
    # the quintic independently on diagonal singular values.
    diagonal = torch.tensor([.0975, .195], dtype=torch.float64)
    diagonal = diagonal / (diagonal.norm() + 1e-7)
    for _ in range(5):
        diagonal = 3.4445 * diagonal - 4.775 * diagonal ** 3 + 2.0315 * diagonal ** 5
    optimizer.step()
    expected = before * (1 - .02 * .01) - .02 * torch.diag(diagonal.float())
    torch.testing.assert_close(weight, expected, rtol=1e-5, atol=1e-7)
