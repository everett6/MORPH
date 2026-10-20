import torch
from morph.model.fast_memory import GraphMemory


def test_memory_is_causal_and_sequence_local():
    torch.manual_seed(3)
    model = GraphMemory(8, 4)
    x = torch.randn(2, 5, 8)
    y, state = model(x)
    changed = x.clone(); changed[:, 3:] += 10
    torch.testing.assert_close(y[:, :3], model(changed)[0][:, :3])
    torch.testing.assert_close(y, model(x)[0])
    assert state.matrix.dtype == torch.float32
    assert state.version == 5
    # Streaming preserves the same causal recurrence.
    first, s = model(x[:, :2]); last, _ = model(x[:, 2:], s)
    torch.testing.assert_close(y, torch.cat((first, last), 1))


def test_memory_training_reaches_write_gates_and_prior_tokens():
    torch.manual_seed(4)
    model = GraphMemory(8, 4)
    x = torch.randn(1, 4, 8, requires_grad=True)
    y, _ = model(x)
    y[:, -1].square().sum().backward()
    for name in ('key', 'value', 'query', 'read', 'gates'):
        assert getattr(model, name).weight.grad.abs().sum() > 0
    assert x.grad[:, 0].abs().sum() > 0


def test_branch_rollback_and_graph_conditioning():
    model = GraphMemory(8, 4)
    x = torch.randn(1, 3, 8)
    _, state = model(x)
    saved = state.snapshot(); branch = state.clone()
    _, updated = model(x, branch)
    assert state.version == 3 and updated.version == 6
    updated.rollback(saved)
    torch.testing.assert_close(updated.matrix, state.matrix)
    updated.reset()
    assert updated.version > saved.version
    assert updated.matrix.count_nonzero() == 0
    assert not torch.allclose(model(x)[0], model(x, graph_context=torch.ones(1, 8))[0])


def test_autocast_keeps_fp32_state_and_first_read_precedes_write():
    model = GraphMemory(8, 4)
    x = torch.randn(1, 2, 8)
    # Empty memory contributes zero to first output, regardless of write target.
    first, _ = model(x)
    torch.testing.assert_close(first[:, 0], x[:, 0])
    with torch.autocast('cpu', dtype=torch.bfloat16):
        y, state = model(x)
    assert state.matrix.dtype == torch.float32
    assert torch.isfinite(y).all()
