import torch
from morph import MorphConfig, MorphModel
from morph.model.graph import aggregate_hyperedges


def test_masked_typed_hyperedges():
    nodes = torch.tensor([[[2., 4.], [6., 8.], [99., 99.]]], requires_grad=True)
    sources = torch.tensor([[[0, 1], [2, 0]]])
    targets = torch.tensor([[1, 0]])
    mask = torch.tensor([[[True, True], [False, False]]])
    relation = torch.tensor([[[1., 1.], [100., 100.]]], requires_grad=True)
    got = aggregate_hyperedges(nodes, sources, targets, mask, relation)
    torch.testing.assert_close(got, torch.tensor([[[0., 0.], [5., 7.], [0., 0.]]]))
    got.sum().backward()
    torch.testing.assert_close(nodes.grad, torch.tensor([[[.5, .5], [.5, .5], [0., 0.]]]))
    torch.testing.assert_close(relation.grad, torch.tensor([[[1., 1.], [0., 0.]]]))


def test_causal_graph_shared_recurrence_chunk_and_backward():
    cfg = MorphConfig(vocab_size=16, d_model=8, num_layers=2, d_ff=16, expand=2,
                      head_dim=4, d_state=2, graph_enabled=True, graph_interval=1,
                      d_graph=4, graph_nodes=3, reasoning_steps=4)
    model = MorphModel(cfg).double()
    ids = torch.tensor([[1, 2, 3, 4]])
    full = model(ids)
    first = model(ids[:, :2]); last = model(ids[:, 2:], first.states)
    torch.testing.assert_close(full.logits, torch.cat((first.logits, last.logits), 1))
    changed = model(torch.tensor([[1, 2, 9, 9]]))
    torch.testing.assert_close(full.logits[:, :2], changed.logits[:, :2])
    full.logits.sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
    graph = model.blocks[0].graph
    assert graph.update.weight.grad.abs().sum() > 0
