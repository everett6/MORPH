import torch
from morph.graph.incremental import IncrementalGraph


def fixture():
    return IncrementalGraph(2, iterations=3), {0: torch.tensor([1., 0.]), 1: torch.tensor([0., 1.]), 2: torch.ones(2)}, [(0, 1, 0), (1, 0, 1)]


def test_incremental_matches_full_with_cycles_edits_and_deletions():
    model, features, edges = fixture(); model.eval()
    with torch.no_grad():
        model(features, edges)
        features[0] = torch.tensor([2., 3.])
        actual = model(features, edges)
        expected = model.full(features, edges)
        for key in expected: torch.testing.assert_close(actual[key], expected[key])
        assert 2 not in model.last_recomputed
        edges.pop(); features[3] = torch.zeros(2); edges.append((1, 3, 2))
        actual = model(features, edges)
        for key, value in model.full(features, edges).items(): torch.testing.assert_close(actual[key], value)
        del features[0]; edges = [(1, 3, 2)]
        actual = model(features, edges)
        assert 0 not in actual
        for key, value in model.full(features, edges).items(): torch.testing.assert_close(actual[key], value)


def test_versions_context_and_training_invalidate_cache():
    model, features, edges = fixture(); model.eval()
    with torch.no_grad():
        model(features, edges)
        model(features, edges, memory_version=1)
        assert model.last_recomputed == set(features)
        model(features, edges, memory_version=1, context=torch.ones(2))
        assert model.last_recomputed == set(features)
        model.update.weight.add_(1)
        actual = model(features, edges, memory_version=1, context=torch.ones(2))
        for k, v in model.full(features, edges, context=torch.ones(2)).items(): torch.testing.assert_close(actual[k], v)
        assert model.last_recomputed == set(features)
    model.train()
    features[0].requires_grad_()
    model(features, edges)[1].sum().backward()
    assert features[0].grad.abs().sum() > 0
    assert model.last_recomputed == set(features)


def test_training_gradients_match_full_reference_after_cached_inference():
    model, features, edges = fixture(); model.eval()
    with torch.no_grad(): model(features, edges)
    model.train()
    actual = model(features, edges)[1].sum()
    actual.backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    model.full(features, edges)[1].sum().backward()
    for p, expected in zip(model.parameters(), grads):
        torch.testing.assert_close(p.grad, expected)


def test_parameter_replacement_and_dtype_migration_invalidate_all_nodes():
    model, features, edges = fixture(); model.eval()
    with torch.no_grad():
        model(features, edges)
        # Replacement can retain the same tensor mutation counter.
        old_version=model.update.weight._version
        replacement=torch.nn.Parameter(model.update.weight.detach().clone()+2)
        while replacement._version < old_version: replacement.add_(0)
        assert replacement._version == old_version
        model.update.weight=replacement
        actual=model(features,edges)
        assert model.last_recomputed == set(features)
        for key,value in model.full(features,edges).items(): torch.testing.assert_close(actual[key],value)
        model.double();features={key:value.double() for key,value in features.items()}
        actual=model(features,edges)
        assert model.last_recomputed == set(features)
        assert all(value.dtype==torch.float64 for value in actual.values())
        for key,value in model.full(features,edges).items(): torch.testing.assert_close(actual[key],value)


def test_equal_valued_context_and_feature_dtype_changes_invalidate_dependencies():
    model,features,edges=fixture();model.eval()
    with torch.no_grad():
        model(features,edges,context=torch.zeros(2))
        features[0]=features[0].half()
        actual=model(features,edges,context=torch.zeros(2))
        assert 0 in model.last_recomputed and 2 not in model.last_recomputed
        for key,value in model.full(features,edges,context=torch.zeros(2)).items(): torch.testing.assert_close(actual[key],value)
        model(features,edges,context=torch.zeros(2,dtype=torch.float16))
        assert model.last_recomputed==set(features)
