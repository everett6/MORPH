"""Exact bounded local inference with conservative, versioned cache invalidation.

Edges are (source, target, relation-type). Training always executes the full graph,
retaining the actual gradient paths. This module does not confer symbolic trust.
"""
import torch
from torch import nn


class IncrementalGraph(nn.Module):
    def __init__(self, dim, iterations=4, num_relations=7):
        super().__init__()
        if iterations < 1:
            raise ValueError('iterations must be positive')
        self.dim, self.iterations = dim, iterations
        self.update = nn.Linear(dim, dim)
        self.relations = nn.Embedding(num_relations, dim)
        self._cache = None
        self.last_recomputed = set()

    def _validate(self, features, edges):
        for feature in features.values():
            if feature.shape != (self.dim,):
                raise ValueError('each node feature must have shape [dim]')
        for source, target, relation in edges:
            if source not in features or target not in features:
                raise ValueError('edge endpoint missing')
            if not 0 <= int(relation) < self.relations.num_embeddings:
                raise ValueError('unknown relation type')

    def _node(self, node, previous, features, edges, context):
        value = features[node]
        if context is not None:
            value = value + context
        for source, target, relation in edges:
            if target == node:
                value = value + previous[source] + self.relations.weight[int(relation)]
        return torch.tanh(self.update(value))

    def full(self, features, edges, context=None):
        self._validate(features, edges)
        previous = features
        for _ in range(self.iterations):
            previous = {node: self._node(node, previous, features, edges, context) for node in features}
        return previous

    def forward(self, features, edges, *, weight_version=0, memory_version=0, context=None):
        edges = tuple(sorted(tuple(edge) for edge in edges))
        self._validate(features, edges)
        # Parameter _version detects normal optimizer/in-place changes in addition
        # to the explicit caller epoch. No detached reuse under autograd.
        signature = (weight_version, memory_version, tuple(
            (id(p), p._version, p.device, p.dtype, tuple(p.shape)) for p in self.parameters()))
        if self.training or torch.is_grad_enabled():
            self.last_recomputed = set(features)
            self._cache = None
            return self.full(features, edges, context)
        cached = self._cache
        invalidate = cached is None or cached['signature'] != signature
        if cached is not None:
            old_context = cached['context']
            invalidate |= (context is None) != (old_context is None)
            if context is not None and old_context is not None:
                invalidate |= (context.device != old_context.device or context.dtype != old_context.dtype
                               or context.shape != old_context.shape or not torch.equal(context, old_context))
        changed = set(features) if invalidate else {
            n for n in features if n not in cached['features']
            or features[n].device != cached['features'][n].device
            or features[n].dtype != cached['features'][n].dtype
            or not torch.equal(features[n], cached['features'][n])}
        if not invalidate:
            changed_edges = set(edges).symmetric_difference(cached['edges'])
            changed.update(target for _, target, _ in changed_edges if target in features)
        previous = features
        layers = [dict(features)]
        affected = changed.copy()
        self.last_recomputed = set()
        for layer in range(1, self.iterations + 1):
            # Exact dependency closure in the bounded unrolled computation.
            affected = changed | {target for source, target, _ in edges if source in affected}
            affected &= features.keys()
            current = {}
            for node in features:
                if node in affected:
                    current[node] = self._node(node, previous, features, edges, context)
                    self.last_recomputed.add(node)
                else:
                    current[node] = cached['layers'][layer][node]
            previous = current
            layers.append(current)
        self._cache = dict(signature=signature, features={n: v.detach().clone() for n, v in features.items()},
                           edges=set(edges), layers=[{n: v.detach().clone() for n, v in layer.items()} for layer in layers],
                           context=None if context is None else context.detach().clone())
        return {n: v.clone() for n, v in previous.items()}
