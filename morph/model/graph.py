"""Fixed-topology, typed hypergraph exchange with causal token updates.

Default neural memory topology is a directed ring of DEPENDS_ON edges. It is
not a mathematical proof graph. Symbolic statements, discrete actions, and
formal verification remain separate future subsystems.
"""
import torch
from torch import nn


def aggregate_hyperedges(nodes, sources, targets, source_mask, relation):
    """Masked source mean plus relation vectors, then mean at destinations.

    sources: [B,E,K], targets: [B,E], source_mask: [B,E,K]. Masked source
    entries may use -1 padding. An edge with no active source contributes zero.
    Indices for active sources and all destinations must be within capacity.
    """
    batch, capacity, width = nodes.shape
    safe = sources.masked_fill(~source_mask, 0)
    gathered = nodes[torch.arange(batch, device=nodes.device)[:, None, None], safe]
    count = source_mask.sum(-1, keepdim=True)
    valid = count > 0
    message = (gathered * source_mask[..., None]).sum(-2) / count.clamp_min(1)
    message = (message + relation) * valid
    result = torch.zeros_like(nodes).scatter_add(1, targets[..., None].expand(-1, -1, width), message)
    degree = nodes.new_zeros(batch, capacity, 1).scatter_add(1, targets[..., None], valid.to(nodes.dtype))
    return result / degree.clamp_min(1)


class GraphExchange(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.steps = cfg.reasoning_steps
        self.capacity = cfg.graph_nodes
        self.width = cfg.d_graph
        self.seed = nn.Parameter(torch.randn(cfg.graph_nodes, cfg.d_graph) * .02)
        self.relation = nn.Parameter(torch.randn(1, cfg.d_graph) * .02)
        self.query = nn.Linear(cfg.d_model, cfg.d_graph, bias=False)
        self.gate = nn.Linear(2 * cfg.d_graph, cfg.d_graph)
        self.update = nn.Linear(2 * cfg.d_graph, cfg.d_graph)
        self.output = nn.Linear(cfg.d_graph, cfg.d_model, bias=False)
        self.register_buffer('sources', torch.arange(cfg.graph_nodes).view(1, -1, 1))
        self.register_buffer('targets', torch.arange(cfg.graph_nodes).roll(-1).view(1, -1))

    def forward(self, x, state=None, memory=None):
        batch = x.shape[0]
        if state is None:
            state = self.seed[None].expand(batch, -1, -1).to(x.dtype)
        if state.shape != (batch, self.capacity, self.width):
            raise ValueError('graph memory shape mismatch')
        sources = self.sources.expand(batch, -1, -1)
        targets = self.targets.expand(batch, -1)
        mask = torch.ones_like(sources, dtype=torch.bool)
        outputs, history = [], []
        for t, token in enumerate(x.unbind(1)):
            query = self.query(token)
            prior = 0 if memory is None else memory[:, t]
            for _ in range(self.steps):
                message = aggregate_hyperedges(state, sources, targets, mask,
                                               self.relation[None].expand(batch, self.capacity, -1))
                combined = torch.cat((state, message + query[:, None] + prior), -1)
                gate = self.gate(combined).sigmoid()
                state = (1 - gate) * state + gate * self.update(combined).tanh()
            history.append(state)
            weights = (state * query[:, None]).sum(-1).div(self.width ** .5).softmax(-1)
            outputs.append(token + self.output((state * weights[..., None]).sum(1)))
        return torch.stack(outputs, 1), state, torch.stack(history, 1)
