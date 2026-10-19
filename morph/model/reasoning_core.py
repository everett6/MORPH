"""Shared post-backbone graph refinement and supervised proof-search heads."""
import torch
from torch import nn
from .graph import aggregate_hyperedges


class ReasoningCore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = cfg.d_graph
        self.width, self.steps = width, cfg.reasoning_steps
        self.node_projection = nn.Linear(cfg.d_model, width)
        self.context = nn.Linear(cfg.d_model, width)
        self.node_types = nn.Embedding(9, width)
        self.statuses = nn.Embedding(5, width)
        self.relations = nn.Embedding(7, width)
        self.gate = nn.Linear(width*2, width)
        self.update = nn.Linear(width*2, width)
        self.action = nn.Linear(width, 11)
        self.target = nn.Linear(width, 11)
        self.value = nn.Linear(width, 1)
        self.dependency_source = nn.Linear(width, width)
        self.dependency_target = nn.Linear(width, width)
        self.dependency = nn.Linear(width, 8)

    def forward(self, hidden, node_embeddings, graph):
        active = graph['active']
        state = self.node_projection(node_embeddings) + self.node_types(graph['node_types']) + self.statuses(graph['statuses'])
        state = state * active[...,None]
        text_mask=graph.get('text_mask',torch.ones(hidden.shape[:2],device=hidden.device,dtype=torch.bool))
        pooled_hidden=(hidden*text_mask[...,None]).sum(1)/text_mask.sum(1,keepdim=True).clamp_min(1)
        context = self.context(pooled_hidden)[:,None]
        iterations = []
        for _ in range(self.steps):
            message = aggregate_hyperedges(state, graph['sources'], graph['targets'], graph['source_mask'], self.relations(graph['relations']))
            combined = torch.cat((state, message+context), -1)
            gate = self.gate(combined).sigmoid()
            state = ((1-gate)*state + gate*self.update(combined).tanh()) * active[...,None]
            iterations.append(state)
        pooled = state.sum(1)/active.sum(1,keepdim=True).clamp_min(1)
        pair = self.dependency_source(state)[:,:,None] + self.dependency_target(state)[:,None,:]
        return dict(action_logits=self.action(pooled).masked_fill(~graph['action_mask'], -1e4),
                    target_logits=self.target(state).transpose(1,2).masked_fill(~graph['target_action_mask'],-1e4),
                    value=self.value(pooled).squeeze(-1), dependency_logits=self.dependency(pair.tanh()),
                    node_states=state, iterations=iterations)
