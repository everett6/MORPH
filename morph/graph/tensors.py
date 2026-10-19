"""Deterministic padded graph batches; symbolic metadata stays outside tensors."""
import torch
from .schema import Status
from .controller import valid_actions


def tensorize(graphs, capacity, statement_length=64, device='cpu'):
    batch = len(graphs)
    max_edges = max(1, max(len(g.edges) for g in graphs))
    max_sources = max(1, max((len(e.sources) for g in graphs for e in g.edges), default=1))
    result = dict(node_types=torch.zeros(batch, capacity, dtype=torch.long),
                  statuses=torch.zeros(batch, capacity, dtype=torch.long),
                  active=torch.zeros(batch, capacity, dtype=torch.bool),
                  tokens=torch.zeros(batch, capacity, statement_length, dtype=torch.long),
                  token_mask=torch.zeros(batch, capacity, statement_length, dtype=torch.bool),
                  sources=torch.zeros(batch, max_edges, max_sources, dtype=torch.long),
                  targets=torch.zeros(batch, max_edges, dtype=torch.long),
                  relations=torch.zeros(batch, max_edges, dtype=torch.long),
                  source_mask=torch.zeros(batch, max_edges, max_sources, dtype=torch.bool),
                  dependency_labels=torch.zeros(batch, capacity, capacity, dtype=torch.long),
                  action_mask=torch.zeros(batch, 11, dtype=torch.bool),
                  target_action_mask=torch.zeros(batch,11,capacity,dtype=torch.bool))
    mappings = []
    for row, graph in enumerate(graphs):
        order = sorted(graph.nodes)
        if len(order) > capacity:
            raise ValueError('proof graph exceeds configured capacity')
        mapping = {node: i for i, node in enumerate(order)}
        mappings.append(mapping)
        for node_id, index in mapping.items():
            node = graph.nodes[node_id]
            result['node_types'][row,index] = node.node_type
            result['statuses'][row,index] = node.status
            result['active'][row,index] = node.status not in (Status.BLOCKED, Status.REFUTED)
            encoded = list(node.statement.encode('utf8'))[:statement_length]
            result['tokens'][row,index,:len(encoded)] = torch.tensor(encoded)
            result['token_mask'][row,index,:len(encoded)] = True
        if not result['active'][row].any():
            raise ValueError('at least one active node required')
        for e, edge in enumerate(graph.edges):
            result['targets'][row,e] = mapping[edge.target]
            result['relations'][row,e] = edge.relation
            for k, source in enumerate(edge.sources):
                result['sources'][row,e,k] = mapping[source]
                result['source_mask'][row,e,k] = (result['active'][row,mapping[source]] and result['active'][row,mapping[edge.target]])
                result['dependency_labels'][row,mapping[source],mapping[edge.target]] = int(edge.relation)+1
        for node_id,index in mapping.items():
            if result['active'][row,index]:
                result['target_action_mask'][row,:,index]=torch.tensor(valid_actions(graph,node_id))
        result['action_mask'][row]=result['target_action_mask'][row].any(-1)
    return {key: value.to(device) for key,value in result.items()}, mappings
