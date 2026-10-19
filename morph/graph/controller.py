"""Deterministic external topology actions and replay, separate from autograd."""
import copy
from .schema import ActionType, Hyperedge, NodeType, ProofNode, Relation, Status


def valid_actions(graph, target):
    result = [True] * len(ActionType)
    if graph.stopped:
        return [False] * (len(ActionType)-1) + [True]
    node = graph.nodes.get(target)
    if node is None or node.status in (Status.BLOCKED, Status.REFUTED, Status.VERIFIED):
        for action in (ActionType.EXPAND, ActionType.REVISE, ActionType.DECOMPOSE, ActionType.VERIFY, ActionType.PRUNE):
            result[action] = False
    if node is None or node.formal_statement is None:
        result[ActionType.VERIFY] = False
    return result


def apply_action(graph, action):
    before = graph.digest()
    kind = ActionType[action['type']] if isinstance(action['type'], str) else ActionType(action['type'])
    target = action.get('target', -1)
    if not valid_actions(graph, target)[kind]:
        raise ValueError('action is invalid for proof state')
    work = copy.deepcopy(graph)
    if kind in (ActionType.CREATE_LEMMA, ActionType.DECOMPOSE, ActionType.RETRIEVE):
        children = action.get('nodes', [])
        if not children:
            raise ValueError('create/decompose/retrieve require explicit nodes')
        for data in children:
            node = ProofNode(**data)
            if kind == ActionType.DECOMPOSE and not set(work.nodes[target].assumptions).issubset(node.assumptions):
                raise ValueError('decomposition dropped assumptions')
            work.add(node)
        if kind == ActionType.DECOMPOSE:
            work.connect(Hyperedge(tuple(data['node_id'] for data in children),target,Relation.REQUIRES))
            for edge in action.get('edges',[]):work.connect(Hyperedge(**edge))
    elif kind == ActionType.REVISE:
        node = work.nodes[target]
        if any(target in (*edge.sources,edge.target) for edge in work.edges):
            raise ValueError('revision would invalidate existing dependencies')
        if 'assumptions' in action and not set(node.assumptions).issubset(action['assumptions']):
            raise ValueError('revision dropped assumptions')
        if not action['statement'].strip():
            raise ValueError('empty revision')
        node.assumptions=tuple(action.get('assumptions',node.assumptions))
        node.statement = action['statement']
        node.formal_statement = action.get('formal_statement')
        node.status, node.proof_hash = Status.PROPOSED, None
    elif kind in (ActionType.CONNECT, ActionType.MERGE):
        work.connect(Hyperedge(**action['edge']))
    elif kind == ActionType.PRUNE:
        work.nodes[target].status = Status.BLOCKED
    elif kind == ActionType.STOP:
        work.stopped = True
    # VERIFY and TEST request external work; neither promotes node trust.
    graph.nodes, graph.edges, graph.stopped = work.nodes, work.edges, work.stopped
    return dict(before=before, action=copy.deepcopy(action), after=graph.digest())


def replay(initial, records):
    graph = copy.deepcopy(initial)
    for record in records:
        if graph.digest() != record['before']:
            raise ValueError('replay boundary mismatch')
        actual = apply_action(graph, record['action'])
        if actual['after'] != record['after']:
            raise ValueError('replay outcome mismatch')
    return graph
