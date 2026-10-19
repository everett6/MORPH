"""Symbolic proof obligations; neural confidence never changes trust status."""
from dataclasses import asdict, dataclass, field
from enum import IntEnum
import copy
import hashlib
import json


class NodeType(IntEnum):
    GOAL=0; ASSUMPTION=1; LEMMA=2; CONJECTURE=3; PROOF_STEP=4
    COUNTEREXAMPLE=5; CONSTRUCTION=6; COMPUTATION=7; VERIFIED_THEOREM=8


class Status(IntEnum):
    OPEN=0; PROPOSED=1; VERIFIED=2; REFUTED=3; BLOCKED=4


class Relation(IntEnum):
    REQUIRES=0; IMPLIES=1; EQUIVALENT_TO=2; SPECIALIZES=3
    CONTRADICTS=4; DEPENDS_ON_ASSUMPTION=5; PROVED_BY=6


class ActionType(IntEnum):
    CREATE_LEMMA=0; DECOMPOSE=1; EXPAND=2; REVISE=3; CONNECT=4
    MERGE=5; PRUNE=6; RETRIEVE=7; TEST=8; VERIFY=9; STOP=10


@dataclass
class ProofNode:
    node_id: int
    statement: str
    node_type: NodeType = NodeType.GOAL
    status: Status = Status.OPEN
    formal_statement: str | None = None
    assumptions: tuple[str, ...] = ()
    provenance: dict = field(default_factory=dict)
    proof_hash: str | None = None

    def __post_init__(self):
        self.node_type = NodeType[self.node_type] if isinstance(self.node_type, str) else NodeType(self.node_type)
        self.status = Status[self.status] if isinstance(self.status, str) else Status(self.status)
        self.assumptions = tuple(self.assumptions)
        if not self.statement.strip() or self.node_id < 0:
            raise ValueError('node requires statement and nonnegative ID')


@dataclass(frozen=True)
class Hyperedge:
    sources: tuple[int, ...]
    target: int
    relation: Relation
    certified: bool = False
    proof_hash: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'sources', tuple(self.sources))
        object.__setattr__(self, 'relation', Relation[self.relation] if isinstance(self.relation, str) else Relation(self.relation))
        if not self.sources or len(set(self.sources)) != len(self.sources):
            raise ValueError('hyperedge needs unique nonempty premises')


@dataclass
class ProofGraph:
    nodes: dict[int, ProofNode] = field(default_factory=dict)
    edges: list[Hyperedge] = field(default_factory=list)
    stopped: bool = False

    def add(self, node):
        if node.node_id in self.nodes:
            raise ValueError('duplicate node ID')
        if node.status == Status.VERIFIED or node.node_type == NodeType.VERIFIED_THEOREM:
            raise ValueError('only checker evidence can establish verification')
        self.nodes[node.node_id] = copy.deepcopy(node)

    def connect(self, edge):
        if edge.certified or edge.proof_hash:
            raise ValueError('certified dependencies require the checker certification interface')
        if any(index not in self.nodes for index in (*edge.sources, edge.target)):
            raise ValueError('dependency refers to missing node')
        if edge not in self.edges:
            self.edges.append(edge)

    def accept_verification(self, node_id, result):
        node = self.nodes[node_id]
        if not result.success or result.statement != node.formal_statement or tuple(result.assumptions) != node.assumptions:
            raise ValueError('checker outcome does not match exact statement and assumptions')
        if not result.proof_hash or result.checker != 'lean4':
            raise ValueError('formal checker provenance required')
        node.status = Status.VERIFIED
        node.proof_hash = result.proof_hash
        node.provenance = {**node.provenance, 'checker': result.checker, 'version': result.version}

    def snapshot(self):
        return dict(nodes=[asdict(self.nodes[i]) for i in sorted(self.nodes)],
                    edges=[asdict(e) for e in self.edges], stopped=self.stopped)

    @classmethod
    def restore(cls, record, evidence=None):
        graph = cls()
        for data in record['nodes']:
            node=ProofNode(**data)
            if node.node_id in graph.nodes:
                raise ValueError('duplicate restored node ID')
            if node.status == Status.VERIFIED or node.node_type == NodeType.VERIFIED_THEOREM:
                result=(evidence or {}).get(node.node_id)
                if result is None:
                    raise ValueError('restoring verified nodes requires checker evidence')
                graph.nodes[node.node_id]=node
                graph.accept_verification(node.node_id,result)
            else:
                graph.nodes[node.node_id]=node
        graph.edges = [Hyperedge(**edge) for edge in record['edges']]
        graph.stopped = record.get('stopped', False)
        for edge in graph.edges:
            if edge.certified and (not (evidence or {}).get(edge.target) or edge.proof_hash!=evidence[edge.target].proof_hash):
                raise ValueError('certified edge restoration requires matching checker evidence')
            if any(i not in graph.nodes for i in (*edge.sources, edge.target)):
                raise ValueError('invalid restored edge')
        return graph

    def digest(self):
        return hashlib.sha256(json.dumps(self.snapshot(), sort_keys=True).encode()).hexdigest()


def _certification_cycle(graph,extra):
    dependencies={}
    for edge in [*graph.edges,extra]:
        if edge.certified or edge is extra:
            dependencies.setdefault(edge.target,[]).extend(edge.sources)
    visiting,finished=set(),set()
    def visit(node):
        if node in visiting:return True
        if node in finished:return False
        visiting.add(node)
        if any(visit(source) for source in dependencies.get(node,[])):return True
        visiting.remove(node);finished.add(node);return False
    return any(visit(node) for node in dependencies)


def certify_edge(graph,index,result):
    from dataclasses import replace
    edge=graph.edges[index];target=graph.nodes[edge.target]
    if any(graph.nodes[source].status!=Status.VERIFIED for source in edge.sources):
        raise ValueError('all inference premises need independent checker evidence')
    if any(not set(graph.nodes[source].assumptions).issubset(target.assumptions) for source in edge.sources):
        raise ValueError('premise assumptions exceed conclusion assumptions')
    if _certification_cycle(graph,edge):raise ValueError('circular certified justification')
    # A receipt for the exact conclusion is mandatory; verified sources alone
    # never turn a speculative dependency into a mathematical proof.
    graph.accept_verification(edge.target,result)
    graph.edges[index]=replace(edge,certified=True,proof_hash=result.proof_hash)


def obligation_closed(graph,node_id):
    visiting=set()
    def closed(index):
        node=graph.nodes[index]
        if index in visiting or node.status!=Status.VERIFIED or not node.proof_hash:return False
        visiting.add(index)
        for edge in graph.edges:
            if edge.target!=index or edge.relation not in (Relation.REQUIRES,Relation.PROVED_BY):continue
            if not edge.certified or not edge.proof_hash:return False
            if any(not set(graph.nodes[source].assumptions).issubset(node.assumptions) or not closed(source) for source in edge.sources):return False
        visiting.remove(index);return True
    return closed(node_id)
