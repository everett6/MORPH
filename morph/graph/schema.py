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
            if any(i not in graph.nodes for i in (*edge.sources, edge.target)):
                raise ValueError('invalid restored edge')
        return graph

    def digest(self):
        return hashlib.sha256(json.dumps(self.snapshot(), sort_keys=True).encode()).hexdigest()
