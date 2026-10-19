"""Serializable architecture dimensions; no parameter allocations."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class MorphConfig:
    vocab_size: int = 256
    d_model: int = 32
    num_layers: int = 2
    d_ff: int = 64
    expand: int = 2
    d_state: int = 8
    head_dim: int = 16
    ssm_backend: str = 'parallel'
    scan_chunk_size: int = 32
    proof_enabled: bool = False
    graph_enabled: bool = False
    graph_update: bool = True
    graph_interval: int = 4
    d_graph: int = 128
    graph_nodes: int = 16
    reasoning_steps: int = 4
    norm_eps: float = 1e-6

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key not in ('norm_eps', 'graph_enabled', 'proof_enabled', 'ssm_backend', 'graph_update') and (type(value) is not int or value <= 0):
                raise ValueError(f'{key} must be a positive integer')
        if self.ssm_backend not in ('reference', 'parallel'):
            raise ValueError('unsupported SSM backend')
        if self.d_model * self.expand % self.head_dim:
            raise ValueError('expanded width must be divisible by head_dim')
        if self.norm_eps <= 0:
            raise ValueError('norm_eps must be positive')

    @classmethod
    def from_file(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def to_dict(self):
        return asdict(self)
