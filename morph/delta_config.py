"""Native sparse MORPH dimensions, separately versioned from dense MORPH."""
from dataclasses import asdict,dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class DeltaConfig:
    model_type:str='delta'
    vocab_size:int=256
    d_model:int=1536
    num_layers:int=10
    d_ff:int=3072
    expand:int=2
    d_state:int=64
    head_dim:int=64
    ssm_backend:str='parallel'
    scan_chunk_size:int=32
    graph_enabled:bool=False  # Legacy token-by-token neural ring is not used.
    graph_update:bool=True
    graph_interval:int=4
    d_graph:int=256
    graph_nodes:int=16
    reasoning_steps:int=4
    proof_enabled:bool=True
    statement_field_length:int=128
    fast_memory_enabled:bool=True
    memory_dim:int=32
    memory_projection_backend:str='reference'
    expert_group_size:int=1
    expert_count:int=64
    expert_depth:int=8
    expert_ffn:int=3840
    latent_dim:int=768
    latent_experts:bool=True
    step_expert_count:int=0
    routing_seed:int=0
    top_k:int=1
    router_aux_weight:float=.01
    norm_eps:float=1e-6

    def __post_init__(self):
        if self.model_type!='delta':raise ValueError('DeltaConfig model_type must be delta')
        for key,value in asdict(self).items():
            if key in ('model_type','memory_projection_backend','ssm_backend','router_aux_weight','norm_eps','step_expert_count','routing_seed') or type(value) is bool:continue
            if type(value) is not int or value<1:raise ValueError(f'{key} must be a positive integer')
        if type(self.step_expert_count) is not int or not 0<=self.step_expert_count<=self.expert_count:raise ValueError('step_expert_count must be between zero and expert_count')
        if type(self.routing_seed) is not int or not 0<=self.routing_seed<2**63:raise ValueError('routing_seed must be a nonnegative 63-bit integer')
        if self.vocab_size!=256:raise ValueError('Delta currently uses the fixed 256-byte vocabulary')
        if self.top_k!=1:raise ValueError('only dropless top-1 routing is implemented')
        if self.graph_enabled:raise ValueError('Delta uses typed graph/session memory, not legacy ring exchange')
        if self.d_model*self.expand%self.head_dim:raise ValueError('expanded width must be divisible by head_dim')
        if self.memory_projection_backend not in ('reference','batched'):raise ValueError('unsupported memory projection backend')
        if self.expert_group_size>self.expert_count:raise ValueError('expert_group_size exceeds expert_count')
        if self.ssm_backend not in ('reference','parallel','triton'):raise ValueError('unsupported SSM backend')
        if self.norm_eps<=0 or self.router_aux_weight<0:raise ValueError('invalid loss/norm coefficient')
        if self.latent_dim>self.d_model:raise ValueError('latent expert width exceeds shared width')

    @classmethod
    def from_file(cls,path):return cls(**json.loads(Path(path).read_text()))
    def to_dict(self):return asdict(self)
