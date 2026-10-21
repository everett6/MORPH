"""Native top-1 sparse SSM/proof model with independently loadable experts.

The optional per-expert latent bottleneck is a research variant, not an exact
implementation of NVIDIA LatentMoE's shared down/up projections. Routing is
per-token, dropless, and differentiable through selected gate probabilities.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .morph_model import RMSNorm,SwiGLU,MorphBlock


def router_balance(gates,eligible_count=None):
    routes=gates.argmax(-1)
    frequency=F.one_hot(routes,num_classes=gates.shape[-1]).float().mean((0,1))
    return (eligible_count or gates.shape[-1])*(frequency.detach()*gates.float().mean((0,1))).sum()


class SparseRouter(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.projection=nn.Linear(cfg.d_model,cfg.expert_count)
        self.eligible=None
    def forward(self,x):
        # Routing logits/probabilities remain FP32 under mixed precision.
        with torch.autocast(x.device.type,enabled=False):
            logits=F.linear(x.float(),self.projection.weight.float(),self.projection.bias.float())
            if self.eligible is not None:
                mask=torch.ones(logits.shape[-1],device=x.device,dtype=torch.bool)
                mask[list(self.eligible)]=False
                logits=logits.masked_fill(mask,float('-inf'))
            return logits.softmax(-1)


class ExpertBundle(nn.Module):
    def __init__(self,cfg):
        super().__init__();width=cfg.latent_dim if cfg.latent_experts else cfg.d_model
        self.input_norm=RMSNorm(cfg.d_model,cfg.norm_eps)
        self.down=nn.Linear(cfg.d_model,width,bias=False) if cfg.latent_experts else nn.Identity()
        self.layers=nn.ModuleList(nn.ModuleDict(dict(norm=RMSNorm(width,cfg.norm_eps),ffn=SwiGLU(width,cfg.expert_ffn))) for _ in range(cfg.expert_depth))
        self.up=nn.Linear(width,cfg.d_model,bias=False) if cfg.latent_experts else nn.Identity()
    def forward(self,x):
        hidden=self.down(self.input_norm(x))
        for layer in self.layers:hidden=hidden+layer['ffn'](layer['norm'](hidden))
        return self.up(hidden)


class GraphSessionMemory(nn.Module):
    """Graph-only external conditioning plus causal differentiable fast memory."""
    def __init__(self,cfg):
        super().__init__()
        from .fast_memory import GraphMemory
        from .proof_head import StatementEncoder
        self.memory=GraphMemory(cfg.d_model,cfg.memory_dim,cfg.memory_projection_backend)
        # Separate node embeddings avoid additional tied-LM parameter uses in
        # intermediate streaming stages. All fields remain explicitly ordered.
        self.tokens=nn.Embedding(cfg.vocab_size,cfg.d_model)
        self.encoder=StatementEncoder(cfg.d_model,cfg.statement_field_length*3)
        self.node_types=nn.Embedding(9,cfg.d_model);self.statuses=nn.Embedding(5,cfg.d_model)
        self.relations=nn.Embedding(7,cfg.d_model)
    def _features(self,graph,indices=None):
        select=lambda key:graph[key] if indices is None else graph[key].index_select(1,indices)
        nodes=self.encoder(self.tokens(select('tokens')),select('token_mask'))
        active=select('active')
        return (nodes+self.node_types(select('node_types'))+self.statuses(select('statuses')))*active[...,None]

    def _full_context(self,graph):
        from .graph import aggregate_hyperedges
        nodes=self._features(graph);active=graph['active']
        messages=aggregate_hyperedges(nodes,graph['sources'],graph['targets'],graph['source_mask'],self.relations(graph['relations']))
        result=(nodes+messages)*active[...,None]
        return result.sum(1)/active.sum(1,keepdim=True).clamp_min(1)

    def graph_context(self,graph,use_cache=True,weight_version=0,memory_version=0):
        """Exact local encoder/one-hop cache for a frozen single proof graph.

        This matches this module's actual mean-hyperedge computation; it does
        not substitute a different learned GNN. Global pooling is recomputed,
        and any downstream token state must still be replayed when it changes.
        Training/autograd and batched graphs always use the full computation.
        """
        if graph is None:return None
        active=graph['active'];self.last_encoded=set();self.last_recomputed=set()
        if self.training or torch.is_grad_enabled() or active.shape[0]!=1 or not use_cache:
            if self.training or torch.is_grad_enabled():self._context_cache=None
            self.last_encoded=self.last_recomputed=set(active[0].nonzero().flatten().tolist())
            return self._full_context(graph)
        signature=(weight_version,memory_version,str(self.tokens.weight.device),str(self.tokens.weight.dtype),
                   tuple((id(parameter),parameter._version) for parameter in self.parameters()))
        fields=('tokens','token_mask','node_types','statuses','active','sources','targets','source_mask','relations')
        observed={key:graph[key].detach().cpu().clone() for key in fields}
        cached=getattr(self,'_context_cache',None)
        invalidate=cached is None or cached['signature']!=signature or cached['features'].shape[1]!=active.shape[1]
        capacity=active.shape[1]
        changed=set(range(capacity)) if invalidate else {
            index for index in range(capacity) if any(not torch.equal(observed[key][:,index],cached['observed'][key][:,index])
                                                      for key in ('tokens','token_mask','node_types','statuses','active'))}
        def incoming(data):
            result={index:[] for index in range(capacity)}
            for edge,target in enumerate(data['targets'][0].tolist()):
                sources=tuple(data['sources'][0,edge][data['source_mask'][0,edge]].tolist())
                if sources:result[target].append((sources,int(data['relations'][0,edge])))
            return result
        edges=incoming(observed)
        affected=set(changed)
        if not invalidate:
            previous=incoming(cached['observed'])
            affected.update(index for index in range(capacity) if edges[index]!=previous[index])
        affected.update(target for target,items in edges.items() if any(any(source in changed for source in sources) for sources,_ in items))
        if invalidate:
            features=self._features(graph);result=torch.zeros_like(features)
        else:
            features=cached['features'].clone();result=cached['result'].clone()
            if changed:
                indices=torch.tensor(sorted(changed),device=features.device)
                features[:,indices]=self._features(graph,indices)
        if affected:
            from .graph import aggregate_hyperedges
            edge_mask=torch.tensor([int(target) in affected for target in observed['targets'][0]],device=features.device,dtype=torch.bool)
            messages=aggregate_hyperedges(features,graph['sources'][:,edge_mask],graph['targets'][:,edge_mask],graph['source_mask'][:,edge_mask],self.relations(graph['relations'][:,edge_mask]))
            indices=torch.tensor(sorted(affected),device=features.device)
            result[:,indices]=(features[:,indices]+messages[:,indices])*active[:,indices,None]
        self.last_encoded=changed
        self.last_recomputed=affected
        self._context_cache=dict(signature=signature,observed=observed,features=features.detach().clone(),result=result.detach().clone())
        return result.sum(1)/active.sum(1,keepdim=True).clamp_min(1)

    def forward(self,x,graph=None,state=None):
        context=self.graph_context(graph,memory_version=0 if state is None else state.version)
        return self.memory(x,state,context)



class EmbeddingStage(nn.Module):
    stage_kind='delta_embedding'
    def __init__(self,embedding):super().__init__();self.embedding=embedding
    def forward(self,x,aux=None,proof_graph=None):return self.embedding(x),aux


class SharedStage(nn.Module):
    stage_kind='delta_shared'
    def __init__(self,block):super().__init__();self.block=block
    def forward(self,x,aux=None,proof_graph=None):return self.block(x)[0],aux


class MemoryStage(nn.Module):
    stage_kind='delta_memory'
    def __init__(self,memory):super().__init__();self.module=memory
    def forward(self,x,aux=None,proof_graph=None):return self.module(x,proof_graph)[0],aux


class RouterStage(nn.Module):
    stage_kind='delta_router'
    def __init__(self,router):super().__init__();self.router=router
    def forward(self,x,aux=None,proof_graph=None):return x,self.router(x)


class ExpertStage(nn.Module):
    stage_kind='delta_expert'
    def __init__(self,expert,index):super().__init__();self.expert=expert;self.expert_index=index
    def active_for(self,x,aux):return aux is not None and bool((aux.argmax(-1)==self.expert_index).any())
    def forward(self,x,aux=None,proof_graph=None):
        if aux is None:raise ValueError('expert stage requires differentiable routing gates')
        selected=aux.argmax(-1)==self.expert_index
        if not bool(selected.any()):return x,aux
        indices=selected.flatten().nonzero().squeeze(-1)
        flat=x.reshape(-1,x.shape[-1]);chosen=flat.index_select(0,indices)
        delta=self.expert(chosen)*aux[...,self.expert_index].reshape(-1).index_select(0,indices)[:,None].to(chosen.dtype)
        # Each selected group is disjoint. Other experts never modify chosen
        # tokens, so current x here equals its original pre-expert input.
        result=flat.index_add(0,indices,delta)
        return result.reshape_as(x),aux


class GroupedExpertStage(nn.Module):
    """Pack only routed experts into padded BMMs; logical parameter names stay stable.

    Padding is bounded by group_size * total_tokens. Stacked weights are transient
    autograd tensors, so capacity planning must include one selected group copy.
    """
    stage_kind='delta_expert_group'
    def __init__(self,experts,indices):
        super().__init__();self.experts=nn.ModuleList(experts);self.expert_indices=tuple(indices)
    def active_for(self,x,aux):
        return aux is not None and any(bool((aux.argmax(-1)==index).any()) for index in self.expert_indices)
    def parameter_names_for(self,states):
        active=set()
        for _,aux in states:
            if aux is not None:active.update(aux.argmax(-1).unique().tolist())
        return {name for name,_ in self.named_parameters() if self.expert_indices[int(name.split('.')[1])] in active}
    @staticmethod
    def _linear(x,modules):
        return torch.bmm(x,torch.stack([module.weight for module in modules]).transpose(1,2))
    @staticmethod
    def _norm(x,modules):
        work=x.float() if x.dtype in (torch.float16,torch.bfloat16) else x
        return (work*torch.rsqrt(work.square().mean(-1,keepdim=True)+modules[0].eps)).to(x.dtype)*torch.stack([module.weight for module in modules])[:,None]
    def forward(self,x,aux=None,proof_graph=None):
        if aux is None:raise ValueError('expert stage requires differentiable routing gates')
        routes=aux.argmax(-1).flatten();flat=x.reshape(-1,x.shape[-1])
        selected=[(expert,index,(routes==index).nonzero().flatten()) for expert,index in zip(self.experts,self.expert_indices)]
        selected=[item for item in selected if item[2].numel()]
        if not selected:return x,aux
        experts=[item[0] for item in selected];capacity=max(item[2].numel() for item in selected)
        hidden=torch.stack([F.pad(flat.index_select(0,indices),(0,0,0,capacity-indices.numel())) for _,_,indices in selected])
        hidden=self._norm(hidden,[expert.input_norm for expert in experts])
        if isinstance(experts[0].down,nn.Linear):hidden=self._linear(hidden,[expert.down for expert in experts])
        for depth in range(len(experts[0].layers)):
            layers=[expert.layers[depth] for expert in experts]
            normed=self._norm(hidden,[layer['norm'] for layer in layers]);ffns=[layer['ffn'] for layer in layers]
            activated=F.silu(self._linear(normed,[ffn.gate for ffn in ffns]))*self._linear(normed,[ffn.up for ffn in ffns])
            hidden=hidden+self._linear(activated,[ffn.down for ffn in ffns])
        if isinstance(experts[0].up,nn.Linear):hidden=self._linear(hidden,[expert.up for expert in experts])
        indices=torch.cat([item[2] for item in selected])
        delta=torch.cat([hidden[row,:positions.numel()]*aux[...,index].reshape(-1).index_select(0,positions)[:,None].to(hidden.dtype)
                         for row,(_,index,positions) in enumerate(selected)])
        return flat.index_add(0,indices,delta).reshape_as(x),aux


class HeadStage(nn.Module):
    stage_kind='delta_head'
    def __init__(self,model):
        super().__init__();self.norm=model.norm;self.weight=model.embedding.weight
        self.proof=model.proof_head if model.config.proof_enabled else None
        self.aux_weight=model.config.router_aux_weight
        self.eligible_count=model.config.step_expert_count or model.config.expert_count
    def forward(self,x,aux=None,proof_graph=None):
        if self.proof is not None and proof_graph is not None:output=self.proof(x,None,proof_graph)
        else:output=dict(logits=F.linear(self.norm(x),self.weight))
        balance=router_balance(aux,self.eligible_count) if aux is not None else x.new_zeros(())
        return dict(**output,lm_logits=output['logits'],router_balance_loss=balance,router_aux_loss=balance*self.aux_weight,
                    routes=None if aux is None else aux.argmax(-1),states=None)


class DeltaOutput(dict):
    def __getattr__(self,key):
        try:return self[key]
        except KeyError:raise AttributeError(key) from None


class DeltaModel(nn.Module):
    supports_cache=False
    def __init__(self,cfg):
        super().__init__();self.config=cfg
        self.embedding=nn.Embedding(cfg.vocab_size,cfg.d_model)
        self.blocks=nn.ModuleList(MorphBlock(cfg,False) for _ in range(cfg.num_layers))
        self.session_memory=GraphSessionMemory(cfg) if cfg.fast_memory_enabled else None
        self.router=SparseRouter(cfg);self.experts=nn.ModuleList(ExpertBundle(cfg) for _ in range(cfg.expert_count))
        self.norm=RMSNorm(cfg.d_model,cfg.norm_eps)
        self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False);self.lm_head.weight=self.embedding.weight
        if cfg.proof_enabled:
            from .reasoning_core import ReasoningCore
            from .proof_head import ProofHead
            self.reasoning=ReasoningCore(cfg);self.proof_head=ProofHead(self)
        self.apply(self._initialize)
        self.set_routing_step(0)
    def _routing_plan(self,step):
        if type(step) is not int or step<0:raise ValueError('routing step/session id must be a nonnegative integer')
        cfg=self.config;count=cfg.step_expert_count or cfg.expert_count
        generator=torch.Generator(device='cpu').manual_seed(cfg.routing_seed)
        # Explicit CPU construction also works while model parameters are meta.
        permutation=torch.randperm(cfg.expert_count,generator=generator,device='cpu').tolist()
        experts=[permutation[(step*count+i)%cfg.expert_count] for i in range(count)]
        return dict(version=1,step=step,seed=cfg.routing_seed,expert_count=cfg.expert_count,
                    step_expert_count=cfg.step_expert_count,experts=experts)

    def set_routing_step(self,step):
        """Install one content-independent roster for every microbatch in a step."""
        plan=self._routing_plan(step);self._routing_step=step
        self.router.eligible=tuple(plan['experts']) if self.config.step_expert_count else None
        return plan

    def set_roster(self,session_id):
        """Inference uses an explicit stable session id with the training policy.

        Keep it fixed throughout a generated sequence, including prefix replays.
        """
        return self.set_routing_step(session_id)

    def get_routing_plan(self):return self._routing_plan(self._routing_step)

    def restore_routing_plan(self,plan):
        if not isinstance(plan,dict) or plan!=self._routing_plan(plan.get('step')):
            raise ValueError('routing plan does not match configured deterministic policy')
        return self.set_routing_step(plan['step'])

    @staticmethod
    def _initialize(module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,std=.02)
            if isinstance(module,nn.Linear) and module.bias is not None:nn.init.zeros_(module.bias)
    def stream_stages(self):
        stages=[EmbeddingStage(self.embedding),*[SharedStage(block) for block in self.blocks]]
        if self.session_memory is not None:stages.append(MemoryStage(self.session_memory))
        stages.append(RouterStage(self.router))
        if self.config.expert_group_size == 1:
            stages.extend(ExpertStage(expert,index) for index,expert in enumerate(self.experts))
        else:
            size=self.config.expert_group_size
            stages.extend(GroupedExpertStage(self.experts[start:start+size],range(start,min(start+size,len(self.experts))))
                          for start in range(0,len(self.experts),size))
        stages.append(HeadStage(self));return stages
    def forward(self,input_ids,proof_graph=None,memory_state=None):
        x,aux=input_ids,None;next_memory=None
        for stage in self.stream_stages()[:-1]:
            if isinstance(stage,MemoryStage):
                x,next_memory=stage.module(x,proof_graph,memory_state)
            else:x,aux=stage(x,aux,proof_graph)
        output=HeadStage(self)(x,aux,proof_graph)
        if next_memory is not None:output['memory_state']=next_memory
        return DeltaOutput(output)
    def forward_proof(self,input_ids,graph,states=None):
        if states is not None:raise ValueError('Delta currently recomputes full prefixes; cached SSM decode is unsupported')
        return self(input_ids,graph)
    def loss(self,tokens,proof_graph=None):
        output=self(tokens[:,:-1],proof_graph)
        return F.cross_entropy(output['logits'].float().flatten(0,1),tokens[:,1:].flatten())+output['router_aux_loss']
