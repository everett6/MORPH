"""Functional proof stage: ordered node text, causal graph readout, joint heads."""
import torch
from torch import nn
from torch.nn import functional as F
from .graph import aggregate_hyperedges


class StatementEncoder(nn.Module):
    def __init__(self,width,max_length=512):
        super().__init__()
        self.position=nn.Embedding(max_length,width)
        self.mix=nn.Linear(width,width)
        self.order=nn.Conv1d(width,width,3,padding=1,groups=width,bias=False)
        self.normalization=nn.LayerNorm(width)

    def forward(self,embeddings,mask):
        batch,nodes,length,width=embeddings.shape
        if length>self.position.num_embeddings:raise ValueError('statement exceeds encoder capacity')
        positions=self.position(torch.arange(length,device=embeddings.device))
        ordered=torch.tanh(self.mix(embeddings)+positions)
        ordered=ordered*mask[...,None]
        ordered=self.order(ordered.reshape(batch*nodes,length,width).transpose(1,2)).transpose(1,2)
        ordered=F.silu(ordered).reshape(batch,nodes,length,width)*mask[...,None]
        return self.normalization(ordered.sum(-2)/mask.sum(-1,keepdim=True).clamp_min(1))


class ProofHead(nn.Module):
    def __init__(self,model):
        super().__init__()
        cfg=model.config
        self.norm=model.norm
        self.weight=model.embedding.weight
        self.reasoning=model.reasoning
        self.encoder=StatementEncoder(cfg.d_model,cfg.statement_field_length*3)
        self.query=nn.Linear(cfg.d_model,cfg.d_graph,bias=False)
        self.write=nn.Linear(cfg.d_graph,cfg.d_model,bias=False)

    def forward(self,x,unused_graph,graph):
        hidden=self.norm(x)
        embedded=F.embedding(graph['tokens'],self.weight)
        nodes=self.encoder(embedded,graph['token_mask'])
        # Graph-only conditioning has no access to future sequence positions.
        memory=(self.reasoning.node_projection(nodes)+self.reasoning.node_types(graph['node_types'])
                +self.reasoning.statuses(graph['statuses']))*graph['active'][...,None]
        message=aggregate_hyperedges(memory,graph['sources'],graph['targets'],graph['source_mask'],
                                     self.reasoning.relations(graph['relations']))
        memory=(memory+message)*graph['active'][...,None]
        score=torch.einsum('btd,bnd->btn',self.query(hidden),memory)/(memory.shape[-1]**.5)
        score=score.masked_fill(~graph['active'][:,None],-1e4)
        conditioned=hidden+self.write(torch.einsum('btn,bnd->btd',score.softmax(-1),memory))
        heads=self.reasoning(hidden,nodes,graph)
        return dict(logits=F.linear(conditioned,self.weight),**heads)
