"""Native sparse model correctness; no full-size parameter allocations."""
import copy
from dataclasses import replace
from pathlib import Path
import torch
import pytest


def api():
    assert Path('morph/model/delta.py').exists(),'native Delta model has not been implemented'
    from morph.delta_config import DeltaConfig
    from morph.model.delta import DeltaModel
    return DeltaConfig,DeltaModel


def tiny(**kwargs):
    Config,Model=api()
    config=Config(d_model=16,num_layers=2,d_ff=32,d_state=4,head_dim=8,d_graph=8,graph_nodes=3,
                  expert_count=4,expert_depth=2,expert_ffn=24,latent_dim=8,**kwargs)
    return config,Model(config)


def test_native_sparse_stage_contract_and_tied_output():
    config,model=tiny();tokens=torch.randint(256,(2,7));output=model(tokens)
    assert output['logits'].shape==(2,7,256)
    assert output.logits is output['logits']
    assert model.lm_head.weight is model.embedding.weight
    x,aux=tokens,None
    stages=model.stream_stages()
    for stage in stages[:-1]:x,aux=stage(x,aux)
    staged=stages[-1](x,aux)
    torch.testing.assert_close(staged['logits'],output['logits'])
    assert aux.shape==(2,7,config.expert_count) and aux.requires_grad
    assert len([s for s in stages if getattr(s,'expert_index',None) is not None])==config.expert_count
    assert all(sum(p.numel() for p in s.parameters())<sum(p.numel() for p in model.parameters()) for s in stages)


def test_dropless_top_one_keeps_router_gradients_and_skips_inactive_parameters():
    _,model=tiny();tokens=torch.randint(256,(2,8))
    with torch.no_grad():model.router.projection.weight.zero_();model.router.projection.bias.zero_();model.router.projection.bias[0]=4
    observed={}
    handles=[expert.register_forward_pre_hook(lambda module,args,index=index:observed.__setitem__(index,observed.get(index,0)+args[0].shape[0])) for index,expert in enumerate(model.experts)]
    output=model(tokens)
    for handle in handles:handle.remove()
    assert sum(observed.values())==tokens.numel()
    assert observed=={0:tokens.numel()}
    loss=torch.nn.functional.cross_entropy(output['logits'].flatten(0,1),tokens.flatten())+output['router_aux_loss'];loss.backward()
    assert model.router.projection.weight.grad is not None and model.router.projection.weight.grad.abs().sum()>0
    assert all(p.grad is not None for p in model.experts[0].parameters())
    assert all(p.grad is None for expert in model.experts[1:] for p in expert.parameters())
    assert output['routes'].numel()==tokens.numel() and (output['routes']==0).all()
    assert output['router_balance_loss']>1


def test_unrolled_stages_gradient_and_expert_permutation_equivalence():
    _,model=tiny();other=copy.deepcopy(model);tokens=torch.randint(256,(2,9))
    model.loss(tokens).backward()
    stages=other.stream_stages();x,aux=tokens[:,:-1],None
    for stage in stages[:-1]:x,aux=stage(x,aux)
    output=stages[-1](x,aux)
    loss=torch.nn.functional.cross_entropy(output['logits'].float().flatten(0,1),tokens[:,1:].flatten())+output['router_aux_loss'];loss.backward()
    for a,b in zip(model.parameters(),other.parameters()):
        if a.grad is None:assert b.grad is None
        else:torch.testing.assert_close(a.grad,b.grad)
    # Top-1 token groups are disjoint, so experts may execute in reverse order.
    x,aux=tokens,None;prefix=[s for s in stages[:-1] if getattr(s,'expert_index',None) is None]
    for stage in prefix:x,aux=stage(x,aux)
    for stage in reversed([s for s in stages if getattr(s,'expert_index',None) is not None]):x,aux=stage(x,aux)
    torch.testing.assert_close(stages[-1](x,aux)['logits'],other(tokens)['logits'])


def test_causal_future_token_perturbation_and_latent_baseline():
    for latent in [True,False]:
        _,model=tiny(latent_experts=latent);tokens=torch.randint(256,(1,8));changed=tokens.clone();changed[:,5:]=torch.randint(256,(1,3))
        torch.testing.assert_close(model(tokens)['logits'][:,:5],model(changed)['logits'][:,:5],atol=1e-6,rtol=1e-5)


def test_delta_5b_meta_counts_and_configuration_validation():
    Config,Model=api();cfg=Config.from_file('configs/morph_delta_5b.json')
    with torch.device('meta'):model=Model(cfg)
    total=sum(p.numel() for p in model.parameters());experts=sum(p.numel() for p in model.experts.parameters());shared=total-experts
    assert 4.95e9<total<5.05e9 and .25e9<shared<.35e9
    assert cfg.expert_count==64 and cfg.vocab_size==256
    assert all(p.is_meta for p in model.parameters())
    assert Config(**cfg.to_dict())==cfg
    with pytest.raises(ValueError):replace(cfg,top_k=2)


def test_session_memory_is_returned_without_persistent_model_mutation():
    _,model=tiny();tokens=torch.randint(256,(1,6));output=model(tokens)
    assert 'memory_state' in output,'inference must expose the newly learned session state'
    before=output['memory_state'].matrix.detach().clone()
    continued=model(tokens,memory_state=output['memory_state'])
    torch.testing.assert_close(output['memory_state'].matrix,before)
    assert not torch.allclose(continued['logits'],output['logits'])
    torch.testing.assert_close(model(tokens)['logits'],output['logits'])


def test_proof_graph_conditions_router_logits_and_all_joint_heads_train():
    from morph.graph import ProofGraph,ProofNode,ActionType,Hyperedge,Relation
    from morph.graph.tensors import tensorize
    from morph.training.proof import joint_loss
    cfg,model=tiny(proof_enabled=True);graph=ProofGraph();graph.add(ProofNode(0,'True',formal_statement='True'))
    graph.add(ProofNode(1,'True implies True',formal_statement='True -> True'));graph.connect(Hyperedge((0,),1,Relation.REQUIRES))
    first,_=tensorize([graph],cfg.graph_nodes,cfg.statement_field_length)
    graph.nodes[0].formal_statement='False';second,_=tensorize([graph],cfg.graph_nodes,cfg.statement_field_length)
    tokens=torch.randint(256,(1,8));a=model.forward_proof(tokens,first);b=model.forward_proof(tokens,second)
    assert not torch.allclose(a['logits'],b['logits'])
    # Router probabilities, not just final output heads, receive graph context.
    def gates(data):
        x,aux=tokens,None
        for stage in model.stream_stages():
            if getattr(stage,'expert_index',None) is not None:break
            x,aux=stage(x,aux,data)
        return aux
    assert not torch.allclose(gates(first),gates(second),atol=1e-8,rtol=1e-6)
    loss,_=joint_loss(a,tokens,first,torch.tensor([ActionType.VERIFY]),torch.tensor([0]),torch.tensor([1.]))
    (loss+a['router_aux_loss']).backward()
    for module in [model.reasoning.action,model.reasoning.target,model.reasoning.value,model.reasoning.dependency,model.blocks[0].ssm,model.session_memory.memory]:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())


def test_frozen_graph_context_cache_recomputes_only_changed_local_dependencies():
    from morph.graph import ProofGraph,ProofNode,Hyperedge,Relation
    from morph.graph.tensors import tensorize
    cfg,model=tiny(proof_enabled=True);model.eval();graph=ProofGraph()
    for index,statement in enumerate(['A','B','C']):graph.add(ProofNode(index,statement,formal_statement='True'))
    graph.connect(Hyperedge((0,),1,Relation.REQUIRES))
    tensors,_=tensorize([graph],cfg.graph_nodes,cfg.statement_field_length)
    module=model.session_memory
    with torch.no_grad():
        context=module.graph_context(tensors,use_cache=True)
        torch.testing.assert_close(context,module.graph_context(tensors,use_cache=False))
        module.graph_context(tensors,use_cache=True)
        assert module.last_recomputed==set()
        graph.nodes[0].statement='changed A'
        edited,_=tensorize([graph],cfg.graph_nodes,cfg.statement_field_length)
        cached=module.graph_context(edited,use_cache=True)
        assert module.last_encoded=={0} and module.last_recomputed=={0,1}
        torch.testing.assert_close(cached,module.graph_context(edited,use_cache=False),atol=1e-6,rtol=1e-5)
        module.graph_context(edited,use_cache=True,memory_version=1)
        assert module.last_recomputed=={0,1,2}
        module.encoder.mix.weight.add_(.01)
        cached=module.graph_context(edited,use_cache=True,memory_version=1)
        assert module.last_recomputed=={0,1,2}
        torch.testing.assert_close(cached,module.graph_context(edited,use_cache=False),atol=1e-6,rtol=1e-5)
    model.train();result=module.graph_context(edited,use_cache=True);result.square().sum().backward()
    assert module.encoder.mix.weight.grad is not None
