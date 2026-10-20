import copy
import pytest
import torch
from morph import MorphConfig, MorphModel
from morph.graph import *
from morph.graph.tensors import tensorize
from morph.training.proof import joint_loss


def graph_fixture():
    graph=ProofGraph()
    graph.add(ProofNode(0,'P implies P',formal_statement='True -> True',assumptions=('P',)))
    graph.add(ProofNode(1,'P',node_type=NodeType.ASSUMPTION,assumptions=('P',)))
    graph.connect(Hyperedge((1,),0,Relation.REQUIRES))
    return graph


def test_actions_replay_assumptions_and_verification_boundary():
    initial=graph_fixture(); graph=copy.deepcopy(initial)
    actions=[dict(type='CREATE_LEMMA',nodes=[dict(node_id=2,statement='candidate',node_type='LEMMA')]),
             dict(type='CONNECT',edge=dict(sources=[1,2],target=0,relation='IMPLIES')),
             dict(type='VERIFY',target=0),dict(type='PRUNE',target=2)]
    records=[apply_action(graph,a) for a in actions]
    assert replay(initial,records).digest()==graph.digest()
    assert graph.nodes[0].status==Status.OPEN
    with pytest.raises(ValueError):
        apply_action(graph,dict(type='CREATE_LEMMA',nodes=[dict(node_id=3,statement='false',status='VERIFIED')]))
    with pytest.raises(ValueError,match='assumptions'):
        apply_action(graph,dict(type='DECOMPOSE',target=0,nodes=[dict(node_id=4,statement='subgoal')]))
    records[0]['after']='tampered'
    with pytest.raises(ValueError):
        replay(initial,records)


def test_symbolic_graph_feeds_shared_core_and_joint_backward():
    cfg=MorphConfig(vocab_size=256,d_model=16,num_layers=2,d_ff=32,d_state=4,
                    head_dim=8,d_graph=8,graph_nodes=4,proof_enabled=True)
    model=MorphModel(cfg)
    graph,_=tensorize([graph_fixture()],4)
    tokens=torch.tensor([[10,20,30,40]])
    output=model.forward_proof(tokens,graph)
    loss,parts=joint_loss(output,tokens,graph,torch.tensor([9]),torch.tensor([0]),torch.tensor([1.]))
    for state in output['iterations']:
        state.retain_grad()
    loss.backward()
    assert all(state.grad is not None and state.grad.abs().sum()>0 for state in output['iterations'])
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    changed=copy.deepcopy(graph);changed['node_types'][0,1]=2
    assert not torch.allclose(output['value'],model.forward_proof(tokens,changed)['value'])
    changed=copy.deepcopy(graph);changed['source_mask'].zero_()
    assert not torch.allclose(output['value'],model.forward_proof(tokens,changed)['value'])


def test_teacher_pair_masks_and_untrusted_restore():
    graph=graph_fixture()
    graph.nodes[0].status=Status.PROPOSED
    tensors,_=tensorize([graph],4)
    assert tensors['target_action_mask'][0,ActionType.VERIFY,0]
    assert not tensors['target_action_mask'][0,ActionType.VERIFY,1]
    duplicate=graph.snapshot();duplicate['nodes'].append(duplicate['nodes'][0])
    with pytest.raises(ValueError,match='duplicate'):ProofGraph.restore(duplicate)
    forged=graph.snapshot();forged['nodes'][0]['status']=int(Status.VERIFIED)
    forged['nodes'][0]['proof_hash']='invented'
    with pytest.raises(ValueError,match='evidence'):ProofGraph.restore(forged)
    cfg=MorphConfig(vocab_size=256,d_model=8,num_layers=1,d_ff=16,d_state=2,head_dim=4,d_graph=4,proof_enabled=True)
    model=MorphModel(cfg);tokens=torch.tensor([[1,2,3]])
    output=model.forward_proof(tokens,tensors)
    with pytest.raises(ValueError,match='inactive'):
        joint_loss(output,tokens,tensors,torch.tensor([ActionType.VERIFY]),torch.tensor([1]),torch.tensor([0.]))
    graph.nodes[0].node_type=NodeType.CONJECTURE
    with pytest.raises(ValueError,match='dependencies'):
        apply_action(graph,dict(type='REVISE',target=0,statement='new statement'))


def test_streaming_proof_core_requires_structured_batch(tmp_path):
    from streamtrain import StreamTrainEngine
    cfg=MorphConfig(proof_enabled=True)
    engine=StreamTrainEngine(MorphModel(cfg))
    assert engine.stages[-1].kind=='proof'
    with pytest.raises(ValueError,match='structured'):
        engine.backward(torch.tensor([[1,2,3]]))
    from streamtrain.disk import disk_train
    with pytest.raises(ValueError,match='structured proof'):
        disk_train(cfg,torch.tensor([[1,2,3]]),tmp_path/'unsupported',1)
    assert not (tmp_path/'unsupported').exists()


def test_rollout_all_pruned_terminal_and_capacity_is_atomic():
    from morph.training.proof_runner import rollout
    cfg=MorphConfig(vocab_size=256,d_model=8,num_layers=1,d_ff=16,d_state=2,head_dim=4,d_graph=4,graph_nodes=1,proof_enabled=True)
    model=MorphModel(cfg)
    graph=ProofGraph();graph.add(ProofNode(0,'goal'))
    with torch.no_grad():
        model.reasoning.action.weight.zero_();model.reasoning.action.bias.zero_()
        model.reasoning.action.bias[ActionType.PRUNE]=100
    events=rollout(model,'goal',graph,lambda *args:{},max_steps=3)
    assert len(events)==1 and graph.nodes[0].status==Status.BLOCKED
    graph=ProofGraph();graph.add(ProofNode(0,'goal'));before=graph.digest()
    with torch.no_grad():
        model.reasoning.action.bias.zero_();model.reasoning.action.bias[ActionType.CREATE_LEMMA]=100
    events=rollout(model,'goal',graph,lambda *args:dict(nodes=[dict(node_id=1,statement='new lemma')]))
    assert events[0]['reason']=='graph capacity exceeded' and graph.digest()==before


def test_decomposition_records_directed_obligations_and_replays():
    graph=graph_fixture();initial=copy.deepcopy(graph)
    event=apply_action(graph,dict(type='DECOMPOSE',target=0,nodes=[dict(node_id=2,statement='subgoal',assumptions=['P'])]))
    assert Hyperedge((2,),0,Relation.REQUIRES) in graph.edges
    assert replay(initial,[event]).digest()==graph.digest()
