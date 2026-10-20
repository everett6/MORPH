import pytest
import torch
from morph import MorphConfig,MorphModel
from morph.graph import *
from morph.training.proof_runner import rollout
from morph.inference_payload import PayloadDecoder,validate_payload
from verifier import LeanClient


def test_no_provider_rollout_uses_decoder_and_masks(monkeypatch):
    cfg=MorphConfig(vocab_size=256,d_model=8,num_layers=1,d_ff=16,d_state=2,head_dim=4,d_graph=4,graph_nodes=2,proof_enabled=True)
    model=MorphModel(cfg);graph=ProofGraph();graph.add(ProofNode(0,'goal'))
    with torch.no_grad():
        model.reasoning.action.weight.zero_();model.reasoning.action.bias.zero_();model.reasoning.action.bias[ActionType.CREATE_LEMMA]=100
    calls=[]
    def decode(self,graph,target,kind,diagnostics=''):
        calls.append(kind);return dict(valid=True,payload=dict(statement='decoder generated candidate'),raw='fixture decoder',model_generated=True)
    monkeypatch.setattr(PayloadDecoder,'decode',decode)
    events=rollout(model,'goal',graph,max_steps=1)
    assert calls==[ActionType.CREATE_LEMMA] and graph.nodes[1].statement=='decoder generated candidate'
    assert events[0]['decoded']['model_generated']
    with pytest.raises(ValueError):validate_payload(dict(statement='candidate',status='VERIFIED'),ActionType.CREATE_LEMMA)


def test_proposed_edges_never_close_false_and_cycles_remain_unsolved():
    graph=ProofGraph();graph.add(ProofNode(0,'P'));graph.add(ProofNode(1,'False',formal_statement='False'))
    graph.nodes[0].status=Status.VERIFIED;graph.nodes[0].proof_hash='unit-test-premise'
    graph.connect(Hyperedge((0,),1,Relation.REQUIRES))
    assert not obligation_closed(graph,1)
    graph.connect(Hyperedge((1,),0,Relation.REQUIRES))
    assert not obligation_closed(graph,0)


@pytest.mark.skipif(not LeanClient().available,reason='real Lean unavailable')
def test_real_checker_closure_and_assumption_audit():
    client=LeanClient()
    assert not client.verify('P -> P','by intro h; exact h').success
    assert client.verify('P -> P','by intro h; exact h',('(P : Prop)',)).success
    graph=ProofGraph();graph.add(ProofNode(0,'True',formal_statement='True'));graph.add(ProofNode(1,'True and True',formal_statement='True ∧ True'))
    graph.accept_verification(0,client.verify('True','by trivial'))
    graph.connect(Hyperedge((0,),1,Relation.REQUIRES))
    assert not obligation_closed(graph,1)
    certify_edge(graph,0,client.verify('True ∧ True','by constructor <;> trivial'))
    assert obligation_closed(graph,1) and graph.edges[0].certified

@pytest.mark.skipif(not LeanClient().available,reason='real Lean unavailable')
def test_repeated_invalid_proof_has_stable_diagnostics():
    client=LeanClient()
    first=client.verify('False','by trivial');second=client.verify('False','by trivial')
    assert not first.success and not second.success
    assert first.diagnostics==second.diagnostics
    assert '/tmp/morph-lean-' not in first.diagnostics

def test_original_goal_cannot_be_revised_into_an_easier_theorem():
    graph=ProofGraph();graph.add(ProofNode(0,'False',formal_statement='False'))
    assert not valid_actions(graph,0)[ActionType.REVISE]
    with pytest.raises(ValueError,match='invalid'):
        apply_action(graph,dict(type='REVISE',target=0,statement='True',formal_statement='True'))
    assert graph.nodes[0].formal_statement=='False' and not obligation_closed(graph,0)
    graph.add(ProofNode(1,'candidate',node_type=NodeType.CONJECTURE,formal_statement='False'))
    assert valid_actions(graph,1)[ActionType.REVISE]
    apply_action(graph,dict(type='REVISE',target=1,statement='revised candidate',formal_statement='True'))
    assert graph.nodes[1].formal_statement=='True' and graph.nodes[1].status==Status.PROPOSED
