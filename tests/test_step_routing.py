from dataclasses import replace
import pytest
import torch
from morph.delta_config import DeltaConfig
from morph.model.delta import DeltaModel,router_balance


def tiny():return replace(DeltaConfig.from_file('configs/morph_delta_tiny.json'),step_expert_count=2,routing_seed=17,proof_enabled=False)


def test_roster_is_content_independent_causal_and_replayable():
    torch.set_num_threads(2);model=DeltaModel(tiny()).eval();model.set_routing_step(3)
    plan=model.get_routing_plan();tokens=torch.randint(0,256,(2,8));changed=tokens.clone();changed[:,4:]=(changed[:,4:]+1)%256
    with torch.no_grad():a=model(tokens);b=model(changed)
    assert model.get_routing_plan()==plan
    torch.testing.assert_close(a.logits[:,:4],b.logits[:,:4])
    assert set(a['routes'].flatten().tolist())<=set(plan['experts'])
    assert a['routes'].numel()==tokens.numel()
    other=DeltaModel(tiny());other.restore_routing_plan(plan);assert other.get_routing_plan()==plan
    with pytest.raises(ValueError):other.restore_routing_plan({**plan,'experts':[0,0]})


def test_rotating_rosters_cover_bank_and_preserve_parameters():
    cfg=replace(tiny(),expert_count=64,step_expert_count=8)
    with torch.device('meta'):model=DeltaModel(cfg);baseline=DeltaModel(replace(cfg,step_expert_count=0))
    assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in baseline.parameters())
    coverage=[]
    for step in range(8):model.set_routing_step(step);coverage+=model.get_routing_plan()['experts']
    assert sorted(coverage)==list(range(64))
    with pytest.raises(ValueError):replace(cfg,step_expert_count=65)


def test_eligible_balance_and_inactive_expert_gradients():
    model=DeltaModel(tiny());model.set_roster(1);plan=model.get_routing_plan()
    output=model(torch.randint(0,256,(2,9)));(output.logits.square().mean()+output.router_aux_loss).backward()
    for index,expert in enumerate(model.experts):
        if index not in plan['experts']:assert all(p.grad is None for p in expert.parameters())
    gates=torch.tensor([[[.5,.5,0.,0.]]])
    torch.testing.assert_close(router_balance(gates,2),torch.tensor(1.))
