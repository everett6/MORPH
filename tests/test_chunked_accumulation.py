"""Bounded activation windows retain the full-step AdamW objective."""
import math
import pytest
import torch
from streamtrain.single_pass import SinglePassDiskDLS
from streamtrain.disk import DiskState
from test_single_pass import fixture


@pytest.mark.parametrize('chunk',[1,2,4])
@pytest.mark.parametrize('joint',[False,True])
def test_chunk_parity_odd_microbatches_and_clipping(tmp_path,chunk,joint):
    model,a,b,batches=fixture(tmp_path)
    batches=(batches*3)[:5]
    if joint:
        from morph.graph.schema import ProofGraph,ProofNode
        from morph.graph.tensors import tensorize
        graph=ProofGraph();graph.add(ProofNode(0,'A'))
        tensors,_=tensorize([graph],3,statement_length=8)
        batches=[dict(tokens=t,graph=tensors,actions=torch.tensor([2]),
                      targets=torch.tensor([0]),values=torch.tensor([.4])) for t in batches]
    actual=SinglePassDiskDLS(model,a,chunk_microbatches=chunk)
    reference=SinglePassDiskDLS(model,b)
    try:
        result=actual.step(batches,max_norm=.01);expected=reference.step(batches,max_norm=.01)
        assert result['loss']==pytest.approx(expected['loss'],rel=2e-6)
        assert result['grad_norm']==pytest.approx(expected['grad_norm'],rel=2e-6)
        assert a.manifest['parameter_steps']==b.manifest['parameter_steps']
        for name in a.entries:
            import numpy as np
            left=np.fromfile(a.root/a.entries[name]['file'],dtype=np.float32)
            right=np.fromfile(b.root/b.entries[name]['file'],dtype=np.float32)
            torch.testing.assert_close(torch.from_numpy(left),torch.from_numpy(right),atol=2e-6,rtol=2e-5)
        assert actual.peak_boundary_microbatches==min(chunk,5)
        assert actual.completed_microbatches==5
        assert actual.active_expert_union
    finally:actual.close();reference.close()


def test_chunks_released_before_next_boundaries_and_fixed_roster(tmp_path):
    import weakref
    model,a,_,batches=fixture(tmp_path)
    trainer=SinglePassDiskDLS(model,a,chunk_microbatches=1)
    original=trainer.boundaries;previous=[];plans=[]
    def watched(batches):
        assert all(ref() is None for ref in previous)
        saved=original(batches)
        previous[:]=[weakref.ref(t) for states in saved for state in states for t in state if t is not None]
        plans.append(model.get_routing_plan())
        return saved
    trainer.boundaries=watched
    try:
        trainer.step(batches)
        assert len(plans)==len(batches)
        assert all(plan==plans[0] for plan in plans)
    finally:trainer.close()


def test_chunk_preflight_precedes_any_reverse(tmp_path):
    model,a,_,batches=fixture(tmp_path)
    trainer=SinglePassDiskDLS(model,a,chunk_microbatches=1,stash_budget_bytes=1)
    trainer.reverse=lambda *args,**kwargs:pytest.fail('backward before union preflight')
    try:
        with pytest.raises(MemoryError):trainer.step(batches)
        assert not a.manifest['dirty']
    finally:trainer.close()


@pytest.mark.parametrize('point',['stored_transaction_durable','partial_state_write','cursor_manifest_durable','empty_stored_stage_durable'])
def test_chunk_resume_applies_saved_gradients_after_empty_gaps(tmp_path,point):
    model,a,b,batches=fixture(tmp_path)
    trainer=SinglePassDiskDLS(model,a,chunk_microbatches=1)
    reference=SinglePassDiskDLS(model,b,chunk_microbatches=1)
    try:
        expected=reference.step(batches)
        def fault(where):
            if where==point:raise OSError('interrupt')
        a.fault=fault
        with pytest.raises(OSError):trainer.step(batches)
    finally:trainer.close();reference.close()
    reopened=DiskState.open(a.root,model)
    resumed=SinglePassDiskDLS(model,reopened,chunk_microbatches=2)
    resumed.reverse=resumed.boundaries=lambda *args,**kwargs:pytest.fail('recovery reran backward')
    try:
        assert resumed.resume_step()==expected
        assert reopened.manifest['parameter_steps']==b.manifest['parameter_steps']
        for name,entry in b.entries.items():
            assert (reopened.root/entry['file']).read_bytes()==(b.root/entry['file']).read_bytes()
    finally:resumed.close()


def test_overflow_replays_all_chunks_before_publication(tmp_path):
    model,a,b,batches=fixture(tmp_path)
    batches=(batches*2)[:3]
    trainer=SinglePassDiskDLS(model,a,chunk_microbatches=1,dtype=torch.float16)
    reference=SinglePassDiskDLS(model,b,chunk_microbatches=1,dtype=torch.float16)
    trainer.scaler.scale=reference.scaler.scale=2.
    reverse=trainer.reverse;calls=0
    def injected(*args,**kwargs):
        nonlocal calls
        calls+=1
        assert not a.manifest['dirty']
        assert not trainer.stash_path.exists()
        result=reverse(*args,**kwargs)
        # Overflow the second chunk once; first chunk must be recomputed too.
        return (float('inf'),result[1]) if calls==2 else result
    trainer.reverse=injected
    try:
        result=trainer.step(batches)
        reference.scaler.scale=1.
        expected=reference.step(batches)
        assert calls==5
        assert result==expected
        assert trainer.completed_microbatches==3
        for name,entry in a.entries.items():
            assert (a.root/entry['file']).read_bytes()==(b.root/entry['file']).read_bytes()
    finally:trainer.close();reference.close()


@pytest.mark.parametrize('value',[-1,True,1.5])
def test_invalid_chunk_count(tmp_path,value):
    with pytest.raises(ValueError,match='chunk'):
        SinglePassDiskDLS(chunk_microbatches=value)


@pytest.mark.parametrize('chunk',[0,2])
def test_grouped_single_pass_uses_only_roster_stash_and_restores_activity(tmp_path,chunk):
    from dataclasses import replace
    from test_execution_optimizations import tiny
    from morph.model.delta import DeltaModel
    torch.manual_seed(31)
    cfg=tiny(step_expert_count=1)
    source=DeltaModel(cfg)
    with torch.device('meta'):
        model=DeltaModel(replace(cfg,expert_group_size=3,memory_projection_backend='batched'))
        plain=DeltaModel(cfg)
    a=DiskState.create(tmp_path/'group',model,source=source)
    b=DiskState.create(tmp_path/'plain',plain,source=source)
    eligible=set(model.get_routing_plan()['experts'])
    budget=sum(entry['size']*4 for name,entry in a.entries.items()
               if SinglePassDiskDLS._expert_id(name) is None or SinglePassDiskDLS._expert_id(name) in eligible)
    assert budget<sum(entry['size']*4 for entry in a.entries.values())
    trainer=SinglePassDiskDLS(model,a,chunk_microbatches=chunk,stash_budget_bytes=budget)
    reference=SinglePassDiskDLS(plain,b)
    batches=[torch.tensor([[1,2,3,4]]),torch.tensor([[4,5,6,7]]),torch.tensor([[7,8,9,10]])]
    def fault(point):
        if point=='stored_transaction_durable':raise OSError('interrupt')
    try:
        expected=reference.step(batches,max_norm=.01)
        a.fault=fault
        with pytest.raises(OSError):trainer.step(batches,max_norm=.01)
    finally:trainer.close();reference.close()
    reopened=DiskState.open(a.root,model)
    resumed=SinglePassDiskDLS(model,reopened,chunk_microbatches=chunk,stash_budget_bytes=budget)
    resumed.reverse=resumed.boundaries=lambda *args,**kwargs:pytest.fail('recovery reran backward')
    try:
        result=resumed.resume_step()
        assert result['loss']==pytest.approx(expected['loss'],rel=2e-6)
        assert result['grad_norm']==pytest.approx(expected['grad_norm'],rel=2e-6)
        assert resumed.active_expert_union==sorted(eligible)
        assert reopened.manifest['parameter_steps']==b.manifest['parameter_steps']
        for name in b.entries:torch.testing.assert_close(reopened.weight(name),b.weight(name),atol=2e-6,rtol=2e-5)
        for stage,active in zip(resumed.stages,resumed.active_stages):
            if hasattr(stage,'expert_indices'):assert active==bool(set(stage.expert_indices)&eligible)
    finally:resumed.close()
