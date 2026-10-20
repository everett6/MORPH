"""Exact saved-gradient AdamW and recovery, using only tiny CPU models."""
import pytest
import torch
from morph.delta_config import DeltaConfig
from morph.model.delta import DeltaModel
from streamtrain.disk import DiskState
from streamtrain.sparse_disk import SparseDiskDLS
from streamtrain.single_pass import SinglePassDiskDLS


def fixture(tmp_path):
    torch.manual_seed(85)
    cfg=DeltaConfig(d_model=8,num_layers=1,d_ff=16,d_state=2,head_dim=4,d_graph=8,
                    graph_nodes=3,memory_dim=4,expert_count=3,expert_depth=1,
                    expert_ffn=16,latent_dim=4,proof_enabled=True,statement_field_length=8,step_expert_count=1)
    source=DeltaModel(cfg)
    with torch.device('meta'):template=DeltaModel(cfg)
    a=DiskState.create(tmp_path/'single',template,source=source)
    b=DiskState.create(tmp_path/'two',template,source=source)
    return template,a,b,[torch.tensor([[1,2,3,4]]),torch.tensor([[3,4,5,6]])]


def identical(a,b):
    assert a.manifest['step']==b.manifest['step']
    assert a.manifest['parameter_steps']==b.manifest['parameter_steps']
    for name,entry in a.entries.items():
        assert (a.root/entry['file']).read_bytes()==(b.root/entry['file']).read_bytes(),name


@pytest.mark.parametrize('joint',[False,True])
def test_single_backward_matches_two_pass_clipping_tied_weights_and_none_clocks(tmp_path,joint):
    model,a,b,batches=fixture(tmp_path)
    if joint:
        from morph.graph.schema import ProofGraph,ProofNode
        from morph.graph.tensors import tensorize
        graph=ProofGraph();graph.add(ProofNode(0,'A'))
        tensors,_=tensorize([graph],3,statement_length=8)
        batches=[dict(tokens=batches[0],graph=tensors,actions=torch.tensor([2]),
                      targets=torch.tensor([0]),values=torch.tensor([.4]))]
    single=SinglePassDiskDLS(model,a,stash_budget_bytes=10_000_000,cpu_prefetch=True)
    two=SparseDiskDLS(model,b)
    count=0;reverse=single.reverse
    def counted(*args,**kwargs):
        nonlocal count
        count+=1;return reverse(*args,**kwargs)
    single.reverse=counted
    try:
        for step in range(2):
            actual=single.step(batches,max_norm=.01)
            expected=two.step(batches,max_norm=.01)
            assert actual==expected
            identical(a,b)
        assert count==2
        assert not (a.root/'gradient-stash').exists()
    finally:single.close();two.close()


def test_stash_budget_rejected_before_reverse_or_weight_mutation(tmp_path):
    model,a,_,batches=fixture(tmp_path)
    before={name:(a.root/entry['file']).read_bytes() for name,entry in a.entries.items()}
    trainer=SinglePassDiskDLS(model,a,stash_budget_bytes=1)
    def forbidden(*args,**kwargs):raise AssertionError('backward allocated before budget check')
    trainer.reverse=forbidden
    try:
        with pytest.raises(MemoryError,match='stash'):trainer.step(batches)
        assert not a.manifest['dirty']
        for name,payload in before.items():assert (a.root/a.entries[name]['file']).read_bytes()==payload
    finally:trainer.close()


@pytest.mark.parametrize('point',['gradient_stash_durable','stored_transaction_durable','journal_durable',
                                  'partial_state_write','cursor_file_durable','cursor_manifest_durable',
                                  'before_final_manifest','final_manifest_durable'])
def test_interruption_recovers_from_gradients_without_backward(tmp_path,point):
    model,a,b,batches=fixture(tmp_path)
    reference=SparseDiskDLS(model,b)
    reference.step(batches,max_norm=.02);reference.close()
    def fault(actual):
        if actual==point:raise OSError('injected '+point)
    a.fault=fault
    trainer=SinglePassDiskDLS(model,a)
    try:
        with pytest.raises(OSError,match='injected'):trainer.step(batches,max_norm=.02)
    finally:trainer.close()
    reopened=DiskState.open(a.root,model)
    resumed=SinglePassDiskDLS(model,reopened)
    try:
        if reopened.manifest.get('transaction'):
            assert reopened.manifest['transaction']['kind']=='stored-gradients-v1'
            def forbidden(*args,**kwargs):raise AssertionError('resume reran backward')
            resumed.reverse=forbidden;resumed.boundaries=forbidden
            plan=reopened.manifest['transaction']['routing_plan']
            model.set_routing_step(99)
            resumed.resume_step()
            assert model.get_routing_plan()==plan
        elif reopened.manifest['step']==0:resumed.step(batches,max_norm=.02)
        identical(reopened,b)
        assert not (reopened.root/'gradient-stash').exists()
    finally:resumed.close()


def test_wrong_engine_fails_closed(tmp_path):
    model,a,_,batches=fixture(tmp_path)
    def fault(actual):
        if actual=='stored_transaction_durable':raise OSError('stop')
    a.fault=fault
    trainer=SinglePassDiskDLS(model,a)
    try:
        with pytest.raises(OSError):trainer.step(batches)
    finally:trainer.close()
    reopened=DiskState.open(a.root,model)
    wrong=SparseDiskDLS(model,reopened)
    try:
        with pytest.raises(ValueError,match='stored.gradient'):wrong.resume_step()
    finally:wrong.close()


def test_fp16_overflow_retries_without_publishing_or_updating_stash(tmp_path):
    model,a,b,batches=fixture(tmp_path)
    trainer=SinglePassDiskDLS(model,a,dtype=torch.float16)
    reference=SparseDiskDLS(model,b,dtype=torch.float16)
    trainer.scaler.scale=reference.scaler.scale=2.**22
    calls=0;reverse=trainer.reverse
    def counted(*args,**kwargs):
        nonlocal calls
        calls+=1
        assert not a.manifest['dirty']
        assert not (a.root/'gradient-stash').exists()
        return reverse(*args,**kwargs)
    trainer.reverse=counted
    try:
        actual=trainer.step(batches,max_norm=.02)
        expected=reference.step(batches,max_norm=.02)
        assert calls>1 and trainer.scaler.scale<2.**22
        assert actual==expected
        identical(a,b)
    finally:trainer.close();reference.close()


def test_same_process_fp16_retry_does_not_double_advance_loss_scaler(tmp_path):
    model,a,b,batches=fixture(tmp_path)
    trainer=SinglePassDiskDLS(model,a,dtype=torch.float16)
    reference=SparseDiskDLS(model,b,dtype=torch.float16)
    def fault(point):
        if point=='before_final_manifest':raise OSError('finish interrupted')
    a.fault=fault
    try:
        with pytest.raises(OSError):trainer.step(batches)
        a.fault=lambda point:None
        trainer.resume_step();reference.step(batches)
        identical(a,b)
        assert trainer.scaler.state_dict()==reference.scaler.state_dict()
    finally:trainer.close();reference.close()


def test_model_without_roster_methods_uses_same_saved_gradient_protocol(tmp_path):
    from morph import MorphConfig,MorphModel
    cfg=MorphConfig(vocab_size=8,d_model=8,num_layers=1,d_ff=16,d_state=2,head_dim=4)
    source=MorphModel(cfg)
    with torch.device('meta'):model=MorphModel(cfg)
    a=DiskState.create(tmp_path/'plain-single',model,source=source)
    b=DiskState.create(tmp_path/'plain-two',model,source=source)
    single=SinglePassDiskDLS(model,a);two=SparseDiskDLS(model,b)
    try:
        batches=[torch.tensor([[1,2,3,4]])]
        assert single.step(batches)==two.step(batches)
        identical(a,b)
    finally:single.close();two.close()
