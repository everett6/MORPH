import json
import pytest
import torch
from test_sparse_disk import tiny_config


def test_packed_multistep_recovery_and_truncation(tmp_path):
    from morph.model.delta import DeltaModel
    from streamtrain.packed_state import PackedDiskState
    from streamtrain.sparse_disk import SparseDiskDLS
    source=DeltaModel(tiny_config())
    with torch.device('meta'):model=DeltaModel(tiny_config())
    a=PackedDiskState.create(tmp_path/'a',model,source=source)
    b=PackedDiskState.create(tmp_path/'b',model,source=source)
    batches=[torch.tensor([[1,2,3,4]])]
    for _ in range(2):SparseDiskDLS(model,b).step(batches)
    SparseDiskDLS(model,a).step(batches)
    def fault(point):
        if point=='partial_state_write':raise OSError('crash')
    a.fault=fault
    with pytest.raises(OSError):SparseDiskDLS(model,a).step(batches)
    journal=a.root/'journal'/'state.bin'
    payload=journal.read_bytes();journal.write_bytes(payload[:-1])
    before={n:(a.root/e['file']).read_bytes() for n,e in a.entries.items()}
    a=PackedDiskState.open(a.root,model)
    with pytest.raises(ValueError,match='journal'):a.recover_stage()
    assert before=={n:(a.root/e['file']).read_bytes() for n,e in a.entries.items()}
    journal.write_bytes(payload)
    trainer=SparseDiskDLS(model,a);trainer.resume_step()
    for n,e in b.entries.items():assert (a.root/e['file']).read_bytes()==(b.root/e['file']).read_bytes()
    assert a.manifest['parameter_steps']==b.manifest['parameter_steps']
    profile=trainer.profile_snapshot()
    assert profile['counts']['update']>0
    assert profile['seconds']['update_reverse_launch_wall']>0
    assert profile['seconds']['expert_update_reverse_launch_wall']>0


def test_packed_cached_parity_and_sync_reduction(tmp_path):
    from morph.model.delta import DeltaModel
    from streamtrain.disk import DiskState
    from streamtrain.packed_state import PackedCachedDiskState, PackedDiskState
    from streamtrain.sparse_disk import SparseDiskDLS
    source=DeltaModel(tiny_config())
    with torch.device('meta'):model=DeltaModel(tiny_config())
    legacy=DiskState.create(tmp_path/'legacy',model,source=source)
    packed=PackedDiskState.create(tmp_path/'packed',model,source=source)
    for store in (legacy,packed):
        store.manifest.update(dirty=True,transaction=dict(journal=None,cursor='cursor-initial.pt'))
        store.prepare_stage(0,list(store.entries))
    assert packed.profile_snapshot()['fsync_count']<legacy.profile_snapshot()['fsync_count']
    assert packed.profile_snapshot()['prepare_stage_bytes']==legacy.profile_snapshot()['prepare_stage_bytes']
    cached=PackedCachedDiskState.create(tmp_path/'cached',model,source=source,cache_bytes=2**20)
    reference=PackedDiskState.create(tmp_path/'reference',model,source=source)
    batches=[torch.tensor([[1,2,3,4]])]
    for store in (cached,reference):SparseDiskDLS(model,store).step(batches)
    for name in reference.entries:torch.testing.assert_close(cached.weight(name),reference.weight(name))
    cached=PackedCachedDiskState.open(cached.root,model,cache_bytes=2**20)
    assert cached.manifest['journal_format']=='packed-v1'


def test_journal_format_guards_and_dirty_cached_open(tmp_path):
    from morph.model.delta import DeltaModel
    from streamtrain.disk import DiskState
    from streamtrain.packed_state import PackedDiskState,PackedCachedDiskState
    source=DeltaModel(tiny_config())
    with torch.device('meta'):model=DeltaModel(tiny_config())
    legacy=DiskState.create(tmp_path/'legacy',model,source=source)
    with pytest.raises(ValueError,match='format'):PackedDiskState.open(legacy.root,model)
    packed=PackedCachedDiskState.create(tmp_path/'packed',model,source=source,cache_bytes=2**20)
    packed.manifest.update(dirty=True,transaction=dict(journal=None,cursor='cursor-initial.pt'))
    packed.prepare_stage(0,list(packed.entries))
    with pytest.raises(ValueError,match='format'):DiskState.open(packed.root,model)
    reopened=PackedCachedDiskState.open(packed.root,model,cache_bytes=2**20)
    reopened.recover_stage()
    assert reopened.manifest['transaction']['journal'] is None


def test_clean_packed_store_rejects_legacy_writer_but_exports(tmp_path):
    from morph.model.delta import DeltaModel
    from streamtrain.disk import DiskState
    from streamtrain.packed_state import PackedDiskState
    from streamtrain.export import export_fp16,load_fp16
    source=DeltaModel(tiny_config())
    with torch.device('meta'):model=DeltaModel(tiny_config())
    packed=PackedDiskState.create(tmp_path/'packed',model,source=source)
    before=(packed.root/'manifest.json').read_bytes()
    with pytest.raises(ValueError,match='format'):DiskState.open(packed.root,model)
    assert (packed.root/'manifest.json').read_bytes()==before
    exported=export_fp16(packed.root,tmp_path/'fp16')
    loaded=load_fp16(tmp_path/'fp16')
    assert exported['step']==0
    for name,param in loaded.named_parameters():
        torch.testing.assert_close(param,dict(source.named_parameters())[name].half(),rtol=0,atol=0)
