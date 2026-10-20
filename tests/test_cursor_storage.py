import json
import pytest
import torch
from morph import MorphConfig, MorphModel
from streamtrain.disk import DiskState, durable_json, durable_torch


def state(tmp_path):
    cfg = MorphConfig(vocab_size=8,d_model=8,num_layers=1,d_ff=16,d_state=2,head_dim=4)
    model = MorphModel(cfg)
    store = DiskState.create(tmp_path/'state', model, source=model)
    durable_torch(store.root/'cursor-initial.pt', dict(cotangents=[None],tied=None))
    store.manifest.update(dirty=True, transaction=dict(cursor='cursor-initial.pt',next_stage=2,journal=None))
    durable_json(store.root/'manifest.json', store.manifest)
    return store, model


def test_tied_cursor_uses_one_durable_payload_and_atomic_clocks(tmp_path):
    store, _ = state(tmp_path)
    store.prepare_stage(2, [])
    tied = torch.arange(20.)
    store.commit_stage(2,[None],tied,parameter_steps={'embedding.weight':1})
    path = store.root/'transaction-tied.pt'; stamp = path.stat().st_mtime_ns
    assert torch.load(store.root/'cursor-2.pt', weights_only=True)['tied_ref'] == path.name
    store.prepare_stage(1,[])
    store.commit_stage(1,[None],tied,parameter_steps={'other':1})
    assert path.stat().st_mtime_ns == stamp
    torch.testing.assert_close(store.load_cursor()['tied'], tied)
    assert store.manifest['parameter_steps'] == {'embedding.weight':1,'other':1}
    assert store.load_cursor()['parameter_steps'] == {'embedding.weight':1,'other':1}
    store.finish_step()
    assert not path.exists()


@pytest.mark.parametrize('fault_point', ['tied_file_durable', 'cursor_file_durable'])
def test_cursor_fault_does_not_publish_clock_or_orphan_tied_reference(tmp_path, fault_point):
    store, model = state(tmp_path)
    store.prepare_stage(2, [])
    def fail(point):
        if point == fault_point: raise OSError('crash')
    store.fault = fail
    with pytest.raises(OSError): store.commit_stage(2,[None],torch.ones(20),parameter_steps={'p':9})
    reopened = DiskState.open(store.root, model); reopened.recover_stage()
    assert 'parameter_steps' not in reopened.manifest
    assert reopened.load_cursor()['tied'] is None
    reopened.prepare_stage(2, [])
    reopened.commit_stage(2,[None],torch.full((20,), 2.),parameter_steps={'p':1})
    torch.testing.assert_close(reopened.load_cursor()['tied'], torch.full((20,), 2.))
    assert json.loads((store.root/'manifest.json').read_text())['parameter_steps'] == {'p':1}


def test_legacy_tensor_cursor_remains_readable(tmp_path):
    store, _ = state(tmp_path)
    durable_torch(store.root/'cursor-initial.pt', dict(cotangents=[None],tied=torch.ones(3)))
    torch.testing.assert_close(store.load_cursor()['tied'], torch.ones(3))


def test_delta_config_reopens_its_own_manifest(tmp_path):
    from morph.delta_config import DeltaConfig
    from morph.model.delta import DeltaModel
    cfg = DeltaConfig(d_model=16,num_layers=1,d_ff=32,d_state=4,head_dim=8,d_graph=8,
                      graph_nodes=3,expert_count=2,expert_depth=1,expert_ffn=16,latent_dim=8)
    model = DeltaModel(cfg)
    store = DiskState.create(tmp_path/'delta', model, source=model)
    assert DiskState.open(store.root, model).manifest['config']['model_type'] == 'delta'
