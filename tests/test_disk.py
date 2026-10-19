import copy
import pytest
import torch
from morph import MorphConfig, MorphModel
from streamtrain.disk import DiskState, DiskDLS


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_disk_two_pass_matches_accumulated_adamw(device, tmp_path):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(3)
    cfg = MorphConfig(vocab_size=16, d_model=8, num_layers=2, d_ff=16, d_state=2,
                      head_dim=4, graph_enabled=True, graph_interval=1, graph_nodes=3, d_graph=4)
    original = MorphModel(cfg)
    reference = copy.deepcopy(original).to(device)
    with torch.device('meta'):
        template = MorphModel(cfg)
    store = DiskState.create(tmp_path / 'state', template, source=original)
    trainer = DiskDLS(template, store, device=device, lr=.003, async_mode=device == 'cuda')
    optimizer = torch.optim.AdamW(reference.parameters(), lr=.003, weight_decay=.01, foreach=False)
    batches = [torch.tensor([[1, 2, 3, 4]]), torch.tensor([[3, 4, 5, 6]])]
    for step in range(3):
        optimizer.zero_grad()
        for batch in batches:
            (reference.loss(batch.to(device)) / len(batches)).backward()
        expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), .05).item()
        optimizer.step()
        result = trainer.step(batches, max_norm=.05)
        assert abs(result['grad_norm'] - expected_norm) < 2e-5
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(store.weight(name), parameter.cpu(), rtol=5e-4, atol=2e-6, msg=name)
        assert store.manifest['step'] == step + 1
    reopened = DiskState.open(tmp_path / 'state', template)
    assert reopened.manifest['step'] == 3
    reopened.begin_step()
    with pytest.raises(RuntimeError, match='incomplete'):
        DiskState.open(tmp_path / 'state', template)


def test_disk_clean_resume_and_failure_semantics(tmp_path):
    from streamtrain.disk import disk_train
    cfg = MorphConfig(vocab_size=8, d_model=8, num_layers=1, d_ff=16, d_state=2, head_dim=4)
    ids = torch.tensor([[1, 2, 3, 1, 2, 3]])
    root, other = tmp_path / 'resume', tmp_path / 'whole'
    disk_train(cfg, ids, root, 1, sequence_length=3)
    disk_train(cfg, ids, root, 3, resume=True, sequence_length=3)
    disk_train(cfg, ids, other, 3, sequence_length=3)
    with torch.device('meta'):
        template = MorphModel(cfg)
    a, b = DiskState.open(root, template), DiskState.open(other, template)
    for name in a.entries:
        torch.testing.assert_close(a.weight(name), b.weight(name), rtol=0, atol=0)
    trainer = DiskDLS(template, a)
    original_update = a.update
    def fail_after_one(*args, **kwargs):
        original_update(*args, **kwargs)
        raise OSError('simulated disk failure')
    a.update = fail_after_one
    with pytest.raises(OSError):
        trainer.step([ids])
    recovered=DiskState.open(root,template)
    DiskDLS(template,recovered).resume_step()
    DiskDLS(template,b).step([ids])
    for name in b.entries:
        torch.testing.assert_close(recovered.weight(name),b.weight(name),rtol=0,atol=0)
    assert recovered.manifest['step']==b.manifest['step']==4
    assert not b.manifest['dirty']


def test_disk_bf16_cuda_smoke(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from streamtrain.disk import disk_train
    cfg = MorphConfig(vocab_size=8, d_model=8, num_layers=1, d_ff=16, d_state=2, head_dim=4)
    ids = torch.tensor([[1, 2, 3, 1, 2, 3]])
    result = disk_train(cfg, ids, tmp_path / 'bf16', 2, device='cuda', dtype=torch.bfloat16, accumulation=2, sequence_length=3)
    assert result['step'] == 2 and result['final']['loss'] > 0


@pytest.mark.parametrize('point',['journal_durable','partial_state_write','cursor_file_durable','cursor_manifest_durable','before_final_manifest','final_manifest_durable'])
def test_transaction_fault_boundaries_resume_exactly(tmp_path,point):
    cfg=MorphConfig(vocab_size=8,d_model=8,num_layers=1,d_ff=16,d_state=2,head_dim=4)
    original=MorphModel(cfg)
    with torch.device('meta'):template=MorphModel(cfg)
    a=DiskState.create(tmp_path/'interrupted',template,source=original)
    b=DiskState.create(tmp_path/'expected',template,source=original)
    batches=[torch.tensor([[1,2,3,4]]),torch.tensor([[2,3,4,5]])]
    DiskDLS(template,b).step(batches)
    def inject(actual):
        if actual==point:raise OSError('injected '+point)
    a.fault=inject
    with pytest.raises(OSError):DiskDLS(template,a).step(batches)
    reopened=DiskState.open(a.root,template)
    if reopened.manifest.get('transaction'):DiskDLS(template,reopened).resume_step()
    assert reopened.manifest['step']==1 and not reopened.manifest['dirty']
    for name in b.entries:
        torch.testing.assert_close(reopened.weight(name),b.weight(name),rtol=0,atol=0)
        assert (reopened.root/reopened.entries[name]['file']).read_bytes()==(b.root/b.entries[name]['file']).read_bytes()
    DiskDLS(template,reopened).step(batches)
    DiskDLS(template,b).step(batches)
    for name in b.entries:
        assert (reopened.root/reopened.entries[name]['file']).read_bytes()==(b.root/b.entries[name]['file']).read_bytes()
