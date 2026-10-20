import copy
import json
import importlib.util
import pytest
import torch
from torch.nn import functional as F


def tiny_config():
    from morph.delta_config import DeltaConfig
    return DeltaConfig(vocab_size=256,d_model=8,num_layers=1,d_ff=16,expand=2,d_state=2,head_dim=4,
                       expert_count=3,expert_depth=1,expert_ffn=16,latent_dim=4,
                       graph_enabled=False,proof_enabled=False)


def test_sparse_streaming_matches_conditional_adamw_and_skips_unused(tmp_path):
    from morph.model.delta import DeltaModel
    from streamtrain.disk import DiskState
    from streamtrain.sparse_disk import SparseDiskDLS
    torch.manual_seed(11)
    cfg=tiny_config();original=DeltaModel(cfg);reference=copy.deepcopy(original)
    with torch.device('meta'):template=DeltaModel(cfg)
    store=DiskState.create(tmp_path/'state',template,source=original)
    trainer=SparseDiskDLS(template,store,lr=.002)
    opt=torch.optim.AdamW(reference.parameters(),lr=.002,weight_decay=.01,foreach=False)
    batches=[torch.tensor([[1,2,3,4]]),torch.tensor([[3,4,5,6]])]
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        for b in batches:(reference.loss(b)/len(batches)).backward()
        norm=torch.nn.utils.clip_grad_norm_(reference.parameters(),1.).item();opt.step()
        result=trainer.step(batches)
        assert result['grad_norm']==pytest.approx(norm,rel=2e-5,abs=2e-6)
        for name,p in reference.named_parameters():
            torch.testing.assert_close(store.weight(name),p,atol=2e-6,rtol=5e-4,msg=name)
        for name,p in reference.named_parameters():
            expected=int(opt.state[p]['step']) if p in opt.state else 0
            assert store.manifest.get('parameter_steps',{}).get(name,0)==expected
    trainer.close()


def test_sparse_recovery_and_fp16_export(tmp_path):
    from morph.model.delta import DeltaModel
    from streamtrain.disk import DiskState
    from streamtrain.sparse_disk import SparseDiskDLS
    from streamtrain.export import export_fp16,load_fp16
    cfg=tiny_config();original=DeltaModel(cfg)
    with torch.device('meta'):template=DeltaModel(cfg)
    a=DiskState.create(tmp_path/'a',template,source=original)
    b=DiskState.create(tmp_path/'b',template,source=original)
    batches=[torch.tensor([[1,2,3,4]])]
    SparseDiskDLS(template,b).step(batches)
    def fault(point):
        if point=='partial_state_write':raise OSError('crash')
    a.fault=fault
    with pytest.raises(OSError,match='crash'):SparseDiskDLS(template,a).step(batches)
    a=DiskState.open(a.root,template);SparseDiskDLS(template,a).resume_step()
    assert a.manifest['parameter_steps']==b.manifest['parameter_steps']
    for n in b.entries:assert (a.root/a.entries[n]['file']).read_bytes()==(b.root/b.entries[n]['file']).read_bytes()
    result=export_fp16(a.root,tmp_path/'fp16')
    assert result['config']['model_type']=='delta'
    loaded=load_fp16(tmp_path/'fp16')
    assert loaded.embedding.weight is loaded.lm_head.weight
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.float16):
        logits=loaded(batches[0][:,:-1])['logits']
    assert torch.isfinite(logits).all()


def test_cuda_accumulates_stage_gradients_before_host_transfer(tmp_path):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from morph.model.delta import DeltaModel
    from streamtrain.disk import DiskState
    from streamtrain.sparse_disk import SparseDiskDLS
    cfg=tiny_config();source=DeltaModel(cfg)
    with torch.device('meta'):template=DeltaModel(cfg)
    store=DiskState.create(tmp_path/'state',template,source=source)
    trainer=SparseDiskDLS(template,store,device='cuda',async_mode=True)
    trainer.step([torch.tensor([[1,2,3,4]]),torch.tensor([[3,4,5,6]]),torch.tensor([[6,5,4,3]])])
    stage_parameters=sum(p.numel() for spec in trainer.specs for _,p in spec)
    assert trainer.transfer_bytes['d2h_gradients']<=2*4*stage_parameters
    trainer.close()
