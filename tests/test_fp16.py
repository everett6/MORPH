import copy
import json
import numpy as np
import pytest
import torch
from morph import MorphConfig,MorphModel
from streamtrain.disk import DiskState,DiskDLS
from streamtrain.engine import StreamTrainEngine
from streamtrain.precision import pack_fp8,unpack_fp8,fp8_eligible
from streamtrain.export import export_fp16


def config():return MorphConfig(vocab_size=256,d_model=16,num_layers=2,d_ff=32,d_state=4,head_dim=8,d_graph=8,graph_nodes=2,graph_enabled=True)


def test_fp8_format_and_fp16_export(tmp_path):
    torch.manual_seed(2);values=torch.randn(519)
    encoded,scales=pack_fp8(values);restored=unpack_fp8(encoded,scales,torch.float32)
    assert (restored-values).norm()/values.norm()<.04
    assert not fp8_eligible('embedding.weight') and not fp8_eligible('blocks.0.ssm.a_log')
    model=MorphModel(config());store=DiskState.create(tmp_path/'run'/'state',model,source=model)
    manifest=export_fp16(store.root,tmp_path/'export',chunk_elements=17)
    assert manifest['parameters']==sum(p.numel() for p in model.parameters())
    for name,p in model.named_parameters():
        values=np.fromfile(tmp_path/'export'/manifest['weights'][name]['file'],dtype='<f2')
        np.testing.assert_array_equal(values,p.detach().numpy().astype(np.float16).flatten())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required for FP16 compute')
def test_fp16_scaled_streamed_and_disk_update_recovery(tmp_path):
    torch.manual_seed(12);torch.set_num_threads(2);source=MorphModel(config());tokens=torch.randint(256,(1,9))
    reference=copy.deepcopy(source).cuda().half();reference.zero_grad()
    with torch.autocast('cuda',dtype=torch.float16):loss=reference.loss(tokens.cuda())
    (loss*128).backward()
    master=copy.deepcopy(source)
    for p,leaf in zip(master.parameters(),reference.parameters()):p.grad=leaf.grad.float().cpu()/128
    streamed=copy.deepcopy(source);engine=StreamTrainEngine(streamed,'cuda',True,torch.float16)
    engine.backward(tokens,loss_scale=128)
    for a,b in zip(master.parameters(),streamed.parameters()):
        torch.testing.assert_close(a.grad.cpu(),b.grad,rtol=.04,atol=3e-5)
    norm=torch.nn.utils.clip_grad_norm_(master.parameters(),1.).item()
    optimizer=torch.optim.AdamW(master.parameters(),lr=.001,weight_decay=.01,foreach=False);optimizer.step()
    store=DiskState.create(tmp_path/'base',source,source=source);disk=DiskDLS(source,store,'cuda',torch.float16,.001,True)
    result=disk.step([tokens]);assert abs(result['grad_norm']-norm)<max(1e-4,norm*.01)
    for name,p in master.named_parameters():
        entry=store.entries[name];arr=np.memmap(store.root/entry['file'],dtype=np.float32,mode='r',shape=(3,entry['size']))
        torch.testing.assert_close(torch.from_numpy(arr[0].copy()).reshape_as(p),p.detach().cpu(),rtol=.01,atol=8e-5)
    faulty=DiskState.create(tmp_path/'fault',source,source=source);executor=DiskDLS(source,faulty,'cuda',torch.float16,.001,True)
    def fail(point):
        if point=='partial_state_write':raise RuntimeError('interruption')
    faulty.fault=fail
    with pytest.raises(RuntimeError,match='interruption'):executor.step([tokens])
    reopened=DiskState.open(tmp_path/'fault',source);recovery=DiskDLS(source,reopened,'cuda',torch.float16,.001,True);recovery.resume_step()
    for entry in store.entries.values():
        a=np.fromfile(store.root/entry['file'],dtype=np.float32);b=np.fromfile(reopened.root/entry['file'],dtype=np.float32)
        np.testing.assert_allclose(a,b,atol=2e-7,rtol=2e-5)
    assert reopened.manifest['loss_scaler']==store.manifest['loss_scaler']

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_fp16_overflow_retries_before_mutation(tmp_path,monkeypatch):
    source=MorphModel(config());store=DiskState.create(tmp_path/'state',source,source=source)
    executor=DiskDLS(source,store,'cuda',torch.float16,.001,True)
    original=executor.reverse;calls=[]
    def reverse(*args,**kwargs):
        calls.append(executor.loss_scale)
        if len(calls)==1:
            assert not store.manifest['dirty'] and store.manifest['step']==0
            return float('inf'),5.
        return original(*args,**kwargs)
    monkeypatch.setattr(executor,'reverse',reverse)
    executor.step([torch.randint(256,(1,9))])
    assert calls[:2]==[128.,64.] and store.manifest['loss_scaler']['scale']==64.


def test_fp16_export_mapped_roundtrip_and_tamper(tmp_path):
    from streamtrain.export import load_fp16
    torch.manual_seed(3);model=MorphModel(config());store=DiskState.create(tmp_path/'run'/'state',model,source=model)
    export_fp16(store.root,tmp_path/'model');loaded=load_fp16(tmp_path/'model')
    assert all(p.dtype==torch.float16 and p.device.type=='cpu' for p in loaded.parameters())
    tokens=torch.randint(256,(1,9))
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.float16):
        torch.testing.assert_close(loaded(tokens).logits,copy.deepcopy(model).half()(tokens).logits,atol=0,rtol=0)
    path=tmp_path/'model'/'00000.bin'
    with path.open('r+b') as handle:handle.write(b'\x00\x00')
    with pytest.raises(ValueError,match='hash'):load_fp16(tmp_path/'model')
