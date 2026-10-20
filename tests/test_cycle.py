import fcntl
import json
import os
import signal
import pytest
import torch
from data.shards import compile_shard
from morph import MorphConfig
from streamtrain.cycle import run_cycle


def fixture(tmp_path):
    cfg=MorphConfig(vocab_size=256,d_model=8,num_layers=1,d_ff=16,d_state=2,head_dim=4,d_graph=4)
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg.to_dict()))
    shard=tmp_path/'data';compile_shard([dict(family='a',split='train',text='theorem local math '*8),dict(family='b',split='validation',text='validation math '*8)],shard)
    return path,shard,tmp_path/'run'


def test_exact_target_tokens_and_safe_resume(tmp_path):
    torch.set_num_threads(2);cfg,data,root=fixture(tmp_path)
    options=dict(device='cpu',precision='fp32',length=4,accumulation=2,validation_every=1,backend='parallel')
    assert run_cycle(cfg,data,root,17,max_steps=1,**options)==8
    assert json.loads((root/'status.json').read_text())['safe_to_resume']
    assert run_cycle(cfg,data,root,17,resume=True,**options)==17
    records=[json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
    assert [r['step_target_tokens'] for r in records]==[8,8,1]
    assert records[-1]['processed_target_tokens']==17
    assert all('validation_loss' in record for record in records)
    with pytest.raises(ValueError,match='settings'):
        run_cycle(cfg,data,root,18,resume=True,**options)


def test_run_lock_prevents_second_writer(tmp_path):
    cfg,data,root=fixture(tmp_path);root.mkdir()
    with (root/'run.lock').open('w') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='another process'):
            run_cycle(cfg,data,root,8,device='cpu',precision='fp32',backend='parallel')


def test_sigterm_finishes_step_then_stops(tmp_path,monkeypatch):
    from streamtrain.disk import DiskDLS
    cfg,data,root=fixture(tmp_path);original=DiskDLS.step
    def step(self,*args,**kwargs):
        result=original(self,*args,**kwargs);os.kill(os.getpid(),signal.SIGTERM);return result
    monkeypatch.setattr(DiskDLS,'step',step)
    assert run_cycle(cfg,data,root,16,device='cpu',precision='fp32',length=4,accumulation=1,validation_every=0,backend='parallel')==4
    assert json.loads((root/'status.json').read_text())['safe_to_resume']
