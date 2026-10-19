"""Preflight must reject unsafe proof sizes without allocating real weights."""
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('engine',['resident','dls','async','disk'])
def test_proof_7b_preflight_is_meta_only(engine,tmp_path):
    cmd=[sys.executable,'train.py','--config','configs/morph_7b.json','--proof-traces','docs/proof_examples.jsonl','--engine',engine,'--state-dir',str(tmp_path/'state')]
    inspected=subprocess.run(cmd+['--inspect'],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert inspected.returncode==0,inspected.stderr
    report=json.loads(inspected.stdout)
    assert report['parameters']>6_900_000_000
    assert report['qualification']['estimated_host_bytes']>0
    # Supply a deterministic tiny available-memory budget via direct main test
    # below; normal CLI inspect above never allocates model weights.

def test_async_proof_guard_rejects_before_real_allocation(monkeypatch,tmp_path):
    import train
    from morph import MorphModel
    original=MorphModel.__init__
    def init(self,*args,**kwargs):
        import torch
        assert torch.empty(()).device.type=='meta','attempted real allocation before guard'
        original(self,*args,**kwargs)
    monkeypatch.setattr(MorphModel,'__init__',init)
    original_read=Path.read_text
    def read(path,*args,**kwargs):
        if str(path)=='/proc/meminfo':return 'MemAvailable: 1024 kB\n'
        return original_read(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    monkeypatch.setattr(sys,'argv',['train.py','--proof-traces','docs/proof_examples.jsonl','--engine','async'])
    with pytest.raises(SystemExit) as error:train.main()
    assert error.value.code==2

@pytest.mark.parametrize('engine',['resident','disk'])
def test_proof_resume_binds_content_and_optimizer_settings(engine,tmp_path):
    import torch
    from morph import MorphConfig
    from morph.training.proof_runner import run_proof_training
    torch.set_num_threads(2)
    cfg=MorphConfig(vocab_size=256,d_model=8,num_layers=1,d_ff=16,d_state=2,head_dim=4,d_graph=4,graph_nodes=4,proof_enabled=True)
    data=tmp_path/'traces.jsonl';data.write_text((ROOT/'docs/proof_examples.jsonl').read_text())
    kwargs=dict(engine=engine,checkpoint=tmp_path/'model.pt' if engine=='resident' else None,state_dir=tmp_path/'store',sequence_length=32)
    run_proof_training(cfg,data,1,**kwargs)
    with pytest.raises(ValueError,match='settings differ'):run_proof_training(cfg,data,2,lr=.002,resume=True,**kwargs)
    with pytest.raises(ValueError,match='settings differ'):run_proof_training(cfg,data,2,seed=99,resume=True,**kwargs)
    data.write_text(data.read_text()+'\n')
    with pytest.raises(ValueError,match='settings differ'):run_proof_training(cfg,data,2,resume=True,**kwargs)
