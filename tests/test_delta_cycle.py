import json
import threading
from http.server import ThreadingHTTPServer
import pytest


def stream_source():
    from test_hf_stream import DatasetAPI
    from data.hf_stream import HFByteStream
    api=DatasetAPI([{'text':'0123456789abcdefghijklmno'}]*5)
    server=ThreadingHTTPServer(('127.0.0.1',0),api.handler())
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}'
    return server,HFByteStream('test/math','default','train','text','main',hub_api_url=url,datasets_api_url=url)


def config_file(tmp_path):
    from morph.delta_config import DeltaConfig
    cfg=DeltaConfig(d_model=8,num_layers=1,d_ff=16,d_state=2,head_dim=4,d_graph=4,graph_nodes=3,
                    expert_count=3,expert_depth=1,expert_ffn=16,latent_dim=4,memory_dim=4,
                    proof_enabled=True,statement_field_length=8)
    p=tmp_path/'config.json';p.write_text(json.dumps(cfg.to_dict()));return p


def test_online_cycle_exact_budget_and_resume(tmp_path):
    from streamtrain.delta_cycle import run_delta_cycle
    config=config_file(tmp_path);server,source=stream_source()
    try:
        a=run_delta_cycle(config,tmp_path/'run',target_tokens=13,source=source,device='cpu',precision='fp32',
                          length=4,batch_size=2,accumulation=2,max_steps=1,backend='parallel',export=False)
        assert a['processed_target_tokens']==12 and a['phase']=='stopped'
        # Use the same endpoint/fingerprint for resume.
        b=run_delta_cycle(config,tmp_path/'run',target_tokens=13,source=source,device='cpu',precision='fp32',
                          length=4,batch_size=2,accumulation=2,max_steps=1,backend='parallel',resume=True,export=True)
        assert b['processed_target_tokens']==13 and b['phase']=='complete'
        progress=json.loads((tmp_path/'run'/'stream-state.json').read_text())
        assert progress['processed_target_tokens']==13 and progress['source']['pending'] is None
        assert (tmp_path/'run'/'fp16-step-000000002'/'manifest.json').exists()
    finally:server.shutdown();server.server_close()


def test_preflight_refuses_space_before_state_allocation(tmp_path,monkeypatch):
    from streamtrain import delta_cycle
    config=config_file(tmp_path);server,source=stream_source()
    try:
        monkeypatch.setattr(delta_cycle.shutil,'disk_usage',lambda _:type('Usage',(),{'free':1})())
        report=delta_cycle.preflight(config,tmp_path/'run',source=source,device='cpu',precision='fp32')
        assert not report['ready'] and 'disk' in ' '.join(report['blockers']).lower()
        assert not (tmp_path/'run'/'state').exists()
    finally:server.shutdown();server.server_close()


def test_optimizer_commit_before_stream_commit_is_reconciled_once(tmp_path,monkeypatch):
    from streamtrain import delta_cycle
    config=config_file(tmp_path);server,source=stream_source()
    original=delta_cycle.SparseDiskDLS.step
    def crash_after_commit(self,batches,*args,**kwargs):
        original(self,batches,*args,**kwargs)
        raise OSError('crash after optimizer commit')
    try:
        monkeypatch.setattr(delta_cycle.SparseDiskDLS,'step',crash_after_commit)
        with pytest.raises(OSError,match='after optimizer commit'):
            delta_cycle.run_delta_cycle(config,tmp_path/'run',target_tokens=12,source=source,device='cpu',precision='fp32',
                                        length=4,batch_size=2,accumulation=2,max_steps=1,backend='parallel',export=False)
        pending=json.loads((tmp_path/'run'/'stream-state.json').read_text())
        assert pending['processed_target_tokens']==0 and pending['pending_targets']==12
        monkeypatch.setattr(delta_cycle.SparseDiskDLS,'step',original)
        result=delta_cycle.run_delta_cycle(config,tmp_path/'run',target_tokens=12,source=source,device='cpu',precision='fp32',
                                          length=4,batch_size=2,accumulation=2,max_steps=1,backend='parallel',export=False,resume=True)
        assert result['step']==1 and result['processed_target_tokens']==12
    finally:server.shutdown();server.server_close()
