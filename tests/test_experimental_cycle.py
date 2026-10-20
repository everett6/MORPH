"""Separate-run experimental cycle; local fake HTTP rows, never model downloads."""
import json
import pytest
from test_delta_cycle import config_file,stream_source
from streamtrain import delta_cycle


def local_config(tmp_path):
    path=config_file(tmp_path);record=json.loads(path.read_text());record['step_expert_count']=1
    path.write_text(json.dumps(record));return path


def options(config,run_dir,source):
    return dict(config=config,run_dir=run_dir,source=source,device='cpu',precision='fp32',
                length=4,batch_size=2,accumulation=2,backend='parallel',export=False,
                engine='single-pass',cache_bytes=100_000,stash_budget_bytes=10_000_000)


def test_experimental_single_pass_cycle_exact_budget_cache_metrics_and_resume(tmp_path):
    path=local_config(tmp_path);server,source=stream_source()
    try:
        opts=options(path,tmp_path/'experiment',source)
        result=delta_cycle.run_delta_cycle(**opts,target_tokens=13,max_steps=1)
        assert result['processed_target_tokens']==12 and result['phase']=='stopped'
        result=delta_cycle.run_delta_cycle(**opts,target_tokens=13,max_steps=1,resume=True)
        assert result['processed_target_tokens']==13 and result['phase']=='complete'
        settings=json.loads((tmp_path/'experiment'/'settings.json').read_text())
        assert settings['experimental']['engine']=='single-pass'
        metrics=[json.loads(line) for line in (tmp_path/'experiment'/'metrics.jsonl').read_text().splitlines()]
        assert metrics[0]['gradient_stash_bytes']>0
        assert metrics[0]['state_cache']['cache_budget_bytes']==100_000
        assert not (tmp_path/'experiment'/'state'/'gradient-stash').exists()
    finally:server.shutdown();server.server_close()


def test_experimental_preflight_combines_cache_stash_and_host_memory(tmp_path,monkeypatch):
    path=local_config(tmp_path);server,source=stream_source()
    try:
        opts=options(path,tmp_path/'experiment',source)
        baseline=delta_cycle.preflight(**opts)
        required=baseline['required_host_bytes']
        monkeypatch.setattr(delta_cycle,'available_ram',lambda:int(required/0.8)-1)
        report=delta_cycle.preflight(**opts)
        assert not report['ready'] and 'host memory' in ' '.join(report['blockers'])
        assert report['budget']['gradient_stash_upper_bytes']>0
        assert report['budget']['state_cache_budget_bytes']==100_000
        assert not (tmp_path/'experiment'/'state').exists()
    finally:server.shutdown();server.server_close()


def test_cycle_rejects_local_roster_two_pass_and_too_small_stash(tmp_path):
    path=local_config(tmp_path);server,source=stream_source()
    try:
        opts=options(path,tmp_path/'experiment',source)
        with pytest.raises(MemoryError,match='single-pass'):
            delta_cycle.run_delta_cycle(**(opts|dict(engine='two-pass')),target_tokens=13,max_steps=1)
        with pytest.raises(MemoryError,match='stash'):
            delta_cycle.run_delta_cycle(**(opts|dict(stash_budget_bytes=1)),target_tokens=13,max_steps=1)
        assert not (tmp_path/'experiment'/'state').exists()
    finally:server.shutdown();server.server_close()


def test_default_cycle_keeps_old_settings_and_default_config_resume(tmp_path):
    path=config_file(tmp_path);server,source=stream_source()
    try:
        opts=options(path,tmp_path/'ordinary',source)
        for key in ('engine','cache_bytes','stash_budget_bytes'):opts.pop(key)
        delta_cycle.run_delta_cycle(**opts,target_tokens=13,max_steps=1)
        settings_path=tmp_path/'ordinary'/'settings.json';settings=json.loads(settings_path.read_text())
        assert 'experimental' not in settings
        settings['config'].pop('step_expert_count',None);settings['config'].pop('routing_seed',None)
        settings_path.write_text(json.dumps(settings))
        result=delta_cycle.run_delta_cycle(**opts,target_tokens=13,max_steps=1,resume=True)
        assert result['processed_target_tokens']==13
    finally:server.shutdown();server.server_close()


def test_saved_gradient_cycle_resume_commits_pending_source_once(tmp_path,monkeypatch):
    from streamtrain.state_cache import CachedDiskState
    path=local_config(tmp_path);server,source=stream_source()
    def fault(self,point):
        if point=='partial_state_write':raise OSError('interrupted stage apply')
    try:
        opts=options(path,tmp_path/'experiment',source)
        monkeypatch.setattr(CachedDiskState,'fault',fault)
        with pytest.raises(OSError,match='interrupted'):
            delta_cycle.run_delta_cycle(**opts,target_tokens=12,max_steps=1)
        progress=json.loads((tmp_path/'experiment'/'stream-state.json').read_text())
        assert progress['pending_targets']==12 and progress['processed_target_tokens']==0
        monkeypatch.setattr(CachedDiskState,'fault',lambda self,point:None)
        result=delta_cycle.run_delta_cycle(**opts,target_tokens=12,max_steps=1,resume=True)
        assert result['step']==1 and result['processed_target_tokens']==12
        progress=json.loads((tmp_path/'experiment'/'stream-state.json').read_text())
        assert progress['source']['pending'] is None and progress['pending_targets']==0
    finally:server.shutdown();server.server_close()


def test_preflight_reserves_durable_gradient_disk_space(tmp_path,monkeypatch):
    path=local_config(tmp_path);server,source=stream_source()
    try:
        opts=options(path,tmp_path/'experiment',source)
        baseline=delta_cycle.preflight(**opts)
        stash=baseline['budget']['gradient_stash_disk_reserve_bytes']
        free=baseline['required_free_disk_bytes']-stash//2
        assert free>baseline['required_free_disk_bytes']-stash
        monkeypatch.setattr(delta_cycle.shutil,'disk_usage',lambda _:type('Usage',(),dict(free=free))())
        report=delta_cycle.preflight(**opts)
        assert not report['ready'] and 'disk' in ' '.join(report['blockers'])
    finally:server.shutdown();server.server_close()
