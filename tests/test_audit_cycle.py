"""Audit options keep activation windows bounded and source commits exact."""
import json
import pytest
from test_delta_cycle import config_file,stream_source
from streamtrain import delta_cycle


def opts(tmp_path,source):
    path=config_file(tmp_path)
    config=json.loads(path.read_text());config['step_expert_count']=1
    path.write_text(json.dumps(config))
    return dict(config=path,run_dir=tmp_path/'audit',source=source,device='cpu',precision='fp32',
                length=4,batch_size=2,accumulation=8,backend='parallel',export=False,
                engine='single-pass',cache_bytes=100000,stash_budget_bytes=10000000)


def test_chunk_preflight_bounds_activation_window_but_keeps_target_budget(tmp_path):
    server,source=stream_source()
    try:
        options=opts(tmp_path,source)
        whole=delta_cycle.preflight(**options)
        bounded=delta_cycle.preflight(**options,chunk_microbatches=2,journal_format='packed')
        assert bounded['accumulation']==8
        assert bounded['budget']['activation_window_microbatches']==2
        assert bounded['budget']['boundary_bytes']<whole['budget']['boundary_bytes']
        assert bounded['required_host_bytes']<whole['required_host_bytes']
    finally:server.shutdown();server.server_close()


def test_packed_chunked_cycle_exact_tail_resume_and_phase_profile(tmp_path):
    server,source=stream_source()
    try:
        options=opts(tmp_path,source)|dict(chunk_microbatches=2,journal_format='packed')
        first=delta_cycle.run_delta_cycle(**options,target_tokens=67,max_steps=1)
        assert first['processed_target_tokens']==64
        final=delta_cycle.run_delta_cycle(**options,target_tokens=67,max_steps=1,resume=True)
        assert final['processed_target_tokens']==67 and final['step']==2
        records=[json.loads(line) for line in (tmp_path/'audit'/'metrics.jsonl').read_text().splitlines()]
        assert sum(row['step_target_tokens'] for row in records)==67
        assert records[0]['peak_boundary_microbatches']<=2
        assert records[0]['phase_profile']['process_cpu_seconds']>=0
        assert records[0]['phase_profile']['storage']['journal_payload_bytes']>0
        assert not (tmp_path/'audit'/'state'/'gradient-stash').exists()
    finally:server.shutdown();server.server_close()


def test_two_pass_rejects_chunk_option_before_allocating_state(tmp_path):
    server,source=stream_source()
    try:
        options=opts(tmp_path,source)|dict(engine='two-pass',chunk_microbatches=2)
        with pytest.raises(ValueError,match='single-pass'):
            delta_cycle.run_delta_cycle(**options,target_tokens=67,max_steps=1)
        assert not (tmp_path/'audit'/'state').exists()
    finally:server.shutdown();server.server_close()


def test_preflight_counts_router_cotangents_and_quadratic_memory(tmp_path):
    server,source=stream_source()
    try:
        options=opts(tmp_path,source)|dict(chunk_microbatches=2)
        small=delta_cycle.preflight(**options)
        cfg=json.loads(options['config'].read_text());cfg['memory_dim']=32
        options['config'].write_text(json.dumps(cfg))
        large=delta_cycle.preflight(**options)
        assert small['budget']['router_cotangent_bytes']==2*2*2*4*cfg['expert_count']*4
        assert large['budget']['fast_memory_recurrence_bytes']>=2*4*32**2*4*8
        assert large['budget']['fast_memory_recurrence_bytes']>small['budget']['fast_memory_recurrence_bytes']
        assert large['budget']['estimated_cuda_bytes']>small['budget']['estimated_cuda_bytes']
    finally:server.shutdown();server.server_close()


def test_grouped_expert_metric_counts_selected_parameters_only():
    from types import SimpleNamespace
    first=object();second=object();third=object()
    trainer=SimpleNamespace(stages=[SimpleNamespace(expert_indices=(0,1,2))],
                            active_stages=[True],specs=[[('experts.0.weight',first),('experts.2.weight',second)]],
                            canonical={id(first):'experts.0.weight',id(second):'experts.2.weight',id(third):'experts.1.weight'})
    assert delta_cycle._active_expert_count(trainer)==2
