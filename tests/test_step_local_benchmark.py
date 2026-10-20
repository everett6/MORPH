from dataclasses import replace
from benchmarks.step_local_training import run_benchmark,default_config


def test_four_variant_benchmark_has_equal_targets_and_optimizer_parity():
    cfg=replace(default_config(),d_model=8,d_ff=16,d_state=2,head_dim=4,expert_count=4,expert_depth=1,expert_ffn=16,latent_dim=4)
    result=run_benchmark(cfg,steps=1,length=4,batch_size=1,accumulation=2,quota=2)
    assert len(result['variants'])==4
    assert {v['total_targets'] for v in result['variants']}=={8}
    assert all(row['clocks_equal'] for row in result['comparisons'].values())
    assert all(row['max_abs_weight']<1e-6 for row in result['comparisons'].values())
    assert result['variants'][-1]['cache_stats']['hits']>0
