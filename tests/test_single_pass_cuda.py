"""Small CUDA qualification; no datasets, model downloads, or production state."""
import pytest
import torch

from morph.delta_config import DeltaConfig
from morph.model.delta import DeltaModel
from streamtrain.disk import DiskState
from streamtrain.single_pass import SinglePassDiskDLS
from streamtrain.sparse_disk import SparseDiskDLS
from streamtrain.state_cache import CachedDiskState


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA qualification requires a GPU')
def test_fp16_async_cached_single_pass_matches_two_pass_across_rosters(tmp_path):
    cfg=DeltaConfig(d_model=16,num_layers=1,d_ff=32,d_state=2,head_dim=8,
                    expert_count=8,step_expert_count=2,routing_seed=17,
                    latent_dim=8,expert_depth=1,expert_ffn=24,memory_dim=4,
                    proof_enabled=False,ssm_backend='parallel')
    torch.manual_seed(26)
    source=DeltaModel(cfg)
    with torch.device('meta'):
        one_model=DeltaModel(cfg);two_model=DeltaModel(cfg)
    cached=CachedDiskState.create(tmp_path/'single',one_model,source=source,cache_bytes=2**20)
    plain=DiskState.create(tmp_path/'two',two_model,source=source)
    one=SinglePassDiskDLS(one_model,cached,device='cuda',dtype=torch.float16,
                          async_mode=True,cpu_prefetch=True,stash_budget_bytes=2**20)
    two=SparseDiskDLS(two_model,plain,device='cuda',dtype=torch.float16,
                      async_mode=True,cpu_prefetch=True)
    batches=[torch.tensor([[1,2,3,4,5],[8,7,6,5,4]]),torch.tensor([[3,4,5,6,7]])]
    try:
        for step in range(2):
            two_model.set_routing_step(step)
            assert one.step(batches,max_norm=.02)==two.step(batches,max_norm=.02)
            assert cached.manifest['parameter_steps']==plain.manifest['parameter_steps']
            for name,entry in cached.entries.items():
                assert (cached.root/entry['file']).read_bytes()==(plain.root/entry['file']).read_bytes(),name
        assert cached.manifest['step']==2
        assert cached.cache_stats['hits']>0
        assert not (cached.root/'gradient-stash').exists()
    finally:
        one.close();two.close()
