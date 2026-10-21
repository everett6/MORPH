"""Combined optional execution paths on small actual FP16 CUDA staging slots."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from morph.delta_config import DeltaConfig
from morph.model.delta import DeltaModel
from streamtrain.disk import DiskState
from streamtrain.packed_state import PackedCachedDiskState
from streamtrain.single_pass import SinglePassDiskDLS
from streamtrain.sparse_disk import SparseDiskDLS


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_combined_chunk_group_packed_fp16_updates(tmp_path):
    torch.manual_seed(124)
    cfg=DeltaConfig(d_model=16,num_layers=1,d_ff=32,d_state=2,head_dim=8,expert_count=8,
                    step_expert_count=2,routing_seed=3,expert_depth=1,expert_ffn=32,
                    latent_dim=8,memory_dim=4,proof_enabled=False)
    source=DeltaModel(cfg)
    with torch.device('meta'):
        reference=DeltaModel(cfg)
        optimized=DeltaModel(replace(cfg,expert_group_size=2,memory_projection_backend='batched'))
    a=DiskState.create(tmp_path/'reference',reference,source=source)
    b=PackedCachedDiskState.create(tmp_path/'optimized',optimized,source=source,cache_bytes=2**20)
    first=SparseDiskDLS(reference,a,device='cuda',dtype=torch.float16,async_mode=True,cpu_prefetch=True)
    second=SinglePassDiskDLS(optimized,b,device='cuda',dtype=torch.float16,async_mode=True,
                             cpu_prefetch=True,chunk_microbatches=2,stash_budget_bytes=2**20)
    batches=[torch.randint(0,256,(2,9)) for _ in range(3)]
    try:
        initial={name:source.state_dict()[name].float().numpy().copy() for name in a.entries}
        reference.set_routing_step(0)
        baseline=first.step(batches,max_norm=.1);actual=second.step(batches,max_norm=.1)
        assert actual['loss']==pytest.approx(baseline['loss'],rel=.005)
        assert actual['grad_norm']==pytest.approx(baseline['grad_norm'],rel=.02)
        assert a.manifest['parameter_steps']==b.manifest['parameter_steps']
        error=scale=0.
        for name,entry in a.entries.items():
            x=np.memmap(a.root/entry['file'],dtype=np.float32,mode='r',shape=(3,entry['size']))
            y=np.memmap(b.root/entry['file'],dtype=np.float32,mode='r',shape=(3,entry['size']))
            direction=x[0]-initial[name].reshape(-1)
            error+=float(np.square((x[0]-y[0]).astype(np.float64)).sum())
            scale+=float(np.square(direction.astype(np.float64)).sum())
            np.testing.assert_allclose(x[1:],y[1:],atol=2e-5,rtol=.03)
        assert (error/max(scale,1e-30))**.5<.1
        assert second.peak_boundary_microbatches==2
        assert len(second.active_expert_union)<=2
        assert b.manifest['step']==1 and not b.manifest['dirty']
    finally:first.close();second.close()
