"""Tiny native Delta full-optimizer FP8 transport gate; never quantizes masters.

This validates E4M3FN copies unpacked into FP16 compute, not FP8 GEMMs or full5B
quality. Both runs start from identical FP32 weights/moments and use the same
batch, loss scaling, learning rate and clipping threshold.
"""
import json
import math
from pathlib import Path
import platform
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from morph.delta_config import DeltaConfig
from morph.model.delta import DeltaModel
from streamtrain.disk import DiskState
from streamtrain.sparse_disk import SparseDiskDLS
from streamtrain.precision import fp8_eligible


def acceptance(reference, approximate, update_relative_l2):
    loss_drift=abs(approximate['loss']-reference['loss'])/max(abs(reference['loss']),1e-12)
    norm_drift=abs(approximate['grad_norm']-reference['grad_norm'])/max(abs(reference['grad_norm']),1e-12)
    finite=all(math.isfinite(value) for value in (*reference.values(),*approximate.values(),update_relative_l2))
    return dict(relative_loss_drift=loss_drift,relative_global_norm_drift=norm_drift,
                update_relative_l2=update_relative_l2,finite=finite,
                limits=dict(relative_loss=.02,relative_global_norm=.10,update_relative_l2=.10),
                passed=finite and loss_drift<=.02 and norm_drift<=.10 and update_relative_l2<=.10)


def run(output='docs/delta_fp8_precision_gate.json',device='cuda'):
    target=torch.device(device)
    if target.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA qualification requested but unavailable')
    torch.set_num_threads(2)
    # fork_rng keeps this explicit qualification from perturbing caller sampling.
    with torch.random.fork_rng(devices=[target.index or torch.cuda.current_device()] if target.type=='cuda' else []):
        torch.manual_seed(43)
        cfg=DeltaConfig(d_model=32,num_layers=2,d_ff=64,d_state=4,head_dim=16,
                        d_graph=16,graph_nodes=4,expert_count=4,expert_depth=2,
                        expert_ffn=64,latent_dim=16,memory_dim=8,proof_enabled=False)
        source=DeltaModel(cfg)
        batches=[torch.randint(256,(2,17)),torch.randint(256,(2,17))]
        eligible=[name for name,_ in source.named_parameters() if fp8_eligible(name)]
        if not any(name.startswith('experts.') for name in eligible):
            raise RuntimeError('gate must exercise expert temporary quantization')
        with torch.device('meta'):template=DeltaModel(cfg)
        with tempfile.TemporaryDirectory(prefix='delta-precision-') as directory:
            stores=[DiskState.create(Path(directory)/name,template,source=source) for name in ('fp16','fp8')]
            trainers=[SparseDiskDLS(template,store,device=str(target),dtype=torch.float16,lr=.0001,
                                    async_mode=target.type=='cuda',cpu_prefetch=True,fp8_transfer=enabled)
                      for store,enabled in zip(stores,(False,True))]
            try:
                results=[trainer.step(batches,max_norm=.2) for trainer in trainers]
                if target.type=='cuda':torch.cuda.synchronize(target)
                update_norm=error_norm=dot=approx_norm=0.
                state_finite=True
                for name,initial in source.named_parameters():
                    base=stores[0].weight(name).double()-initial.detach().double()
                    approx=stores[1].weight(name).double()-initial.detach().double()
                    update_norm+=base.square().sum().item()
                    approx_norm+=approx.square().sum().item()
                    error_norm+=(approx-base).square().sum().item()
                    dot+=(base*approx).sum().item()
                    state_finite &= bool(torch.isfinite(base).all() and torch.isfinite(approx).all())
                drift=math.sqrt(error_norm/max(update_norm,1e-30))
                assessment=acceptance(results[0],results[1],drift)
                assessment['passed'] &= state_finite and update_norm>0
                report=dict(schema_version=1,architecture='delta',config=cfg.to_dict(),
                            platform=dict(python=platform.python_version(),torch=torch.__version__,
                                          cuda=torch.version.cuda,device=str(target),
                                          name=torch.cuda.get_device_name(target) if target.type=='cuda' else platform.processor(),
                                          capability=list(torch.cuda.get_device_capability(target)) if target.type=='cuda' else None),
                            seed=43,training_tokens=64,microbatches=2,
                            precision='FP32 canonical masters and Adam moments; FP16 compute',
                            fp8='E4M3FN temporary transport; 256-element blocks; FP32 scales; unpack to FP16',
                            eligible_parameters=eligible,reference=results[0],approximate=results[1],
                            learning_rate=.0001,max_norm=.2,
                            clip_reference=min(1.,.2/(results[0]['grad_norm']+1e-6)),
                            clip_approximate=min(1.,.2/(results[1]['grad_norm']+1e-6)),
                            update_cosine=dot/max(math.sqrt(update_norm*approx_norm),1e-30),
                            transfer_bytes=[trainer.transfer_bytes for trainer in trainers],
                            **assessment,
                            claims='Tiny two-pass full optimizer numerical gate only; no full5B, proof-quality, convergence, FP8-GEMM, or throughput claim')
            finally:
                for trainer in trainers:trainer.close()
    destination=Path(output);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,allow_nan=False))
    return report


if __name__=='__main__':
    run(sys.argv[1] if len(sys.argv)>1 else 'docs/delta_fp8_precision_gate.json',
        sys.argv[2] if len(sys.argv)>2 else 'cuda')
