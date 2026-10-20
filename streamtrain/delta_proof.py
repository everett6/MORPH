"""Local-trace, resource-qualified sparse proof training; no network access."""
import argparse
from dataclasses import replace
import fcntl
import json
from pathlib import Path
import shutil
import torch
from morph.delta_config import DeltaConfig
from morph.model.delta import DeltaModel
from morph.training.proof_runner import load_examples,collate_proof,run_proof_training
from .cycle import available_ram
from .disk import estimate_disk_budget

DTYPES={'fp16':torch.float16,'bf16':torch.bfloat16,'fp32':torch.float32}


def load_config(path,backend='triton'):
    return replace(DeltaConfig.from_file(path),ssm_backend=backend)


def proof_preflight(config,traces,run_dir,device='cuda',precision='fp16',sequence_length=128,batch_size=1,accumulation=1,export=False):
    cfg=replace(config,proof_enabled=True)
    if min(sequence_length,batch_size,accumulation)<1:raise ValueError('positive sequence/batch/accumulation required')
    if 40*Path(traces).stat().st_size>.5*available_ram():raise MemoryError('proof trace parsing envelope exceeds available RAM')
    examples=load_examples(traces,'train')
    if not examples:raise ValueError('no training traces')
    # Validate every statement, graph, action and payload before allocating weights.
    for example in examples:collate_proof([example],cfg,sequence_length)
    with torch.device('meta'):model=DeltaModel(cfg)
    budget=estimate_disk_budget(model,batch_size,sequence_length,accumulation,DTYPES[precision])
    boundary=accumulation*batch_size*sequence_length*(cfg.d_model+cfg.expert_count)*4*len(model.stream_stages())
    extra=max(0,boundary-budget['boundary_bytes'])
    budget['estimated_host_bytes']+=3*extra+40*Path(traces).stat().st_size
    budget['disk_reserve_bytes']+=3*extra
    budget['boundary_bytes']=boundary
    parent=Path(run_dir).resolve()
    while not parent.exists():parent=parent.parent
    exists=(Path(run_dir)/'state'/'manifest.json').exists()
    disk=budget['disk_reserve_bytes']+(0 if exists else budget['state_disk_bytes'])+(2*budget['parameters'] if export else 0)
    free_disk=shutil.disk_usage(parent).free;free_ram=available_ram();blockers=[]
    if disk>free_disk:blockers.append('insufficient disk for canonical state, recovery and requested export')
    host=budget['estimated_host_bytes']+(budget['estimated_cuda_bytes'] if device=='cpu' else 0)
    if host>.8*free_ram:blockers.append('host memory exceeds 80% of available RAM')
    if str(device).startswith('cuda'):
        if not torch.cuda.is_available():blockers.append('CUDA unavailable')
        elif budget['estimated_cuda_bytes']>.85*torch.cuda.mem_get_info(device)[0]:blockers.append('GPU envelope exceeds 85% free VRAM')
    return dict(ready=not blockers,blockers=blockers,parameters=budget['parameters'],canonical_state_bytes=budget['state_disk_bytes'],
                required_disk_bytes=disk,available_disk_bytes=free_disk,available_ram_bytes=free_ram,budget=budget,training_examples=len(examples))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/morph_delta_tiny.json');parser.add_argument('--traces',required=True)
    parser.add_argument('--backend',choices=('reference','parallel','triton'),default='triton')
    parser.add_argument('--run-dir',required=True);parser.add_argument('--steps',type=int,default=1)
    parser.add_argument('--device',default='cuda');parser.add_argument('--precision',choices=DTYPES,default='fp16')
    parser.add_argument('--sequence-length',type=int,default=128);parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--accumulation',type=int,default=1);parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--continue-lm-state',action='store_true',help='explicit clean LM-to-proof transition in same run; --steps is ADDITIONAL')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--export',action='store_true',help='write a bounded FP16 export after clean training')
    args=parser.parse_args();torch.set_num_threads(args.threads)
    cfg=load_config(args.config,args.backend)
    report=proof_preflight(cfg,args.traces,args.run_dir,args.device,args.precision,args.sequence_length,args.batch_size,args.accumulation,args.export)
    print(json.dumps(report,indent=2),flush=True)
    if args.preflight:return
    if not report['ready']:raise MemoryError('proof preflight failed; no weights allocated')
    root=Path(args.run_dir);root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('another process owns this run')
        total_steps=args.steps
        if args.continue_lm_state:
            if args.resume:raise ValueError('choose continue-lm-state for first transition or resume for later proof continuation')
            manifest=json.loads((root/'state'/'manifest.json').read_text())
            total_steps+=manifest['step']
        model,losses=run_proof_training(cfg,args.traces,total_steps,args.lr,args.device,engine='disk',
            sequence_length=args.sequence_length,batch_size=args.batch_size,accumulation=args.accumulation,
            dtype=DTYPES[args.precision],state_dir=root/'state',resume=args.resume or args.continue_lm_state,seed=args.seed,continue_lm_state=args.continue_lm_state)
        try:
            print(json.dumps(dict(losses=losses,step=model.executor.store.manifest['step'])),flush=True)
            if args.export:
                from .export import export_fp16
                export_fp16(root/'state',root/f"fp16-step-{model.executor.store.manifest['step']:09d}",lock_held=True)
        finally:model.close()


if __name__=='__main__':main()
