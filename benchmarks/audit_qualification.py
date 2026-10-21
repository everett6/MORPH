"""Bounded small-model qualification; never accesses production checkpoints."""
import argparse
from datetime import datetime,timezone
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import time
import torch
from data.hf_stream import HFByteStream
from morph.delta_config import DeltaConfig
from morph.model.delta import DeltaModel
from streamtrain.disk import DiskState
from streamtrain.packed_state import PackedDiskState
from streamtrain.sparse_disk import SparseDiskDLS
from streamtrain.single_pass import SinglePassDiskDLS


def guarded_count(cfg):
    with torch.device('meta'):model=DeltaModel(cfg)
    count=sum(p.numel() for p in model.parameters())
    if count>5_000_000:raise MemoryError('qualification refuses configurations above 5 million parameters before allocation')
    # Canonical state, undo, gradient stash, inputs and snapshots stay below 300 MiB.
    if count*48>220*2**20:raise MemoryError('qualification temporary storage budget exceeded')
    return count


def collect_data(spec,steps,batch_size,length,accumulation):
    source=HFByteStream(**spec)
    train=[]
    for _ in range(steps):
        micros=[]
        for _ in range(accumulation):
            value=source.next_batch(batch_size,length)
            if value is None:raise ValueError('source exhausted')
            micros.append(value);source.commit_batch()
        train.append(micros)
    end=source.state_dict()['committed']['next_offset']
    heldout_start=1_000_000
    if end>=heldout_start:raise ValueError('training rows overlap heldout start')
    heldout_source=HFByteStream(**{**spec,'start_cursor':heldout_start})
    heldout=heldout_source.next_batch(batch_size,length)
    if heldout is None:raise ValueError('heldout source exhausted')
    heldout_source.commit_batch()
    metadata=dict(kind='revision-pinned Hugging Face rows',real_qualification=True,revision=source.resolved_revision,
        source_fingerprint=source.source_fingerprint,train_start_row=spec.get('start_cursor',0),
        train_end_row_exclusive=end,heldout_start_row=heldout_start,
        heldout_end_row_exclusive=heldout_source.state_dict()['committed']['next_offset'],row_disjoint=True)
    return train,heldout,metadata


def io_counters():
    return {k:int(v) for k,v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}


def difference(after,before):return {k:v-before.get(k,0) for k,v in after.items()}


def evaluate(source,store,batch,cfg,step):
    model=DeltaModel(cfg)
    with torch.no_grad():
        for name,p in model.named_parameters():p.copy_(store.weight(name) if store else dict(source.named_parameters())[name])
        model.set_routing_step(step)
        # loss shifts inputs/targets causally; each call starts fresh FP32 memory.
        return model.loss(batch).item()


def run_case(label,cfg,source,train,heldout,device,optimized):
    dtype=torch.float16 if device=='cuda' else torch.float32
    with torch.device('meta'):model=DeltaModel(cfg)
    with tempfile.TemporaryDirectory(prefix='.audit-qualification-',dir=Path.cwd()) as directory:
        store=(PackedDiskState if optimized else DiskState).create(Path(directory)/'state',model,source=source)
        engine=(SinglePassDiskDLS if optimized else SparseDiskDLS)(model,store,device=device,dtype=dtype,
            async_mode=device=='cuda',cpu_prefetch=device=='cuda',lr=.001,**(dict(chunk_microbatches=2,stash_budget_bytes=32*2**20) if optimized else {}))
        initial=evaluate(source,None,heldout,cfg,0);records=[]
        try:
            for step,batches in enumerate(train):
                model.set_routing_step(step)
                before=engine.profile_snapshot();disk_before=io_counters()
                if device=='cuda':torch.cuda.reset_peak_memory_stats()
                wall=time.perf_counter();cpu=time.process_time();result=engine.step(batches)
                if device=='cuda':torch.cuda.synchronize()
                wall=time.perf_counter()-wall;cpu=time.process_time()-cpu
                after=engine.profile_snapshot();disk_after=io_counters()
                targets=sum(batch.shape[0]*(batch.shape[1]-1) for batch in batches)
                records.append(dict(step=step+1,loss=result['loss'],grad_norm=result['grad_norm'],wall_seconds=wall,
                    cpu_process_seconds=cpu,byte_targets=targets,byte_targets_per_second=targets/wall,
                    profile_delta={k:difference(after[k],before[k]) for k in after},
                    process_io_delta=difference(disk_after,disk_before),
                    cuda_allocated_bytes=torch.cuda.memory_allocated() if device=='cuda' else 0,
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else 0,
                    gradient_stash_payload_bytes=getattr(engine,'last_stash_bytes',0),
                    active_expert_union=getattr(engine,'active_expert_union',None)))
            final=evaluate(source,store,heldout,cfg,len(train))
            weights={name:store.weight(name) for name in store.entries}
            total_time=sum(r['wall_seconds'] for r in records);total_targets=sum(r['byte_targets'] for r in records)
            report=dict(label=label,device=device,precision=str(dtype),config=cfg.to_dict(),steps=records,
                total_byte_targets=total_targets,total_step_wall_seconds=total_time,
                byte_targets_per_second=total_targets/total_time,
                threshold_2000_status='MET' if total_targets/total_time>2000 else 'NOT MET',
                heldout_initial_loss=initial,heldout_final_loss=final,heldout_improved=final<initial,
                final_parameter_clocks=store.manifest.get('parameter_steps',{}),
                final_temporary_file_bytes=sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file()))
            return report,weights
        finally:engine.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--output',default='docs/audit_qualification.json');args=parser.parse_args()
    run_started_utc=datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(2);torch.manual_seed(4201)
    base=replace(DeltaConfig.from_file('configs/morph_delta_tiny_local2.json'),proof_enabled=False,ssm_backend='parallel',step_expert_count=0)
    parameters=guarded_count(base);source=DeltaModel(base)
    spec=json.loads(Path('configs/hf_math_stream.json').read_text());spec.update(retries=2,timeout=15.)
    started=time.perf_counter()
    try:
        train,heldout,data=collect_data(spec,args.steps,4,128,64)
    except Exception as error:
        generator=torch.Generator().manual_seed(4202)
        train=[[torch.randint(0,256,(4,129),generator=generator) for _ in range(64)] for _ in range(args.steps)]
        heldout=torch.randint(0,256,(4,129),generator=generator)
        causes=[];cause=error
        while cause is not None:
            causes.append(f'{type(cause).__name__}: {cause}');cause=cause.__cause__
        data=dict(kind='synthetic fallback',real_qualification=False,error_type=type(error).__name__,error=str(error),error_chain=causes,row_disjoint=None)
    data['fetch_wall_seconds']=time.perf_counter()-started
    data['train_payload_sha256']=hashlib.sha256(b''.join(b.numpy().tobytes() for micros in train for b in micros)).hexdigest()
    controlled=[micros[:2] for micros in train]
    report=dict(parameters=parameters,torch_version=torch.__version__,source=data,variants=[],
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        tracked_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','HEAD'])).hexdigest(),
        source_file_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ('streamtrain','morph/model') for p in Path(directory).glob('*.py')},
        scope='Short three-commit small-model full-step qualification, not evidence of convergence or sustained 5B throughput. No 5B throughput or deadline projection.',
        timing_notes='Wall phases and CUDA stream intervals overlap and must not be added. Stream elapsed includes kernels and stream idle; it is not pure GPU busy time. Process I/O is process-accounted, not device-wide traffic.',
        recovery_evidence='tests/test_packed_state.py: multi-step partial-state-write recovery parity, truncated journal fail closed, wrong-format guards, cached recovery.',
        target_bytes_per_large_update=32768,
        threshold_scope='Optimizer-step-only timing on prebuffered RAM inputs; excludes source fetch, initialization, heldout evaluation and export.',
        cpu_prefetch=args.device=='cuda')
    baseline,weights=run_case('reference_two_pass',base,source,controlled,heldout,args.device,False);report['variants'].append(baseline)
    optimized_cfg=replace(base,expert_group_size=2,memory_projection_backend='batched')
    optimized,other=run_case('packed_single_pass_chunk2_group2_batched',optimized_cfg,source,controlled,heldout,args.device,True);report['variants'].append(optimized)
    report['controlled_comparison']=dict(same_initial_weights=True,same_byte_targets=True,max_absolute_final_weight_difference=max((weights[n]-other[n]).abs().max().item() for n in weights),parameter_clocks_equal=baseline['final_parameter_clocks']==optimized['final_parameter_clocks'])
    large,_=run_case('large_accumulation_32768_targets_chunk2',optimized_cfg,source,train,heldout,args.device,True);report['variants'].append(large)
    report['rng_settings']=dict(model_torch_seed=4201,synthetic_input_generator_seed=4202,threads=2,deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    report['run_started_utc']=run_started_utc
    report['run_completed_utc']=datetime.now(timezone.utc).isoformat()
    path=Path(args.output);path.write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Small-model audit qualification','',report['scope'],'',f"Data: {data['kind']}."]
    if 'error' in data:lines+=['',f"Real HF qualification failed: {data['error_type']}: {data['error']}"]
    lines+=['','| Variant | Byte targets/s | >2,000 status | Heldout initial → final | Improved |','|---|---:|---|---|---|']
    for r in report['variants']:lines.append(f"| {r['label']} | {r['byte_targets_per_second']:.1f} | {r['threshold_2000_status']} | {r['heldout_initial_loss']:.5f} → {r['heldout_final_loss']:.5f} | {r['heldout_improved']} |")
    lines+=['',report['threshold_scope'],f"Source fetch wall time: {data['fetch_wall_seconds']:.3f} seconds.",'',report['timing_notes'],'',report['recovery_evidence']]
    path.with_suffix('.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(output=str(path),source=data,results=[{k:r[k] for k in ('label','byte_targets_per_second','threshold_2000_status','heldout_improved')} for r in report['variants']])))

if __name__=='__main__':main()
