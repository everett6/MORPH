"""Token-budgeted native sparse training from RAM-only Hugging Face row streams."""
import argparse
import base64
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import time
import torch
from data.hf_stream import HFByteStream
from morph.delta_config import DeltaConfig
from morph.model.delta import DeltaModel
from .cycle import available_ram
from .disk import DiskState,durable_json,estimate_disk_budget
from .sparse_disk import SparseDiskDLS
from .export import export_fp16


DTYPES={'fp16':torch.float16,'bf16':torch.bfloat16,'fp32':torch.float32}
DEFAULT_STASH_BUDGET=4*2**30
PRODUCTION_RUN='runs/morph-delta5b-hf-math'


def experimental_options(cfg,engine,cache_bytes,stash_budget_bytes,chunk_microbatches=0,journal_format='legacy'):
    if engine not in ('two-pass','single-pass'):raise ValueError('invalid training engine')
    if type(cache_bytes) is not int or cache_bytes<0:raise ValueError('nonnegative integer cache budget required')
    if type(stash_budget_bytes) is not int or stash_budget_bytes<1:raise ValueError('positive integer stash budget required')
    if type(chunk_microbatches) is not int or chunk_microbatches<0:raise ValueError('nonnegative chunk microbatch count required')
    if chunk_microbatches and engine!='single-pass':raise ValueError('bounded chunk accumulation requires single-pass')
    if journal_format not in ('legacy','packed'):raise ValueError('unsupported journal format')
    return (engine!='two-pass' or cache_bytes>0 or stash_budget_bytes!=DEFAULT_STASH_BUDGET or cfg.step_expert_count>0
            or chunk_microbatches>0 or journal_format!='legacy' or cfg.expert_group_size>1 or cfg.memory_projection_backend!='reference')


def source_from_file(path):
    return HFByteStream(**json.loads(Path(path).read_text()))


def preflight(config,run_dir,source=None,source_config='configs/hf_math_stream.json',device='cuda',
              precision='fp16',length=128,batch_size=4,accumulation=8,backend='triton',export=True,
              engine='two-pass',cache_bytes=0,stash_budget_bytes=DEFAULT_STASH_BUDGET,
              chunk_microbatches=0,journal_format='legacy'):
    cfg=replace(DeltaConfig.from_file(config),ssm_backend=backend)
    experimental=experimental_options(cfg,engine,cache_bytes,stash_budget_bytes,chunk_microbatches,journal_format)
    if min(length,batch_size,accumulation)<1:raise ValueError('positive sequence/batch/accumulation required')
    with torch.device('meta'):model=DeltaModel(cfg)
    window=min(accumulation,chunk_microbatches) if chunk_microbatches else accumulation
    budget=estimate_disk_budget(model,batch_size,length,window,DTYPES[precision])
    stages=model.stream_stages()
    # Every expert boundary includes differentiable router gates. Count the
    # actual execution stages, rather than only the shared backbone's layers.
    boundary=window*batch_size*length*(cfg.d_model+cfg.expert_count)*4*len(stages)
    extra=max(0,boundary-budget['boundary_bytes'])
    budget['estimated_host_bytes']+=extra*3
    budget['disk_reserve_bytes']+=extra*3
    budget['boundary_bytes']=boundary
    budget['activation_window_microbatches']=window
    # The pending byte targets remain bounded CPU input; only activations use windows.
    pending_input=accumulation*batch_size*(length+1)*8
    budget['estimated_host_bytes']+=pending_input
    budget['pending_input_bytes']=pending_input
    group=cfg.expert_group_size;slot_bytes=4 if precision=='fp32' else 2
    group_extra=0
    if group>1:
        width=cfg.latent_dim if cfg.latent_experts else cfg.d_model
        expert_size=max(sum(p.numel() for p in expert.parameters()) for expert in model.experts)
        group_extra=(2*group*expert_size*slot_bytes
                     +slot_bytes*group*batch_size*length*cfg.expert_depth*(3*cfg.expert_ffn+6*width)
                     +group*batch_size*length*(width*4+cfg.d_model*slot_bytes*2))
    if cfg.fast_memory_enabled and cfg.memory_projection_backend=='batched':
        group_extra+=batch_size*length*((3*cfg.memory_dim+2)*4+cfg.d_model*slot_bytes)+batch_size*cfg.d_model*slot_bytes
    router_cotangents=2*window*batch_size*length*cfg.expert_count*4
    recurrence=(batch_size*length*cfg.memory_dim**2*4*8 if cfg.fast_memory_enabled else 0)
    budget['execution_extra_bytes']=group_extra
    budget['router_cotangent_bytes']=router_cotangents
    budget['fast_memory_recurrence_bytes']=recurrence
    budget['estimated_cuda_bytes']+=group_extra+router_cotangents+recurrence
    parent=Path(run_dir).resolve()
    while not parent.exists():parent=parent.parent
    exists=(Path(run_dir)/'state'/'manifest.json').exists()
    required_disk=budget['disk_reserve_bytes']+(0 if exists else budget['state_disk_bytes'])
    if export:required_disk+=budget['parameters']*(2 if exists else 4)
    free_disk=shutil.disk_usage(parent).free;free_ram=available_ram();blockers=[]
    if cfg.step_expert_count and engine!='single-pass':
        blockers.append('step-local rosters require single-pass in the training cycle until two-pass roster recovery is qualified')
    quota=cfg.step_expert_count or cfg.expert_count
    expert_sizes=sorted((sum(p.numel() for p in expert.parameters()) for expert in model.experts),reverse=True)
    shared=sum(p.numel() for name,p in model.named_parameters() if not name.startswith('experts.'))
    active_upper=shared+sum(expert_sizes[:quota])
    stash_upper=active_upper*4 if engine=='single-pass' else 0
    if stash_upper>stash_budget_bytes:blockers.append('active FP32 gradient stash exceeds configured stash budget')
    # Tensor serialization headers and immutable metadata need disk space too.
    stash_disk=stash_upper+len(list(model.named_parameters()))*4096+16*2**20 if stash_upper else 0
    required_disk+=stash_disk
    budget.update(active_parameter_upper_bound=active_upper,gradient_stash_upper_bytes=stash_upper,
                  gradient_stash_disk_reserve_bytes=stash_disk,state_cache_budget_bytes=cache_bytes)

    if required_disk>free_disk:blockers.append('disk space below canonical state + recovery + FP16 export requirement')
    required_host=budget['estimated_host_bytes']+(budget['estimated_cuda_bytes'] if torch.device(device).type=='cpu' else 0)+stash_upper+cache_bytes
    if required_host>free_ram*.8:blockers.append('host memory envelope exceeds 80% of available RAM')
    free_cuda=None
    if device.startswith('cuda'):
        if not torch.cuda.is_available():blockers.append('CUDA is unavailable')
        else:
            free_cuda=torch.cuda.mem_get_info(device)[0]
            if budget['estimated_cuda_bytes']>free_cuda*.85:blockers.append('GPU memory envelope exceeds 85% of free VRAM')
    source=source or source_from_file(source_config)
    before=source.state_dict()
    try:
        # Bounded access check, restored afterward: no source targets consumed.
        if before.get('pending') is None:source.next_batch(1,16)
    finally:source.load_state_dict(before)
    return dict(ready=not blockers,blockers=blockers,parameters=budget['parameters'],
                expert_parameters=sum(p.numel() for p in model.experts.parameters()),
                budget=budget,required_host_bytes=required_host,required_free_disk_bytes=required_disk,available_disk_bytes=free_disk,
                experimental=experimental,engine=engine,cache_bytes=cache_bytes,stash_budget_bytes=stash_budget_bytes,
                chunk_microbatches=chunk_microbatches,journal_format=journal_format,
                available_ram_bytes=free_ram,available_cuda_bytes=free_cuda,
                source_fingerprint=source.source_fingerprint,dataset_revision=source.resolved_revision,
                dataset_cache_bytes=0,precision=precision,optimizer='FP32 conditional AdamW',
                batch_size=batch_size,sequence_length=length,accumulation=accumulation)


def run_delta_cycle(config,run_dir,target_tokens=1_000_000_000,source=None,source_config='configs/hf_math_stream.json',
                    device='cuda',precision='fp16',length=128,batch_size=4,accumulation=8,lr=1e-4,
                    max_steps=None,backend='triton',resume=False,export=True,seed=42,cpu_prefetch=True,export_every=1000,
                    engine='two-pass',cache_bytes=0,stash_budget_bytes=DEFAULT_STASH_BUDGET,
              chunk_microbatches=0,journal_format='legacy'):
    if target_tokens<1 or lr<=0 or (max_steps is not None and max_steps<1):raise ValueError('invalid token/step/learning-rate schedule')
    source=source or source_from_file(source_config)
    report=preflight(config,run_dir,source=source,device=device,precision=precision,length=length,
                     batch_size=batch_size,accumulation=accumulation,backend=backend,export=export,
                     engine=engine,cache_bytes=cache_bytes,stash_budget_bytes=stash_budget_bytes,
                     chunk_microbatches=chunk_microbatches,journal_format=journal_format)
    if not report['ready']:raise MemoryError(json.dumps(report))
    cfg=replace(DeltaConfig.from_file(config),ssm_backend=backend)
    with torch.device('meta'):model=DeltaModel(cfg)
    experimental=experimental_options(cfg,engine,cache_bytes,stash_budget_bytes,chunk_microbatches,journal_format)
    root=Path(run_dir)
    if experimental and root.name==Path(PRODUCTION_RUN).name:
        raise ValueError('experimental engine/cache/roster options require a separate experiment run directory')
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'run.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:lock.close();raise RuntimeError('another process owns this training run')
    stop=[False];trainer=None
    previous={s:signal.signal(s,lambda *_:stop.__setitem__(0,True)) for s in (signal.SIGTERM,signal.SIGINT)}
    def status(phase,**values):
        result=dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),**values)
        durable_json(root/'status.json',result);return result
    settings=dict(config=cfg.to_dict(),source_fingerprint=source.source_fingerprint,target_tokens=target_tokens,
                  batch_size=batch_size,sequence_length=length,accumulation=accumulation,precision=precision,
                  lr=lr,seed=seed,optimizer='conditional-adamw-fp32',device_type=torch.device(device).type,export_every=export_every)
    if experimental:
        settings['experimental']=dict(engine=engine,cache_bytes=cache_bytes,stash_budget_bytes=stash_budget_bytes,
                     chunk_microbatches=chunk_microbatches,journal_format=journal_format)
    if journal_format=='packed':
        from .packed_state import PackedDiskState,PackedCachedDiskState
        state_class=PackedCachedDiskState if cache_bytes else PackedDiskState
        state_options=dict(cache_bytes=cache_bytes) if cache_bytes else {}
    elif cache_bytes:
        from .state_cache import CachedDiskState
        state_class=CachedDiskState;state_options=dict(cache_bytes=cache_bytes)
    else:state_class=DiskState;state_options={}
    try:
        settings_path=root/'settings.json';stream_path=root/'stream-state.json'
        if resume:
            saved_settings=json.loads(settings_path.read_text())
            # New default-valued config fields must not invalidate an old run.
            saved_settings['config']=DeltaConfig(**saved_settings['config']).to_dict()
            if 'experimental' in saved_settings:
                saved_settings['experimental'].setdefault('chunk_microbatches',0)
                saved_settings['experimental'].setdefault('journal_format','legacy')
            if saved_settings!=settings:raise ValueError('resume model/data/optimizer settings differ')
            store=state_class.open(root/'state',model,**state_options)
            progress=json.loads(stream_path.read_text());source.load_state_dict(progress['source'])
        else:
            if settings_path.exists() or (root/'state').exists():raise ValueError('run exists; use --resume')
            durable_json(settings_path,settings)
            progress=dict(step=0,processed_target_tokens=0,pending_targets=0,source=source.state_dict())
            durable_json(stream_path,progress)
            last=[0.]
            def initializing(name,total,size):
                if time.monotonic()-last[0]>2:
                    status('initializing',parameters=report['parameters'],tensor=name);last[0]=time.monotonic()
            store=state_class.create(root/'state',model,seed=seed,progress=initializing,**state_options)
        if engine=='single-pass':
            from .single_pass import SinglePassDiskDLS
            engine_class=SinglePassDiskDLS;engine_options=dict(stash_budget_bytes=stash_budget_bytes,chunk_microbatches=chunk_microbatches)
        else:engine_class=SparseDiskDLS;engine_options={}
        trainer=engine_class(model,store,device=device,dtype=DTYPES[precision],lr=lr,
                            async_mode=device.startswith('cuda'),cpu_prefetch=cpu_prefetch,**engine_options)
        last=[0.]
        def phase(name,index,micro=None):
            now=time.monotonic()
            if now-last[0]>2:
                status('training',training_phase=name,stage=index,step=store.manifest['step'],
                       processed_target_tokens=progress['processed_target_tokens'],target_tokens=target_tokens)
                last[0]=now
        trainer.progress=phase
        if store.manifest.get('transaction'):trainer.resume_step()
        # Optimizer commit wins a crash between state commit and stream commit.
        # Reconcile exactly once; pending bytes are the durable replay record.
        def reconcile():
            if progress['pending_targets']:
                if store.manifest['step']==progress['step']+1:
                    source.commit_batch();progress['processed_target_tokens']+=progress['pending_targets']
                    progress.update(step=store.manifest['step'],pending_targets=0,source=source.state_dict())
                    progress.pop('pending_microbatch_size',None)
                    durable_json(stream_path,progress)
                elif store.manifest['step']!=progress['step']:raise ValueError('source cursor and optimizer step disagree')
            elif store.manifest['step']!=progress['step']:raise ValueError('source cursor and optimizer step disagree')
        reconcile();start_step=store.manifest['step']
        def write_export():
            output=root/f'fp16-step-{store.manifest["step"]:09d}'
            if not (output/'manifest.json').exists():
                if shutil.disk_usage(root).free<report['parameters']*2+report['budget']['disk_reserve_bytes']:
                    raise OSError('insufficient space for FP16 export and recovery reserve')
                status('exporting_fp16',step=store.manifest['step'],processed_target_tokens=progress['processed_target_tokens'],target_tokens=target_tokens)
                export_fp16(store.root,output,lock_held=True)
            durable_json(root/'fp16-latest.json',dict(path=str(output),step=store.manifest['step']))
            for old in root.glob('fp16-step-*'):
                if old!=output and (old/'manifest.json').exists():shutil.rmtree(old)
        while progress['processed_target_tokens']<target_tokens and not stop[0] and (max_steps is None or store.manifest['step']-start_step<max_steps):
            if progress['pending_targets']:
                pending=source.state_dict()['pending'];tokens=source.next_batch(pending['requested_batch_size'],pending['length'])
                micro_size=progress['pending_microbatch_size']
            else:
                remaining=target_tokens-progress['processed_target_tokens']
                window=min(length,remaining);windows=min(batch_size*accumulation,max(1,remaining//window))
                try:tokens=source.next_batch(windows,window)
                except StopIteration:
                    tail=base64.b64decode(source.state_dict()['committed']['buffer_b64'])
                    if len(tail)<2:raise RuntimeError('Hugging Face split exhausted before the requested token budget')
                    tokens=source.next_batch(1,min(len(tail)-1,remaining))
                micro_size=batch_size if tokens.shape[0]%batch_size==0 else 1
                count=tokens.shape[0]*(tokens.shape[1]-1)
                progress.update(pending_targets=count,pending_microbatch_size=micro_size,source=source.state_dict())
                durable_json(stream_path,progress)
            batches=list(tokens.split(micro_size));count=progress['pending_targets']
            profile_before=trainer.profile_snapshot();storage_before=store.profile_snapshot()
            io_before=_io();started=time.perf_counter();cpu_started=time.process_time()
            metrics=trainer.step(batches)
            profile_after=trainer.profile_snapshot(profile_before)
            seconds=time.perf_counter()-started;cpu_seconds=time.process_time()-cpu_started;io_after=_io()
            storage_after=store.profile_snapshot()
            storage_delta={k:v-storage_before.get(k,0) for k,v in storage_after.items()}
            storage_delta['journal_payload_bytes']=storage_delta.get('prepare_stage_bytes',0)
            metrics['phase_profile']=dict(**profile_after,process_cpu_seconds=cpu_seconds,storage=storage_delta,
                timing_semantics='wall and CUDA stream intervals overlap; CUDA intervals include stream idle; do not sum them')
            reconcile()
            metrics.update(step=store.manifest['step'],processed_target_tokens=progress['processed_target_tokens'],
                           step_target_tokens=count,step_seconds=seconds,tokens_per_second=count/max(seconds,1e-9),
                           active_experts=_active_expert_count(trainer),
                           process_io_bytes={k:io_after[k]-io_before[k] for k in io_before},
                           transfer_bytes=dict(trainer.transfer_bytes),loss_scale=trainer.loss_scale)
            if experimental:
                metrics.update(engine=engine,routing_plan=model.get_routing_plan(),
                               gradient_stash_bytes=getattr(trainer,'last_stash_bytes',0),
                               peak_boundary_microbatches=getattr(trainer,'peak_boundary_microbatches',len(batches)),
                               completed_microbatches=getattr(trainer,'completed_microbatches',len(batches)),
                               journal_format=journal_format,chunk_microbatches=chunk_microbatches,
                               state_cache=getattr(store,'cache_stats',dict(cache_bytes=0,cache_budget_bytes=0)),
                               active_parameter_upper_bound=report['budget']['active_parameter_upper_bound'])
            with (root/'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(metrics)+'\n');handle.flush();os.fsync(handle.fileno())
            print(json.dumps(metrics),flush=True)
            if export and (store.manifest['step']==1 or (export_every and store.manifest['step']%export_every==0)):write_export()
        trainer.close();trainer=None
        if export and store.manifest['step']:
            write_export()
        phase_name='complete' if progress['processed_target_tokens']==target_tokens else 'stopped'
        return status(phase_name,step=store.manifest['step'],processed_target_tokens=progress['processed_target_tokens'],
                      target_tokens=target_tokens,safe_to_resume=True)
    except Exception as error:
        state_path=root/'state'/'manifest.json'
        checkpoint=json.loads(state_path.read_text()) if state_path.exists() else {}
        status('error',error=str(error),safe_to_resume=bool(checkpoint and (not checkpoint.get('dirty') or checkpoint.get('transaction'))))
        raise
    finally:
        if trainer is not None:trainer.close()
        for s,handler in previous.items():signal.signal(s,handler)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def _active_expert_count(trainer):
    union=getattr(trainer,'active_expert_union',None)
    if union is not None:return len(union)
    selected=set()
    for spec,active in zip(trainer.specs,trainer.active_stages):
        if not active:continue
        for _,parameter in spec:
            name=trainer.canonical[id(parameter)]
            pieces=name.split('.')
            if len(pieces)>2 and pieces[0]=='experts' and pieces[1].isdecimal():selected.add(int(pieces[1]))
    return len(selected)


def _io():
    return {key:int(value) for key,value in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines()) if key in ('read_bytes','write_bytes')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/morph_delta_5b.json')
    parser.add_argument('--source-config',default='configs/hf_math_stream.json')
    parser.add_argument('--run-dir')
    parser.add_argument('--experiment-run-dir',help='separate directory for experimental engine/cache/roster runs')
    parser.add_argument('--engine',choices=['two-pass','single-pass'],default='two-pass')
    parser.add_argument('--cache-bytes',type=int,default=0)
    parser.add_argument('--stash-budget-bytes',type=int,default=DEFAULT_STASH_BUDGET)
    parser.add_argument('--chunk-microbatches',type=int,default=0,help='maximum microbatches with live activation boundaries')
    parser.add_argument('--journal-format',choices=['legacy','packed'],default='legacy')
    parser.add_argument('--target-tokens',type=int,default=1_000_000_000)
    parser.add_argument('--device',default='cuda');parser.add_argument('--precision',choices=DTYPES,default='fp16')
    parser.add_argument('--sequence-length',type=int,default=128);parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--accumulation',type=int,default=8);parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--max-steps',type=int);parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--backend',choices=['parallel','reference','triton'],default='triton')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--no-export',action='store_true');parser.add_argument('--no-cpu-prefetch',action='store_true')
    parser.add_argument('--export-every',type=int,default=1000)
    args=parser.parse_args();torch.set_num_threads(args.threads)
    cfg=DeltaConfig.from_file(args.config)
    experimental=experimental_options(cfg,args.engine,args.cache_bytes,args.stash_budget_bytes,args.chunk_microbatches,args.journal_format)
    if args.experiment_run_dir and not experimental:parser.error('--experiment-run-dir requires experimental options')
    if args.experiment_run_dir and args.run_dir:parser.error('choose --run-dir or --experiment-run-dir')
    run_dir=args.experiment_run_dir or args.run_dir or ('runs/morph-delta5b-experimental' if experimental else PRODUCTION_RUN)
    options=dict(config=args.config,run_dir=run_dir,source_config=args.source_config,device=args.device,
                 precision=args.precision,length=args.sequence_length,batch_size=args.batch_size,accumulation=args.accumulation,
                 backend=args.backend,export=not args.no_export,engine=args.engine,
                 cache_bytes=args.cache_bytes,stash_budget_bytes=args.stash_budget_bytes,
                 chunk_microbatches=args.chunk_microbatches,journal_format=args.journal_format)
    if args.preflight:
        result=preflight(**options);print(json.dumps(result,indent=2));return 0 if result['ready'] else 2
    result=run_delta_cycle(**options,target_tokens=args.target_tokens,lr=args.lr,max_steps=args.max_steps,
                           resume=args.resume,cpu_prefetch=not args.no_cpu_prefetch,export_every=args.export_every)
    print(json.dumps(result));return 0


if __name__=='__main__':raise SystemExit(main())
