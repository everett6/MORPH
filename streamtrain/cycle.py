"""Persistent, local-data-only, token-budgeted disk AdamW training cycle."""
import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time
import torch
from torch.func import functional_call
from torch.nn import functional as F
from morph import MorphConfig,MorphModel
from data import MappedShard
from .disk import DiskState,DiskDLS,durable_json,estimate_disk_budget


def available_ram():
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    # Respect a tighter container limit when present, without assuming cgroupv2.
    for root in (Path('/sys/fs/cgroup'),Path('/sys/fs/cgroup/memory')):
        for limit_name,current_name in [('memory.max','memory.current'),('memory.limit_in_bytes','memory.usage_in_bytes')]:
            try:
                limit=(root/limit_name).read_text().strip()
                if limit!='max':available=min(available,max(0,int(limit)-int((root/current_name).read_text())))
            except (OSError,ValueError):pass
    return available


def validate(trainer,tokens,length,seed):
    batch=tokens.sample(torch.Generator().manual_seed(seed),1,length+1)
    state=(batch[:,:-1].to(trainer.device),None)
    for index,stage in enumerate(trainer.stages):
        slot=trainer.slots[index%2];trainer.load(index,slot)
        parameters=trainer.parameters(index,slot,False)
        with torch.no_grad(),torch.autocast(trainer.device.type,dtype=trainer.dtype,enabled=trainer.dtype in (torch.bfloat16,torch.float16)):
            output=functional_call(stage,parameters,state)
        trainer.release(slot)
        if index==len(trainer.stages)-1:
            return F.cross_entropy(output.float().flatten(0,1),batch[:,1:].to(trainer.device).flatten()).item()
        state=output


def run_cycle(config,shard,run_dir,target_tokens,device='cuda',precision='bf16',length=128,accumulation=8,lr=.0001,
              seed=42,validation_every=100,resume=False,max_steps=None,backend='triton',fp8_transfer=False,precision_transition=False,export_every=1000,pack_factor=1,execution_transition=False):
    if target_tokens<1 or length<1 or accumulation<1 or lr<=0:raise ValueError('invalid training schedule')
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    lock=(run_dir/'run.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('another process owns this training run')
    lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
    stop=[False]
    previous={sig:signal.signal(sig,lambda *_:stop.__setitem__(0,True)) for sig in (signal.SIGTERM,signal.SIGINT)}
    started=time.time();status_path=run_dir/'status.json';metrics_path=run_dir/'metrics.jsonl'
    def status(phase,**values):
        payload=dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),started_unix=started,**values)
        durable_json(status_path,payload);print(json.dumps(payload),flush=True)
    try:
        cfg=replace(MorphConfig.from_file(config),ssm_backend=backend)
        if cfg.proof_enabled:raise ValueError('Stage A is causal math-source LM; proof mode is a separate stage')
        dtype={'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}[precision]
        with torch.device('meta'):model=MorphModel(cfg)
        if accumulation%pack_factor:raise ValueError('packing factor must divide accumulation')
        budget=estimate_disk_budget(model,pack_factor,length,accumulation//pack_factor,dtype)
        if fp8_transfer:
            gate_path=Path('docs/fp8_precision_gate.json')
            if not gate_path.exists() or not json.loads(gate_path.read_text()).get('passed'):
                print('FP8 numerical gate absent/failed; using FP16 temporary transfer',flush=True);fp8_transfer=False
        if fp8_transfer:
            from .precision import fp8_eligible
            largest_fp8=max((p.numel() for name,p in model.named_parameters() if fp8_eligible(name)),default=0)
            budget['estimated_host_bytes']+=largest_fp8*16;budget['estimated_cuda_bytes']+=largest_fp8*12
        required_host=budget['estimated_host_bytes']+(budget['estimated_cuda_bytes'] if device=='cpu' else 0)
        if required_host>available_ram()*.8:raise MemoryError('stage/checkpoint host budget exceeds available RAM')
        if device.startswith('cuda'):
            if not torch.cuda.is_available():raise MemoryError('CUDA unavailable')
            if budget['estimated_cuda_bytes']>torch.cuda.mem_get_info(device)[0]*.85:raise MemoryError('stage GPU budget exceeds free VRAM')
        required_disk=budget['disk_reserve_bytes']+(0 if resume else budget['state_disk_bytes'])
        if required_disk>shutil.disk_usage(run_dir).free:raise OSError('insufficient disk for canonical state plus transaction reserve')
        manifest=json.loads(Path(shard,'manifest.json').read_text())
        settings=dict(config=cfg.to_dict(),data_fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                      target_tokens=target_tokens,sequence_length=length,accumulation=accumulation,batch_size=1,lr=lr,seed=seed,
                      precision=precision,fp8_transfer=fp8_transfer,export_every=export_every,pack_factor=pack_factor,forward_order='layer_major',device_type=torch.device(device).type,validation_every=validation_every)
        settings_path=run_dir/'settings.json'
        if resume:
            old=json.loads(settings_path.read_text())
            if old!=settings:
                checkpoint=json.loads((run_dir/'state'/'manifest.json').read_text())
                allowed={'precision','fp8_transfer','export_every','pack_factor','forward_order'} if precision_transition else {'pack_factor','forward_order'}
                comparable={k:v for k,v in settings.items() if k not in allowed}
                previous_settings={k:v for k,v in old.items() if k not in allowed}
                permitted=(precision_transition and checkpoint['step']==0) or execution_transition
                if not permitted or checkpoint['dirty'] or comparable!=previous_settings:
                    raise ValueError('resume settings/data/token budget differ')
                durable_json(run_dir/f'settings-before-step-{checkpoint["step"]:09d}.json',old);durable_json(settings_path,settings)
        else:
            if settings_path.exists():raise ValueError('run already exists; use resume')
            durable_json(settings_path,settings)
        status('validating_local_data',parameters=budget['parameters'],budget=budget)
        train=MappedShard(shard,'train');validation=MappedShard(shard,'validation')
        initialized=[0]
        def init_progress(name,total,size):
            initialized[0]+=size
            status('initializing',parameters=budget['parameters'],initialized_parameters=initialized[0],fraction=initialized[0]/total,tensor=name)
        store=DiskState.open(run_dir/'state',model) if resume else DiskState.create(run_dir/'state',model,seed=seed,progress=init_progress)
        trainer=DiskDLS(model,store,device,dtype,lr,async_mode=device.startswith('cuda'),fp8_transfer=fp8_transfer)
        last_layer_status=[0.,None]
        def layer_progress(phase,index,micro=None):
            # Durable layer status at most once per layer; microloops stay quiet.
            if micro not in (None,0):return
            now=time.monotonic()
            if phase==last_layer_status[1] and now-last_layer_status[0]<2.:return
            last_layer_status[:]=[now,phase]
            status('training_layer',training_phase=phase,layer=index,microbatch=micro,step=store.manifest['step'],processed_target_tokens=min(store.manifest['step']*length*accumulation,target_tokens),target_tokens=target_tokens)
        trainer.progress=layer_progress
        if store.manifest.get('transaction'):trainer.resume_step()
        stride=length*accumulation
        completed=min(store.manifest['step']*stride,target_tokens)
        initial_tokens=completed;initial_step=store.manifest['step'];training_start=time.perf_counter()
        if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
        status('running',parameters=budget['parameters'],step=store.manifest['step'],processed_target_tokens=completed,target_tokens=target_tokens)
        def export_clean():
            if not export_every:return
            from .export import export_fp16
            output=run_dir/f'fp16-step-{store.manifest["step"]:09d}'
            if (output/'manifest.json').exists():return
            if shutil.disk_usage(run_dir).free<budget['parameters']*2+budget['disk_reserve_bytes']:
                raise OSError('insufficient export space plus training recovery reserve')
            status('exporting_fp16',step=store.manifest['step'],processed_target_tokens=completed,output=str(output))
            export_fp16(store.root,output,lock_held=True)
            durable_json(run_dir/'fp16-latest.json',dict(path=str(output),step=store.manifest['step']))
            for old in run_dir.glob('fp16-step-*'):
                if old!=output and (old/'manifest.json').exists():shutil.rmtree(old)
        def io_snapshot():
            return {key:int(value) for key,value in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines()) if key in ('read_bytes','write_bytes')}
        while completed<target_tokens and not stop[0]:
            if max_steps is not None and store.manifest['step']-initial_step>=max_steps:break
            step=store.manifest['step'];remaining=min(stride,target_tokens-completed)
            generator=torch.Generator().manual_seed(seed+step);batches=[]
            while remaining:
                size=min(length,remaining);batches.append(train.sample(generator,1,size+1));remaining-=size
            from .precision import pack_lm_microbatches
            original_micros=len(batches);batches=pack_lm_microbatches(batches,pack_factor)
            step_tokens=sum(batch.shape[0]*(batch.shape[1]-1) for batch in batches)
            timing_before=dict(trainer.phase_seconds)
            io_before=io_snapshot();step_started=time.perf_counter();result=trainer.step(batches);io_after=io_snapshot()
            completed+=step_tokens
            if device.startswith('cuda'):torch.cuda.synchronize(device)
            timings=trainer.collect_compute_timing()
            elapsed=time.perf_counter()-training_start
            record=dict(execution_policy=dict(forward_order='layer_major',requested_pack_factor=pack_factor,effective_pack_factor=original_micros//len(batches),compute_batch_size=batches[0].shape[0],compute_microbatches=len(batches)),phase_seconds={key:value-timing_before.get(key,0.) for key,value in timings.items()},step=store.manifest['step'],processed_target_tokens=completed,step_target_tokens=step_tokens,
                        loss=result['loss'],grad_norm=result['grad_norm'],step_seconds=time.perf_counter()-step_started,
                        tokens_per_second=(completed-initial_tokens)/max(elapsed,1e-9),target_tokens=target_tokens,
                        process_io_bytes={key:io_after[key]-io_before[key] for key in io_before},logical_state_and_journal_bytes=24*budget['parameters'],transfer_bytes=dict(trainer.transfer_bytes),unix=time.time(),parameters=budget['parameters'],loss_scale=trainer.loss_scale)
            if validation_every and store.manifest['step']%validation_every==0:
                record['validation_loss']=validate(trainer,validation,length,seed+900_000)
            if device.startswith('cuda'):record['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(device)
            import resource
            record['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            with metrics_path.open('a') as output:output.write(json.dumps(record)+'\n');output.flush();os.fsync(output.fileno())
            status('running',**record)
            if export_every and store.manifest['step']%export_every==0:export_clean()
        if store.manifest['step']>initial_step:export_clean()
        status('completed' if completed>=target_tokens else 'stopped',step=store.manifest['step'],processed_target_tokens=completed,target_tokens=target_tokens,
               safe_to_resume=not store.manifest['dirty'])
        return completed
    except BaseException as error:
        status('failed',error=repr(error));raise
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--shard',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--target-tokens',type=int,required=True);p.add_argument('--device',default='cuda');p.add_argument('--precision',choices=['bf16','fp16','fp32'],default='bf16')
    p.add_argument('--sequence-length',type=int,default=128);p.add_argument('--accumulation',type=int,default=8);p.add_argument('--lr',type=float,default=.0001)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--validation-every',type=int,default=100);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--pack-factor',type=int,default=1);p.add_argument('--execution-transition',action='store_true')
    p.add_argument('--fp8-transfer',action='store_true');p.add_argument('--precision-transition',action='store_true');p.add_argument('--export-every',type=int,default=1000)
    p.add_argument('--resume',action='store_true');p.add_argument('--max-steps',type=int);p.add_argument('--backend',choices=['parallel','triton','reference'],default='triton')
    a=p.parse_args();torch.set_num_threads(a.threads)
    run_cycle(a.config,a.shard,a.run_dir,a.target_tokens,a.device,a.precision,a.sequence_length,a.accumulation,a.lr,a.seed,a.validation_every,a.resume,a.max_steps,a.backend,a.fp8_transfer,a.precision_transition,a.export_every,a.pack_factor,a.execution_transition)

if __name__=='__main__':main()
