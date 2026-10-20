"""Small CPU-only comparison; never opens production training state."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import tempfile
import time
import numpy as np
import torch
from morph.delta_config import DeltaConfig
from morph.model.delta import DeltaModel
from streamtrain.disk import DiskState
from streamtrain.sparse_disk import SparseDiskDLS
from streamtrain.single_pass import SinglePassDiskDLS
from streamtrain.state_cache import CachedDiskState


def io_counters():
    return {key:int(value) for key,value in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}


def default_config():
    return DeltaConfig(d_model=32,num_layers=1,d_ff=64,d_state=4,head_dim=16,d_graph=16,graph_nodes=4,
        proof_enabled=False,fast_memory_enabled=False,expert_count=64,expert_depth=2,expert_ffn=128,latent_dim=16,memory_dim=8)


def run_benchmark(config=None,steps=3,length=16,batch_size=2,accumulation=2,quota=8,seed=42,threads=2,cache_bytes=32*2**20):
    if min(steps,length,batch_size,accumulation,threads)<1:raise ValueError('positive schedule required')
    torch.set_num_threads(threads);torch.manual_seed(seed)
    cfg=replace(config or default_config(),ssm_backend='parallel',step_expert_count=0,routing_seed=seed)
    source=DeltaModel(cfg);count=sum(p.numel() for p in source.parameters())
    # One sequential store, transient journal/stash and model snapshots; refuse large experiments.
    if count*max(48,128*steps)>400*2**20:raise MemoryError('benchmark limited to <400MiB estimated cumulative payload I/O; use a smaller configuration')
    generator=torch.Generator().manual_seed(seed+1)
    batches=[[torch.randint(0,256,(batch_size,length+1),generator=generator) for _ in range(accumulation)] for _ in range(steps)]
    variants=[('unrestricted_two_pass',0,False,False),('quota_two_pass',quota,False,False),
              ('quota_single_pass',quota,True,False),('quota_single_pass_cache',quota,True,True)]
    reports=[];reference=None;comparisons={}
    for label,eligible,single,cached in variants:
        variant=replace(cfg,step_expert_count=eligible)
        with tempfile.TemporaryDirectory(prefix='.morph-step-local-',dir=Path.cwd()) as temporary:
            model=DeltaModel(variant)
            store_type=CachedDiskState if cached else DiskState
            kwargs=dict(cache_bytes=cache_bytes) if cached else {}
            store=store_type.create(Path(temporary)/'state',model,source=source,**kwargs)
            logical=dict(weight_reads=0,canonical_state_writes=0,journal_payload_writes=0)
            read,update,prepare=store.read_into,store.update,store.prepare_stage
            def counted_read(name,destination):
                logical['weight_reads']+=store.entries[name]['size']*4;return read(name,destination)
            def counted_update(name,*args,**kw):
                logical['canonical_state_writes']+=store.entries[name]['size']*12;return update(name,*args,**kw)
            def counted_prepare(index,names):
                logical['journal_payload_writes']+=sum(store.entries[name]['size']*12 for name in names)
                return prepare(index,names)
            store.read_into=counted_read;store.update=counted_update;store.prepare_stage=counted_prepare
            engine=(SinglePassDiskDLS if single else SparseDiskDLS)(model,store,device='cpu',lr=1e-3)
            records=[]
            try:
                for step,microbatches in enumerate(batches):
                    model.set_routing_step(store.manifest['step'])
                    before=io_counters();logical_before=logical.copy();started=time.perf_counter()
                    result=engine.step(microbatches);elapsed=time.perf_counter()-started;after=io_counters()
                    active=sum(engine.active_stages[i] for i,stage in enumerate(engine.stages) if hasattr(stage,'expert_index'))
                    targets=batch_size*length*accumulation
                    records.append(dict(step=step+1,seconds=elapsed,targets=targets,targets_per_second=targets/elapsed,
                        active_expert_union=active,loss=result['loss'],grad_norm=result['grad_norm'],
                        process_io_delta={key:after[key]-before[key] for key in before},
                        logical_payload_bytes={key:logical[key]-logical_before[key] for key in logical},
                        gradient_stash_payload_bytes=getattr(engine,'last_stash_bytes',0)))
                snapshot={name:np.memmap(store.root/entry['file'],dtype=np.float32,mode='r',shape=(3,entry['size'])).copy() for name,entry in store.entries.items()}
                clocks=store.manifest.get('parameter_steps',{}).copy()
                if eligible:
                    if reference is None:reference=(snapshot,clocks)
                    else:
                        maxima=[max(float(np.max(np.abs(snapshot[name][row]-reference[0][name][row]))) for name in snapshot) for row in range(3)]
                        match=clocks==reference[1]
                        comparisons[label]=dict(max_abs_weight=maxima[0],max_abs_m=maxima[1],max_abs_v=maxima[2],clocks_equal=match)
                        if not match or max(maxima)>1e-6:raise AssertionError(f'optimizer parity failed: {comparisons[label]}')
                reports.append(dict(variant=label,steps=records,total_targets=sum(r['targets'] for r in records),
                    total_seconds=sum(r['seconds'] for r in records),cache_stats=store.cache_stats if cached else None))
            finally:engine.close()
    return dict(parameters=count,config=cfg.to_dict(),quota=quota,steps=steps,sequence_length=length,batch_size=batch_size,
        accumulation=accumulation,seed=seed,threads=threads,device='cpu',precision='float32',torch_version=torch.__version__,
        temporary_store_parent=str(Path.cwd()),cpu=platform.processor() or next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown'),
        comparisons=comparisons,variants=reports,
        limitations=['CPU small-model experiment; production GPU training was concurrent.',
          'OS page cache is warm and uncontrolled; /proc/self/io physical counters are process-accounted, not device-wide traffic.',
          'Logical payload counters exclude serialization metadata, cursor files and gradient-stash framing; cache read avoidance is estimated.',
          'Quota changes the computation; unrestricted versus quota is not a numerical-parity comparison.',
          'No full5B speed or convergence extrapolation.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config');parser.add_argument('--steps',type=int,default=3);parser.add_argument('--length',type=int,default=16)
    parser.add_argument('--batch-size',type=int,default=2);parser.add_argument('--accumulation',type=int,default=2)
    parser.add_argument('--quota',type=int,default=8);parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--output',default='docs/step_local_benchmark.json');args=parser.parse_args()
    report=run_benchmark(DeltaConfig.from_file(args.config) if args.config else None,args.steps,args.length,args.batch_size,args.accumulation,args.quota,threads=args.threads)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(parameters=report['parameters'],comparisons=report['comparisons'],output=args.output)))

if __name__=='__main__':main()
