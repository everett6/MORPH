"""Multi-step FP16 AdamW accuracy gate for optional execution changes.

Compares identical parameters, byte windows and all-expert rosters. This is a
small numerical gate, independent of optimizer-step throughput qualification.
"""
import argparse
from dataclasses import replace
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import torch
from benchmarks.audit_qualification import collect_data,guarded_count
from benchmarks.delta_precision import acceptance
from morph.delta_config import DeltaConfig
from morph.model.delta import DeltaModel
from streamtrain.disk import DiskState
from streamtrain.packed_state import PackedDiskState
from streamtrain.sparse_disk import SparseDiskDLS
from streamtrain.single_pass import SinglePassDiskDLS


def relative_error(reference,approximate):
    numerator=sum((approximate[k]-v).double().square().sum().item() for k,v in reference.items())
    denominator=sum(v.double().square().sum().item() for v in reference.values())
    return math.sqrt(numerator/max(denominator,1e-30))


def compare_states(initial,reference,approximate):
    reference_updates={n:v[0]-initial[n] for n,v in reference.items()}
    approximate_updates={n:v[0]-initial[n] for n,v in approximate.items()}
    stats=dict(update_relative_l2=relative_error(reference_updates,approximate_updates),
        reference_update_l2=math.sqrt(sum(v.double().square().sum().item() for v in reference_updates.values())),
        max_absolute_weight_difference=max((reference[n][0]-approximate[n][0]).abs().max().item() for n in reference))
    for row,label in ((1,'first_moment'),(2,'second_moment')):
        expected={n:v[row] for n,v in reference.items()};actual={n:v[row] for n,v in approximate.items()}
        stats[label+'_relative_l2']=relative_error(expected,actual)
        stats[label+'_max_absolute_difference']=max((expected[n]-actual[n]).abs().max().item() for n in expected)
    stats['finite']=all(math.isfinite(v) for v in stats.values())
    return stats


def snapshot(store):
    return {name:torch.from_numpy(np.memmap(store.root/entry['file'],dtype='float32',mode='r',shape=(3,entry['size'])).copy()) for name,entry in store.entries.items()}


def execute(cfg,source,train,device,optimized):
    with torch.device('meta'):model=DeltaModel(cfg)
    with tempfile.TemporaryDirectory(prefix='.audit-precision-',dir=Path.cwd()) as directory:
        store=(PackedDiskState if optimized else DiskState).create(Path(directory)/'state',model,source=source)
        trainer=(SinglePassDiskDLS if optimized else SparseDiskDLS)(model,store,device=device,
            dtype=torch.float16 if device=='cuda' else torch.float32,lr=.001,async_mode=device=='cuda',
            cpu_prefetch=device=='cuda',**(dict(chunk_microbatches=2,stash_budget_bytes=32*2**20) if optimized else {}))
        results=[];snapshots=[];clocks=[]
        try:
            for step,micros in enumerate(train):
                model.set_routing_step(step);results.append(trainer.step(micros))
                snapshots.append(snapshot(store));clocks.append(dict(store.manifest.get('parameter_steps',{})))
            return results,snapshots,clocks
        finally:trainer.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--output',default='docs/audit_precision.json');args=parser.parse_args()
    started=datetime.now(timezone.utc).isoformat();torch.set_num_threads(2);torch.manual_seed(4201)
    base=replace(DeltaConfig.from_file('configs/morph_delta_tiny_local2.json'),proof_enabled=False,ssm_backend='parallel',step_expert_count=0)
    count=guarded_count(base);source=DeltaModel(base);initial={n:p.detach().flatten().clone() for n,p in source.named_parameters()}
    spec=json.loads(Path('configs/hf_math_stream.json').read_text());spec.update(retries=2,timeout=15.)
    try:train,heldout,data=collect_data(spec,3,4,128,4)
    except Exception as error:
        generator=torch.Generator().manual_seed(4202)
        train=[[torch.randint(0,256,(4,129),generator=generator) for _ in range(4)] for _ in range(3)]
        chain=[];cause=error
        while cause is not None:chain.append(f'{type(cause).__name__}: {cause}');cause=cause.__cause__
        data=dict(kind='synthetic fallback',real_qualification=False,error_chain=chain)
    reference,reference_states,reference_clocks=execute(base,source,train,args.device,False)
    report=dict(parameters=count,seed=4201,source=data,device=args.device,torch_version=torch.__version__,
        train_payload_sha256=hashlib.sha256(b''.join(b.numpy().tobytes() for micros in train for b in micros)).hexdigest(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        tracked_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','HEAD'])).hexdigest(),
        source_file_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ('streamtrain','morph/model') for p in Path(directory).glob('*.py')},
        learning_rate=.001,max_norm=1.,steps=3,batch_size=4,length=128,accumulation=4,chunk_microbatches=2,
        roster='All four experts eligible, identical routing step, seed, inputs and initial FP32 canonical tensors.',
        update_error_definition='sqrt(sum((W_reference-W_optimized)^2)/sum((W_reference-W_initial)^2)), over all canonical parameters after each committed step.',
        threshold=dict(relative_loss_drift=.02,relative_global_norm_drift=.10,update_relative_l2=.10),reference=reference,variants=[],
        scope='Three-step small-model numerical accuracy gate only; no convergence, full5B, or throughput inference.')
    for group in (1,2):
        cfg=replace(base,expert_group_size=group,memory_projection_backend='batched')
        results,states,clocks=execute(cfg,source,train,args.device,True);steps=[]
        for index,(ref,actual) in enumerate(zip(reference,results)):
            stats=compare_states(initial,reference_states[index],states[index]);gate=acceptance(ref,actual,stats['update_relative_l2'])
            clock_match=clocks[index]==reference_clocks[index]
            gate['passed'] &= stats['finite'] and clock_match and stats['reference_update_l2']>0
            steps.append(dict(step=index+1,reference=ref,optimized=actual,statistics=stats,parameter_clocks_equal=clock_match,gate=gate))
        report['variants'].append(dict(expert_group_size=group,memory_projection_backend='batched',steps=steps,passed=all(s['gate']['passed'] for s in steps)))
    report['passed']=all(v['passed'] for v in report['variants']);report['run_started_utc']=started;report['run_completed_utc']=datetime.now(timezone.utc).isoformat()
    path=Path(args.output);path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# Optional execution precision gate','',report['scope'],'',f"Data: {data['kind']}. FP16 CUDA compute with FP32 canonical weights and Adam moments." if args.device=='cuda' else f"Data: {data['kind']}. FP32 CPU compute.",'',report['update_error_definition'],'','| Group size | Step | Relative update error | First moment error | Second moment error | Gate |','|---|---:|---:|---:|---:|---|']
    for variant in report['variants']:
        for step in variant['steps']:
            stats=step['statistics'];lines.append(f"| {variant['expert_group_size']} | {step['step']} | {stats['update_relative_l2']:.6f} | {stats['first_moment_relative_l2']:.6f} | {stats['second_moment_relative_l2']:.6f} | {'PASS' if step['gate']['passed'] else 'FAIL'} |")
    lines+=['','Limits: update error ≤10%, loss drift ≤2%, global gradient norm drift ≤10%; finite state and matching per-parameter clocks are required. Moment errors are reported separately.','',f"Overall gate: {'PASS' if report['passed'] else 'FAIL'}. Failure keeps the affected optimization optional and calls for the reference fallback."]
    path.with_suffix('.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(output=str(path),passed=report['passed'],source=data,variants=[dict(group=v['expert_group_size'],passed=v['passed'],errors=[s['statistics']['update_relative_l2'] for s in v['steps']]) for v in report['variants']])))

if __name__=='__main__':main()
