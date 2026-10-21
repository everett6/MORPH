"""Small RAM-streamed training cycle including fetch and durable source commits."""
import argparse,contextlib,json,time,tempfile,resource
from dataclasses import replace
from pathlib import Path
import torch
from morph.delta_config import DeltaConfig
from streamtrain.delta_cycle import run_delta_cycle

from benchmarks.audit_qualification import guarded_count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/morph_delta_tiny_local2.json')
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--output',default='docs/audit_streaming_cycle.json')
    args=parser.parse_args()
    if args.steps<1:raise ValueError('positive step count required')
    Path('.generated').mkdir(exist_ok=True)
    torch.set_num_threads(2)
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    report=dict(parameters=None,scope='Small real HF streamed cycle; includes source resolution/fetch, initialization, durable source commits and three optimizer updates; excludes Python imports and CUDA context warmup. No full-size throughput projection.',target_byte_count=32768*args.steps,byte_targets_per_update=32768)
    with tempfile.TemporaryDirectory(prefix='audit-stream-cycle-',dir='.generated') as temporary:
        root=Path(temporary)
        cfg=replace(DeltaConfig.from_file(args.config),proof_enabled=False,step_expert_count=0,expert_group_size=2,memory_projection_backend='batched')
        report['parameters']=guarded_count(cfg)
        report['config']=cfg.to_dict();report['seed']=4201
        config=root/'config.json';config.write_text(json.dumps(cfg.to_dict()))
        spec=json.loads(Path('configs/hf_math_stream.json').read_text());spec.update(retries=2,timeout=15.)
        report['dataset_revision']=spec['revision'];report['dataset_repo']=spec['repo']
        source_config=root/'source.json';source_config.write_text(json.dumps(spec))
        start=time.perf_counter()
        try:
            with Path('.generated/audit_streaming_cycle.log').open('w') as log,contextlib.redirect_stdout(log):
                result=run_delta_cycle(config,root/'run',target_tokens=32768*args.steps,source_config=source_config,
                                       device=args.device,precision='fp16' if args.device=='cuda' else 'fp32',length=128,batch_size=4,accumulation=64,
                                       lr=.001,max_steps=args.steps,backend='parallel',export=False,seed=4201,
                                       engine='single-pass',cache_bytes=1048576,chunk_microbatches=2,journal_format='packed')
            elapsed=time.perf_counter()-start
            records=[json.loads(line) for line in (root/'run'/'metrics.jsonl').read_text().splitlines()]
            report.update(completed=True,result=result,total_wall_seconds=elapsed,
                          end_to_end_byte_targets_per_second=result['processed_target_tokens']/elapsed,
                          optimizer_step_byte_targets_per_second=sum(r['step_target_tokens'] for r in records)/sum(r['step_seconds'] for r in records),
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else 0,
                          peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if args.device=='cuda' else 0,
                          process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,steps=records)
            report['end_to_end_threshold_2000_status']='MET' if report['end_to_end_byte_targets_per_second']>2000 else 'NOT MET'
        except Exception as error:
            report.update(completed=False,error_type=type(error).__name__,error=str(error),total_wall_seconds=time.perf_counter()-start)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='steps'}))

    return 0 if report['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
