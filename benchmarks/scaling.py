"""Resource-qualified size/scan/graph/engine ablation benchmark; no downloads."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from morph import MorphConfig,MorphModel
from morph.training.runner import train_run
from streamtrain.disk import disk_train,estimate_disk_budget


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/morph_100m.json')
    parser.add_argument('--component',choices=['model','ssm','graph'],default='model')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--engine',choices=['resident','async','disk'],default='resident')
    parser.add_argument('--backend',choices=['reference','parallel','triton'],default='parallel')
    parser.add_argument('--ablation',choices=['ssm','static_graph','graph_recurrence','full_proof'],default='ssm')
    parser.add_argument('--steps',type=int,default=1)
    parser.add_argument('--sequence-length',type=int,default=8)
    parser.add_argument('--output')
    args=parser.parse_args();torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg=MorphConfig.from_file(args.config)
    cfg=replace(cfg,ssm_backend=args.backend,graph_enabled=args.ablation in ('static_graph','graph_recurrence'),
                graph_update=args.ablation!='static_graph')
    with torch.device('meta'):probe=MorphModel(cfg)
    count=sum(p.numel() for p in probe.parameters())
    cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown')
    result=dict(metadata=dict(steps=args.steps,sequence_length=args.sequence_length,batch_size=1,accumulation=1,precision='fp32',seed=42,threads=2,torch_version=str(torch.__version__),cpu_model=cpu),parameters=count,config=cfg.to_dict(),device=args.device,engine=args.engine,ablation=args.ablation,
                measured=False,verified_proof_success=None)
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    budget=estimate_disk_budget(probe,1,args.sequence_length,1,torch.float32)
    required_host=count*20+512*2**20 if args.engine=='resident' else budget['estimated_host_bytes']+(count*16 if args.engine=='async' else 0)
    required_cuda=count*20+512*2**20 if args.engine=='resident' else budget['estimated_cuda_bytes']
    if args.component!='model':
        from morph.model.ssm import SelectiveSSM
        from morph.model.graph import GraphExchange
        with torch.device('meta'):component=SelectiveSSM(cfg) if args.component=='ssm' else GraphExchange(cfg)
        parameters=sum(p.numel() for p in component.parameters())
        required_host=512*2**20
        required_cuda=parameters*16+budget['estimated_cuda_bytes']-2*budget['largest_stage_parameters']*12
    result['qualification']=dict(estimated_host_bytes=required_host,estimated_cuda_bytes=required_cuda)
    if required_host>available*.75:result['skip']='insufficient available RAM for selected residency mode'
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():result['skip']='CUDA unavailable'
        else:
            free,_=torch.cuda.mem_get_info(args.device)
            if required_cuda>free*.8:result['skip']='insufficient free VRAM for selected stage/component budget'
    if args.ablation=='full_proof':result['skip']='use structured proof training/evaluation; LM-only timing is not a proof ablation'
    if args.engine=='disk':
        import shutil
        budget=estimate_disk_budget(probe,1,args.sequence_length,1,torch.float32)
        if budget['state_disk_bytes']+budget['disk_reserve_bytes']>shutil.disk_usage(tempfile.gettempdir()).free:
            result['skip']='insufficient free disk including transaction journal'
    if 'skip' not in result and args.component!='model':
        from morph.model.ssm import SelectiveSSM
        from morph.model.graph import GraphExchange
        module=(SelectiveSSM(cfg) if args.component=='ssm' else GraphExchange(cfg)).to(args.device)
        sample=torch.randn(1,args.sequence_length,cfg.d_model,device=args.device,requires_grad=True)
        started=time.perf_counter()
        for _ in range(args.steps):
            module.zero_grad(set_to_none=True)
            output=module(sample)[0]
            output.square().mean().backward()
        if args.device.startswith('cuda'):torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        result.update(measured=True,component=args.component,component_parameters=sum(p.numel() for p in module.parameters()),
                      metrics=dict(elapsed_seconds=elapsed,tokens_per_second=args.steps*args.sequence_length/elapsed))
    elif 'skip' not in result:
        tokens=torch.arange(max(128,args.sequence_length+1)).remainder(256)[None]
        if args.engine=='disk':
            with tempfile.TemporaryDirectory(prefix='morph-scaling-') as folder:
                metrics=disk_train(cfg,tokens,Path(folder)/'state',args.steps,device=args.device,sequence_length=args.sequence_length)
        else:
            run=train_run(cfg,tokens,args.steps,device=args.device,sequence_length=args.sequence_length,engine=args.engine,optimizer_name='adamw')
            metrics=run['metrics']
            import resource
            metrics['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        result.update(measured=True,metrics=metrics)
    output=json.dumps(result,indent=2)
    if args.output:Path(args.output).write_text(output+'\n')
    print(output)


if __name__=='__main__':main()
