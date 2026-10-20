"""Measured small K8 forward scheduling comparison; same weights/windows."""
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
import types
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.func import functional_call
from morph import MorphConfig,MorphModel
from streamtrain.disk import DiskDLS,DiskState
from streamtrain.precision import pack_lm_microbatches


def old_boundaries(self,batches):
    checkpoints=[]
    for batch in batches:
        state=(batch[:,:-1].to(self.device),None);saved=[];self.load(0,self.slots[0])
        for index in range(len(self.stages)-1):
            saved.append(tuple(v.detach().cpu() if v is not None else None for v in state));slot=self.slots[index%2]
            if index+1<len(self.stages)-1:self.load(index+1,self.slots[(index+1)%2])
            params=self.parameters(index,slot,False)
            with torch.no_grad(),torch.autocast(self.device.type,dtype=self.dtype):state=functional_call(self.stages[index],params,state)
            self.release(slot)
        saved.append(tuple(v.detach().cpu() if v is not None else None for v in state));checkpoints.append(saved)
    return checkpoints


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    cfg=MorphConfig(vocab_size=256,d_model=128,num_layers=8,d_ff=256,d_state=8,head_dim=32,graph_enabled=True,d_graph=32,graph_nodes=4)
    source=MorphModel(cfg);batches=[torch.randint(256,(1,129)) for _ in range(8)];results={}
    with tempfile.TemporaryDirectory(prefix='morph-forward-') as folder:
        store=DiskState.create(Path(folder)/'state',source,source=source)
        for label,factor,legacy in [('microbatch_major',1,True),('layer_major',1,False),('layer_major_b2',2,False)]:
            executor=DiskDLS(source,store,'cuda',torch.float16,async_mode=True,fp8_transfer=True)
            if legacy:executor.boundaries=types.MethodType(old_boundaries,executor)
            batch=pack_lm_microbatches(batches,factor)
            executor.boundaries(batch);executor.collect_compute_timing();executor.phase_seconds={};executor.transfer_bytes=dict(h2d_weights=0,d2h_gradients=0)
            timings=[]
            for _ in range(3):
                started=time.perf_counter();executor.boundaries(batch);torch.cuda.synchronize();timings.append(time.perf_counter()-started)
            phases=executor.collect_compute_timing()
            results[label]=dict(median_forward_seconds=statistics.median(timings),trials=timings,transfer_bytes=executor.transfer_bytes,phase_seconds=phases)
            del executor
    report=dict(config=cfg.to_dict(),parameters=sum(p.numel() for p in source.parameters()),device=torch.cuda.get_device_name(),precision='FP16 with FFN FP8 temporary transfer',seed=42,threads=2,sequence_length=128,effective_batch=8,warmup_forwards=1,measured_forwards=3,results=results,scope='forward only; small model, shared loaded host/GPU; not 5B throughput')
    Path('docs/layer_major_forward_profile.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
