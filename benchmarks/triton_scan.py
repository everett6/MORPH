"""Small long-sequence fused-vs-PyTorch scan forward/backward measurement."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from morph.model.ssm import chunked_scan
from morph.model.triton_scan import triton_scan


def main():
    torch.set_num_threads(1);torch.manual_seed(42)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    args=[torch.randn(1,1024,4,8,device='cuda'),torch.rand(1,1024,4,device='cuda')*.1+.1,-torch.ones(4,device='cuda'),
          torch.randn(1,1024,4,16,device='cuda'),torch.randn(1,1024,4,16,device='cuda'),torch.zeros(1,4,8,16,device='cuda')]
    args=[value.requires_grad_() for value in args]
    results={}
    for name,scan in [('parallel',chunked_scan),('triton',triton_scan)]:
        def run():
            output=scan(*args,chunk_size=16)
            return torch.autograd.grad(sum(value.square().mean() for value in output),args)
        run();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        for _ in range(5):run()
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        results[name]=dict(seconds=elapsed,iterations=5,peak_cuda_bytes=torch.cuda.max_memory_allocated(),tokens_per_second=5*1024/elapsed)
    report=dict(shape=dict(batch=1,length=1024,heads=4,channels=8,state=16),precision='fp32',torch_version=str(torch.__version__),results=results,
                claim='single primitive microbenchmark; not model throughput or deterministic gradients')
    Path('docs/triton_scan_gpu.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
