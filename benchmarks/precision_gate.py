"""Small complete-model FP8 temporary-transfer acceptance gate, not quality proof."""
import copy
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from morph import MorphConfig,MorphModel
from streamtrain.disk import DiskState,DiskDLS
from streamtrain.precision import fp8_eligible,pack_fp8,unpack_fp8


def run(output):
    torch.set_num_threads(2);torch.manual_seed(42)
    cfg=MorphConfig(vocab_size=256,d_model=64,num_layers=4,d_ff=128,d_state=8,head_dim=16,graph_enabled=True,d_graph=16,graph_nodes=4)
    source=MorphModel(cfg);tokens=torch.randint(256,(1,33));reference=copy.deepcopy(source).cuda().half();quantized=copy.deepcopy(reference)
    for name,p in quantized.named_parameters():
        if fp8_eligible(name):
            encoded,scales=pack_fp8(p.detach().cpu());p.data.copy_(unpack_fp8(encoded,scales).reshape_as(p).cuda())
    def gradients(model):
        with torch.autocast('cuda',dtype=torch.float16):loss=model.loss(tokens.cuda())
        (loss*128).backward()
        return loss.item(),{name:p.grad.float().cpu()/128 for name,p in model.named_parameters()}
    baseline,basegrad=gradients(reference);approx,quantgrad=gradients(quantized)
    with tempfile.TemporaryDirectory(prefix='morph-precision-') as folder:
        store=DiskState.create(Path(folder)/'state',source,source=source)
        executor=DiskDLS(source,store,'cuda',torch.float16,.0001,True,fp8_transfer=True)
        observed={};original=store.update
        def update(name,gradient,*args):observed[name]=gradient.clone();return original(name,gradient,*args)
        store.update=update
        result=executor.step([tokens])
        transfer=executor.transfer_bytes
    def relative(a,b):return (sum((a[k]-b[k]).double().square().sum() for k in a)/sum(a[k].double().square().sum() for k in a)).sqrt().item()
    approximation=relative(basegrad,quantgrad);execution_error=relative(quantgrad,observed)
    report=dict(config=cfg.to_dict(),device=torch.cuda.get_device_name(),seed=42,tokens=32,precision='FP16 compute/FP32 canonical AdamW',
                fp8='E4M3FN/256-element blocks/FP32 scales; FFN gate/up/down only',loss_fp16=baseline,loss_fp8=approx,
                relative_loss_drift=abs(approx-baseline)/baseline,relative_gradient_drift=approximation,
                quantized_resident_vs_disk_gradient_relative_l2=execution_error,disk_loss=result['loss'],transfer_bytes=transfer,
                limits=dict(relative_loss=.02,relative_gradient=.10,execution_gradient=.02),
                passed=abs(approx-baseline)/baseline<.02 and approximation<.10 and execution_error<.02,
                claims='small numerical acceptance gate; no convergence or throughput guarantee')
    Path(output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return report

if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else 'docs/fp8_precision_gate.json')
