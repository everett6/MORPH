import copy
from dataclasses import replace
import pytest
import torch
from morph.delta_config import DeltaConfig
from morph.model.fast_memory import GraphMemory,FastMemoryState
from morph.model.delta import DeltaModel,ExpertStage,GroupedExpertStage


def tiny(**kwargs):
    return DeltaConfig(d_model=8,num_layers=1,d_ff=12,d_state=2,head_dim=4,
                       expert_count=4,expert_depth=2,expert_ffn=12,latent_dim=4,
                       proof_enabled=False,memory_dim=3,**kwargs)


def gradients_match(a,b,atol=2e-5):
    for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
        assert name==other
        assert (p.grad is None)==(q.grad is None),name
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=atol,rtol=2e-4)


def test_memory_batched_outputs_state_and_all_gradients():
    torch.manual_seed(7)
    a=GraphMemory(8,3);b=copy.deepcopy(a);b.projection_backend='batched'
    x=torch.randn(2,5,8,requires_grad=True);y=x.detach().clone().requires_grad_()
    c=torch.randn(2,8,requires_grad=True);d=c.detach().clone().requires_grad_()
    m=torch.randn(2,3,3,requires_grad=True);n=m.detach().clone().requires_grad_()
    out,s=a(x,FastMemoryState(m,9),c);got,t=b(y,FastMemoryState(n,9),d)
    torch.testing.assert_close(out,got);torch.testing.assert_close(s.matrix,t.matrix)
    assert s.version==t.version==14
    (out.square().sum()+s.matrix.square().sum()).backward();(got.square().sum()+t.matrix.square().sum()).backward()
    gradients_match(a,b)
    for p,q in ((x,y),(c,d),(m,n)):torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=2e-4)
    # Prefix output and state do not depend on future tokens.
    prefix,state=b(y[:,:3],FastMemoryState(n,9),d)
    torch.testing.assert_close(prefix,got[:,:3])
    assert state.matrix.dtype==torch.float32


def test_grouped_full_model_and_dormant_gradients():
    torch.manual_seed(2)
    a=DeltaModel(tiny());b=DeltaModel(replace(a.config,expert_group_size=3,memory_projection_backend='batched'))
    b.load_state_dict(a.state_dict())
    # Two active experts; two dormant experts must keep None gradients.
    a.router.eligible=b.router.eligible=(0,2)
    tokens=torch.randint(0,256,(2,6))
    out=a(tokens);got=b(tokens)
    torch.testing.assert_close(out.logits,got.logits,atol=2e-6,rtol=2e-5)
    assert torch.equal(out.routes,got.routes)
    (out.logits.square().sum()+out.router_aux_loss).backward();(got.logits.square().sum()+got.router_aux_loss).backward()
    gradients_match(a,b)
    for index in (1,3):assert all(p.grad is None for p in b.experts[index].parameters())
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    prefix=b(tokens[:,:3]);torch.testing.assert_close(prefix.logits,got.logits[:,:3],atol=2e-6,rtol=2e-5)


def test_group_stage_inputs_and_selection():
    torch.manual_seed(4);model=DeltaModel(tiny())
    experts=copy.deepcopy(model.experts[:3]);group=GroupedExpertStage(experts,range(3))
    x=torch.randn(2,5,8,requires_grad=True);y=x.detach().clone().requires_grad_()
    gates=torch.zeros(2,5,4);gates[...,0]=.6;gates[:,::2,2]=.9
    gates.requires_grad_();other=gates.detach().clone().requires_grad_()
    out=x
    for index,expert in enumerate(model.experts[:3]):out,_=ExpertStage(expert,index)(out,gates)
    got,_=group(y,other);torch.testing.assert_close(out,got)
    out.square().sum().backward();got.square().sum().backward()
    torch.testing.assert_close(x.grad,y.grad);torch.testing.assert_close(gates.grad,other.grad)
    gradients_match(torch.nn.ModuleList(list(model.experts[:3])),experts)
    names=group.parameter_names_for([(y,other)])
    assert names and all(not name.startswith('experts.1.') for name in names)


def test_execution_config_validation():
    assert DeltaConfig().memory_projection_backend=='reference' and DeltaConfig().expert_group_size==1
    with pytest.raises(ValueError):tiny(memory_projection_backend='bad')
    with pytest.raises(ValueError):tiny(expert_group_size=5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required for FP16 streaming slot parity')
def test_cuda_fp16_functional_storage():
    from torch.func import functional_call
    torch.manual_seed(9)
    reference=GraphMemory(8,3).to('meta');batched=GraphMemory(8,3,projection_backend='batched').to('meta')
    source=GraphMemory(8,3)
    p={name:value.detach().cuda().half().requires_grad_() for name,value in source.named_parameters()}
    q={name:value.detach().clone().requires_grad_() for name,value in p.items()}
    x=torch.randn(2,7,8,device='cuda',dtype=torch.float16,requires_grad=True);y=x.detach().clone().requires_grad_()
    c=torch.randn(2,8,device='cuda',dtype=torch.float16,requires_grad=True);d=c.detach().clone().requires_grad_()
    m=torch.randn(2,3,3,device='cuda',requires_grad=True);n=m.detach().clone().requires_grad_()
    with torch.autocast('cuda',dtype=torch.float16):
        out,s=functional_call(reference,p,(x,FastMemoryState(m),c))
        got,t=functional_call(batched,q,(y,FastMemoryState(n),d))
    torch.testing.assert_close(out,got,atol=.003,rtol=.003);torch.testing.assert_close(s.matrix,t.matrix,atol=.003,rtol=.003)
    (out.float().square().mean()+s.matrix.square().mean()).backward();(got.float().square().mean()+t.matrix.square().mean()).backward()
    for name in p:torch.testing.assert_close(p[name].grad,q[name].grad,atol=.004,rtol=.02)
    for left,right in ((x,y),(c,d),(m,n)):torch.testing.assert_close(left.grad,right.grad,atol=.004,rtol=.02)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cuda_fp16_grouped_full_model():
    torch.manual_seed(12)
    a=DeltaModel(tiny()).cuda().half()
    b=DeltaModel(replace(a.config,expert_group_size=3,memory_projection_backend='batched')).cuda().half()
    b.load_state_dict(a.state_dict());a.router.eligible=b.router.eligible=(0,2)
    tokens=torch.randint(0,256,(2,6),device='cuda')
    with torch.autocast('cuda',dtype=torch.float16):
        out=a(tokens);got=b(tokens)
        left=out.logits.float().square().mean()+out.router_aux_loss
        right=got.logits.float().square().mean()+got.router_aux_loss
    torch.testing.assert_close(out.logits,got.logits,atol=.002,rtol=.01)
    left.backward();right.backward();gradients_match(a,b,atol=.003)
