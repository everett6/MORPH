import copy
import pytest
import torch
from morph import MorphConfig,MorphModel
from morph.graph import ProofGraph,ProofNode,Hyperedge,Relation
from morph.graph.tensors import tensorize
from morph.training.proof import joint_loss
from streamtrain import StreamTrainEngine
from streamtrain.disk import DiskState,DiskDLS


def configuration():
    return MorphConfig(vocab_size=256,d_model=8,num_layers=2,d_ff=16,d_state=2,head_dim=4,
                       d_graph=4,graph_nodes=3,graph_interval=1,graph_enabled=True,proof_enabled=True)


def batch():
    graph=ProofGraph();graph.add(ProofNode(0,'P -> Q',formal_statement='True',assumptions=('P',)))
    graph.add(ProofNode(1,'Q -> P',assumptions=('Q',)));graph.connect(Hyperedge((1,),0,Relation.REQUIRES))
    tensors,_=tensorize([graph],3)
    return dict(tokens=torch.tensor([[10,20,30,40]]),graph=tensors,actions=torch.tensor([2]),targets=torch.tensor([0]),values=torch.tensor([.3]))


def loss(model,data,device):
    graph={key:value.to(device) for key,value in data['graph'].items()}
    tokens=data['tokens'].to(device)
    return joint_loss(model.forward_proof(tokens,graph),tokens,graph,data['actions'].to(device),data['targets'].to(device),data['values'].to(device))[0]


@pytest.mark.parametrize('device,asynchronous',[('cpu',False),('cuda',False),('cuda',True)])
def test_all_joint_gradients_and_adam_updates_streamed(device,asynchronous):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(3);model=MorphModel(configuration());resident=copy.deepcopy(model).to(device)
    engine=StreamTrainEngine(model,device,asynchronous)
    a=torch.optim.AdamW(model.parameters(),lr=.001);b=torch.optim.AdamW(resident.parameters(),lr=.001)
    data=batch()
    for _ in range(2):
        a.zero_grad();b.zero_grad()
        expected=loss(resident,data,device);expected.backward()
        actual=engine.backward(data)
        assert abs(actual-expected.item())<1e-4
        for (name,p),(_,q) in zip(model.named_parameters(),resident.named_parameters()):
            assert p.grad is not None and q.grad is not None,name
            torch.testing.assert_close(p.grad,q.grad.cpu(),rtol=2e-3,atol=3e-5,msg=name)
        a.step();b.step()
        for p,q in zip(model.parameters(),resident.parameters()):torch.testing.assert_close(p,q.cpu(),rtol=2e-3,atol=3e-5)
        engine.parameter_version+=1


def test_joint_disk_accumulation_and_recoverable_updates(tmp_path):
    torch.manual_seed(4);original=MorphModel(configuration());resident=copy.deepcopy(original)
    with torch.device('meta'):template=MorphModel(configuration())
    store=DiskState.create(tmp_path/'state',template,source=original);engine=DiskDLS(template,store)
    optimizer=torch.optim.AdamW(resident.parameters(),lr=.001,weight_decay=.01,foreach=False)
    second=batch();second['tokens']=torch.tensor([[40,30,20,10]]);second['values']=torch.tensor([.8]);second['actions']=torch.tensor([9])
    second['graph']['relations'][0,0]=2;second['graph']['dependency_labels'][0,1,0]=3
    batches=[batch(),second]
    for data in batches:(loss(resident,data,'cpu')/2).backward()
    norm=torch.nn.utils.clip_grad_norm_(resident.parameters(),1.).item();optimizer.step()
    result=engine.step(batches)
    assert abs(result['grad_norm']-norm)<1e-4
    for name,p in resident.named_parameters():torch.testing.assert_close(store.weight(name),p,rtol=2e-3,atol=3e-5,msg=name)
    optimizer.zero_grad()
    for data in batches:(loss(resident,data,'cpu')/2).backward()
    torch.nn.utils.clip_grad_norm_(resident.parameters(),1.);optimizer.step()
    def fail(point):
        if point=='partial_state_write':raise OSError('injected proof update failure')
    store.fault=fail
    with pytest.raises(OSError):engine.step(batches)
    reopened=DiskState.open(store.root,template);DiskDLS(template,reopened).resume_step()
    import numpy as np
    for name,p in resident.named_parameters():
        state=np.memmap(reopened.root/reopened.entries[name]['file'],dtype='float32',mode='r',shape=(3,p.numel()))
        for index,expected in enumerate([p,optimizer.state[p]['exp_avg'],optimizer.state[p]['exp_avg_sq']]):
            torch.testing.assert_close(torch.from_numpy(state[index].copy()).view_as(expected),expected,rtol=3e-3,atol=3e-5,msg=name)


def test_order_graph_conditioning_and_token_causality():
    model=MorphModel(configuration());data=batch();tokens=data['tokens'];graph=data['graph']
    first=model.forward_proof(tokens,graph)
    swapped=copy.deepcopy(graph);swapped['tokens'][0,0],swapped['tokens'][0,1]=graph['tokens'][0,1].clone(),graph['tokens'][0,0].clone()
    assert not torch.allclose(first['logits'],model.forward_proof(tokens,swapped)['logits'])
    changed=tokens.clone();changed[:,2:]=99
    torch.testing.assert_close(first['logits'][:,:2],model.forward_proof(changed,graph)['logits'][:,:2])
    ids=torch.tensor([[[ord(c) for c in 'P -> Q'],[ord(c) for c in 'Q -> P']]])
    embedded=model.embedding(ids);mask=torch.ones_like(ids,dtype=torch.bool)
    state=model.proof_head.encoder(embedded,mask)
    assert not torch.allclose(state[:,0],state[:,1])


def test_explicit_symbolic_capacity_and_cached_generation():
    graph=ProofGraph();graph.add(ProofNode(0,'x'*129))
    with pytest.raises(ValueError,match='capacity'):tensorize([graph],3,statement_length=128)
    model=MorphModel(configuration());data=batch();g=data['graph'];ids=data['tokens']
    full=model.forward_proof(ids,g)
    first=model.forward_proof(ids[:,:2],g);last=model.forward_proof(ids[:,2:],g,first['states'])
    torch.testing.assert_close(full['logits'],torch.cat((first['logits'],last['logits']),1))
    changed=copy.deepcopy(g);changed['tokens'][0,0,128]=ord('F')
    assert not torch.allclose(full['logits'],model.forward_proof(ids,changed)['logits'])
    changed=copy.deepcopy(g);changed['tokens'][0,0,256]=ord('R')
    assert not torch.allclose(full['logits'],model.forward_proof(ids,changed)['logits'])


def test_bf16_joint_cuda_repeated_async_buffers():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(7);master=MorphModel(configuration());resident=copy.deepcopy(master).cuda().bfloat16()
    engine=StreamTrainEngine(master,'cuda',True,torch.bfloat16);data=batch()
    for _ in range(3):
        master.zero_grad();resident.zero_grad()
        with torch.autocast('cuda',dtype=torch.bfloat16):expected=loss(resident,data,'cuda')
        expected.backward();actual=engine.backward(data)
        assert abs(actual-expected.item())<.03
        for (name,p),(_,q) in zip(master.named_parameters(),resident.named_parameters()):
            torch.testing.assert_close(p.grad,q.grad.float().cpu(),rtol=.15,atol=.006,msg=name)


def test_padding_invariance_of_policy_and_value():
    model=MorphModel(configuration());data=batch();graph=data['graph'];ids=data['tokens']
    graph['text_mask']=torch.ones_like(ids,dtype=torch.bool)
    first=model.forward_proof(ids,graph)
    padded=torch.cat((ids,torch.zeros(1,3,dtype=torch.long)),1)
    other=copy.deepcopy(graph);other['text_mask']=torch.tensor([[1,1,1,1,0,0,0]],dtype=torch.bool)
    second=model.forward_proof(padded,other)
    for key in ['value','action_logits','target_logits','dependency_logits']:
        torch.testing.assert_close(first[key],second[key])
