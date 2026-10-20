"""Small end-to-end qualification of proof replay and CUDA prefetch isolation."""
import copy
import pytest
import torch
from morph.delta_config import DeltaConfig
from morph.model.delta import DeltaModel
from streamtrain.disk import DiskState
from streamtrain.sparse_disk import SparseDiskDLS
from streamtrain.engine import stage_loss


def config(**kwargs):
    return DeltaConfig(d_model=8,num_layers=1,d_ff=16,d_state=2,head_dim=4,
                       d_graph=8,graph_nodes=3,expert_count=3,expert_depth=1,
                       expert_ffn=16,latent_dim=4,memory_dim=4,
                       statement_field_length=16,**kwargs)


def test_cuda_fp16_prefetch_never_fetches_or_updates_inactive_experts(tmp_path):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(23)
    cfg=config(proof_enabled=False)
    source=DeltaModel(cfg)
    with torch.no_grad():
        source.router.projection.weight.zero_()
        source.router.projection.bias.copy_(torch.tensor([10.,-10.,-10.]))
    with torch.device('meta'): template=DeltaModel(cfg)
    a=DiskState.create(tmp_path/'async',template,source=source)
    b=DiskState.create(tmp_path/'sync',template,source=source)
    inactive={name:(a.root/entry['file']).read_bytes() for name,entry in a.entries.items()
              if name.startswith(('experts.1.','experts.2.'))}
    reads=[];read=a.read_into
    def recording(name,destination):
        reads.append(name);return read(name,destination)
    a.read_into=recording
    trainers=[SparseDiskDLS(template,a,device='cuda',dtype=torch.float16,async_mode=True,cpu_prefetch=True),
              SparseDiskDLS(template,b,device='cuda',dtype=torch.float16,async_mode=True)]
    try:
        batch=[torch.tensor([[1,2,3,4,5]])]
        actual=trainers[0].step(batch);expected=trainers[1].step(batch)
        assert actual==expected
        for name in a.entries:
            assert (a.root/a.entries[name]['file']).read_bytes()==(b.root/b.entries[name]['file']).read_bytes()
        assert not any(name in inactive for name in reads)
        for name,payload in inactive.items():
            assert (a.root/a.entries[name]['file']).read_bytes()==payload
            assert name not in a.manifest['parameter_steps']
    finally:
        for trainer in trainers:trainer.close()


def proof_batch():
    from morph.graph.schema import ProofGraph,ProofNode,Hyperedge,Relation
    from morph.graph.tensors import tensorize
    graph=ProofGraph();graph.add(ProofNode(0,'a'));graph.add(ProofNode(1,'b'))
    graph.connect(Hyperedge((0,),1,Relation.REQUIRES))
    tensors,_=tensorize([graph],3,statement_length=16)
    return dict(tokens=torch.tensor([[1,2,3,4]]),graph=tensors,
                actions=torch.tensor([2]),targets=torch.tensor([0]),values=torch.tensor([.4]))


@pytest.mark.parametrize('joint', [False, True])
def test_joint_proof_replay_matches_resident_conditional_adam_and_clocks(tmp_path, joint):
    torch.manual_seed(31)
    cfg=config(proof_enabled=True)
    original=DeltaModel(cfg);resident=copy.deepcopy(original)
    with torch.device('meta'): template=DeltaModel(cfg)
    store=DiskState.create(tmp_path/'joint',template,source=original)
    trainer=SparseDiskDLS(template,store,cpu_prefetch=True,lr=.002)
    optimizer=torch.optim.AdamW(resident.parameters(),lr=.002,weight_decay=.01,foreach=False)
    data=proof_batch() if joint else torch.tensor([[1,2,3,4]])
    try:
        optimizer.zero_grad(set_to_none=True)
        output=resident(data['tokens'],data['graph']) if joint else resident(data[:,:-1])
        loss=stage_loss(output,data,'cpu');loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(resident.parameters(),1.).item()
        optimizer.step()
        actual=trainer.step([data])
        assert actual['loss']==pytest.approx(loss.item(),rel=1e-6)
        assert actual['grad_norm']==pytest.approx(norm,rel=2e-5,abs=2e-6)
        for name,parameter in resident.named_parameters():
            torch.testing.assert_close(store.weight(name),parameter,rtol=5e-4,atol=3e-6,msg=name)
            expected=int(optimizer.state[parameter]['step']) if parameter in optimizer.state else 0
            assert store.manifest['parameter_steps'].get(name,0)==expected,name
    finally:trainer.close()


def test_meta_disk_initialization_is_repeatable_and_preserves_ssm_and_norm_defaults(tmp_path):
    cfg=config(proof_enabled=True)
    with torch.device('meta'): template=DeltaModel(cfg)
    a=DiskState.create(tmp_path/'seed-a',template,seed=17)
    b=DiskState.create(tmp_path/'seed-b',template,seed=17)
    for name in a.entries:
        assert (a.root/a.entries[name]['file']).read_bytes()==(b.root/b.entries[name]['file']).read_bytes()
    torch.testing.assert_close(a.weight('norm.weight'),torch.ones(8))
    torch.testing.assert_close(a.weight('session_memory.encoder.normalization.weight'),torch.ones(8))
    torch.testing.assert_close(a.weight('experts.0.input_norm.weight'),torch.ones(8))
    torch.testing.assert_close(a.weight('session_memory.memory.gates.bias'),torch.zeros(2))
    torch.testing.assert_close(a.weight('blocks.0.ssm.dt_bias'),torch.full((4,),-2.))
