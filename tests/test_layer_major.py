import json
import types
import numpy as np
import torch
from torch.func import functional_call
from morph import MorphConfig,MorphModel
from streamtrain.disk import DiskState,DiskDLS


def legacy_boundaries(self,batches):
    checkpoints=[]
    for batch in batches:
        state=(batch[:,:-1].to(self.device),None);saved=[];self.load(0,self.slots[0])
        for index in range(len(self.stages)-1):
            saved.append(tuple(v.detach().cpu() if v is not None else None for v in state))
            slot=self.slots[index%2]
            if index+1<len(self.stages)-1:self.load(index+1,self.slots[(index+1)%2])
            params=self.parameters(index,slot,False)
            with torch.no_grad(),torch.autocast(self.device.type,dtype=self.dtype,enabled=self.dtype in (torch.float16,torch.bfloat16)):
                state=functional_call(self.stages[index],params,state)
            self.release(slot)
        saved.append(tuple(v.detach().cpu() if v is not None else None for v in state));checkpoints.append(saved)
    return checkpoints


def test_layer_major_distinct_k3_boundaries_and_exact_updates(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(5)
    cfg=MorphConfig(vocab_size=256,d_model=16,num_layers=5,d_ff=32,d_state=4,head_dim=8,graph_enabled=True,d_graph=8,graph_nodes=2,graph_interval=2)
    source=MorphModel(cfg);oldstore=DiskState.create(tmp_path/'old',source,source=source);newstore=DiskState.create(tmp_path/'new',source,source=source)
    old=DiskDLS(source,oldstore);new=DiskDLS(source,newstore);old.boundaries=types.MethodType(legacy_boundaries,old)
    batches=[torch.randint(256,(1,9)) for _ in range(3)]
    a=old.boundaries(batches);b=new.boundaries(batches)
    for micro in range(3):
        for state_a,state_b in zip(a[micro],b[micro]):
            for va,vb in zip(state_a,state_b):
                if va is not None:torch.testing.assert_close(va,vb,atol=0,rtol=0)
    assert old.transfer_bytes['h2d_weights']==new.transfer_bytes['h2d_weights']*3
    first=old.step(batches);second=new.step(batches)
    assert first==second
    for entry in oldstore.entries.values():
        np.testing.assert_array_equal(np.fromfile(oldstore.root/entry['file'],dtype=np.float32),np.fromfile(newstore.root/entry['file'],dtype=np.float32))


def test_equal_packing_preserves_effective_gradient_and_token_count(tmp_path):
    from streamtrain.precision import pack_lm_microbatches
    torch.manual_seed(9);cfg=MorphConfig(vocab_size=256,d_model=16,num_layers=2,d_ff=32,d_state=4,head_dim=8)
    source=MorphModel(cfg);batches=[torch.randint(256,(1,9)) for _ in range(8)];packed=pack_lm_microbatches(batches,2)
    assert len(packed)==4 and sum(x.shape[0]*(x.shape[1]-1) for x in packed)==64
    assert pack_lm_microbatches(batches[:3],2) is not None
    assert len(pack_lm_microbatches(batches[:3],2))==3
    for row,packed_row in zip(batches,torch.cat(packed).split(1)):torch.testing.assert_close(row,packed_row)
    import copy
    a=copy.deepcopy(source);b=copy.deepcopy(source)
    for batch in batches:(a.loss(batch)/8).backward()
    for batch in packed:(b.loss(batch)/4).backward()
    for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa.grad,pb.grad,atol=2e-7,rtol=2e-5)
    stores=[DiskState.create(tmp_path/name,source,source=source) for name in ['unpacked','packed']]
    DiskDLS(source,stores[0]).step(batches);DiskDLS(source,stores[1]).step(packed)
    for entry in stores[0].entries.values():
        np.testing.assert_allclose(np.fromfile(stores[0].root/entry['file'],dtype=np.float32),np.fromfile(stores[1].root/entry['file'],dtype=np.float32),atol=2e-6,rtol=2e-4)
