"""Bounded-stage proof inference from canonical disk state, no full export."""
import torch
from torch.func import functional_call
from .disk import DiskDLS


class DiskProofModel:
    supports_cache=False
    def __init__(self,template,store,device='cpu',dtype=torch.float32,executor=None):
        if store.manifest['dirty']:raise ValueError('resume disk transaction before inference')
        self.config=template.config
        from .sparse_disk import SparseDiskDLS
        executor_type=SparseDiskDLS if getattr(template.config,'model_type',None)=='delta' else DiskDLS
        self.executor=executor or executor_type(template,store,device,dtype,async_mode=str(device).startswith('cuda'),cpu_prefetch=True)
        self.closed=False

    def close(self):
        if not self.closed:
            self.executor.close();self.closed=True

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def forward_proof(self,input_ids,graph,states=None):
        if self.closed:raise RuntimeError('disk proof model is closed')
        if states is not None:raise ValueError('disk proof inference recomputes full prefixes')
        executor=self.executor
        state=(input_ids.to(executor.device),None)
        graph={key:value.to(executor.device) for key,value in graph.items()}
        for index,stage in enumerate(executor.stages):
            if hasattr(stage,'active_for') and not stage.active_for(*state):continue
            slot=executor.slots[index%2];executor.load(index,slot)
            params=executor.parameters(index,slot,False)
            with torch.no_grad(),torch.autocast(executor.device.type,dtype=executor.dtype,enabled=executor.dtype in (torch.bfloat16,torch.float16)):
                output=functional_call(stage,params,(*state,graph))
            executor.release(slot)
            if index==len(executor.stages)-1:return {**output,'states':None}
            state=output
