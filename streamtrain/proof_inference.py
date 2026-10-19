"""Bounded-stage proof inference from canonical disk state, no full export."""
import torch
from torch.func import functional_call
from .disk import DiskDLS


class DiskProofModel:
    supports_cache=False
    def __init__(self,template,store,device='cpu',dtype=torch.float32):
        if store.manifest['dirty']:raise ValueError('resume disk transaction before inference')
        self.config=template.config
        self.executor=DiskDLS(template,store,device,dtype,async_mode=device.startswith('cuda'))

    def forward_proof(self,input_ids,graph,states=None):
        executor=self.executor
        state=(input_ids.to(executor.device),None)
        graph={key:value.to(executor.device) for key,value in graph.items()}
        for index,stage in enumerate(executor.stages):
            slot=executor.slots[index%2];executor.load(index,slot)
            params=executor.parameters(index,slot,False)
            with torch.no_grad(),torch.autocast(executor.device.type,dtype=executor.dtype,enabled=executor.dtype==torch.bfloat16):
                output=functional_call(stage,params,(*state,graph))
            executor.release(slot)
            if index==len(executor.stages)-1:return dict(**output,states=None)
            state=output
