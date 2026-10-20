"""Dropless expert-stage replay with conditional (None-gradient) FP32 AdamW.

An inactive expert has no fetch, gradient allocation or optimizer update. Clocks
advance per parameter only when autograd supplies a gradient, matching AdamW
with set_to_none=True. Full accumulated gradients are globally clipped.
"""
import math
import time
import torch
from torch.func import functional_call
from .disk import DiskDLS
from .engine import unpack_batch,stage_loss


class SparseDiskDLS(DiskDLS):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.all_specs=[list(spec) for spec in self.specs]
        self.active_stages=[True]*len(self.stages)

    def select(self,index,states):
        predicate=getattr(self.stages[index],'active_for',None)
        active=predicate is None or any(predicate(*state) for state in states)
        self.active_stages[index]=active
        self.specs[index]=self.all_specs[index] if active else []
        return active

    def boundaries(self,batches):
        saved=[[] for _ in batches];states=[];graphs=[];pending=set()
        for batch in batches:
            tokens,graph=unpack_batch(batch,'cpu');graphs.append(graph)
            states.append((tokens if graph is not None else tokens[:,:-1],None))
        for index in range(len(self.stages)-1):
            active=self.select(index,states)
            for micro,state in enumerate(states):saved[micro].append(state)
            if not active:continue
            slot=self.slots[index%2]
            if index not in pending:self.load(index,slot)
            # A router produces a previously unknown route plan. Wait for that
            # boundary before prefetching experts. Expert stages preserve gates.
            following=index+1
            if following<len(self.stages)-1:
                next_pred=getattr(self.stages[following],'active_for',None)
                current_pred=getattr(self.stages[index],'active_for',None)
                if (next_pred is None or current_pred is not None) and self.select(following,states):
                    self.load(following,self.slots[following%2]);pending.add(following)
            parameters=self.parameters(index,slot,False)
            for micro,state in enumerate(states):
                self.report('forward',index,micro)
                device_state=tuple(v.to(self.device) if v is not None else None for v in state)
                graph=None if graphs[micro] is None else {k:v.to(self.device) for k,v in graphs[micro].items()}
                with torch.no_grad(),torch.autocast(self.device.type,dtype=self.dtype,enabled=self.dtype in (torch.float16,torch.bfloat16)):
                    result=functional_call(self.stages[index],parameters,(*device_state,graph))
                states[micro]=tuple(v.detach().cpu() if v is not None else None for v in result)
            self.release(slot)
        for micro,state in enumerate(states):saved[micro].append(state)
        return saved

    def reverse(self,batches,checkpoints,update=False,clip=1.,initial_cursor=False,gradient_callback=None):
        for index in range(len(self.stages)):
            self.select(index,[saved[index] for saved in checkpoints])
        cotangents=[None]*len(batches);tied=None;start=len(self.stages)-1;pending=set()
        if update and self.store.manifest.get('transaction') and not initial_cursor:
            self.store.recover_stage();cursor=self.store.load_cursor()
            cotangents=[None if values is None else [v.to(self.device) for v in values] for values in cursor['cotangents']]
            tied=cursor['tied'];start=self.store.manifest['transaction']['next_stage']
        embedding_name=self.canonical[id(self.model.embedding.weight)]
        norm_squared=total_loss=0.
        for index in reversed(range(start+1)):
            self.report('update_reverse' if update else 'norm_reverse',index)
            if not self.active_stages[index]:
                if update:
                    self.store.prepare_stage(index,[])
                    self.store.commit_stage(index,cotangents,tied)
                continue
            slot=self.slots[index%2]
            if index not in pending:self.load(index,slot)
            if index and self.active_stages[index-1]:
                self.load(index-1,self.slots[(index-1)%2]);pending.add(index-1)
            sizes=[p.numel() for _,p in self.specs[index]]
            total=sum(sizes)
            if self.device.type=='cuda':
                # Slot already reserves one FP32 gradient window. Sum all
                # microbatches here and evacuate once, rather than K transfers.
                flat_gradient=slot.grad_device[:total];flat_gradient.zero_()
                accumulated=[v.view_as(p) for v,(_,p) in zip(flat_gradient.split(sizes),self.specs[index])]
            else:accumulated=[torch.zeros(p.shape) for _,p in self.specs[index]]
            seen=[False]*len(accumulated)
            for micro,batch in enumerate(batches):
                _,graph=unpack_batch(batch,self.device)
                params=self.parameters(index,slot,True)
                state=tuple(v.to(self.device).detach().requires_grad_(bool(index)) if v is not None else None for v in checkpoints[micro][index])
                state_values=[v for v in state if v is not None] if index else []
                weights=[params[name] for name,_ in self.specs[index]]
                with torch.enable_grad(),torch.autocast(self.device.type,dtype=self.dtype,enabled=self.dtype in (torch.float16,torch.bfloat16)):
                    output=functional_call(self.stages[index],params,(*state,graph))
                    if cotangents[micro] is None:
                        loss=stage_loss(output,batch,self.device)
                        total_loss+=loss.detach().item()/len(batches)
                        grads=torch.autograd.grad(loss*self.loss_scale/len(batches),state_values+weights,allow_unused=True)
                    else:
                        grads=torch.autograd.grad([v for v in output if v is not None],state_values+weights,cotangents[micro],allow_unused=True)
                cotangents[micro]=[torch.zeros_like(v) if g is None else g for v,g in zip(state_values,grads)] if index else None
                for j,(destination,g) in enumerate(zip(accumulated,grads[len(state_values):])):
                    if g is not None:
                        seen[j]=True
                        value=g.detach().float()/self.loss_scale
                        destination.add_(value if self.device.type=='cuda' else value.cpu())
                        if self.device.type!='cuda':self.transfer_bytes['d2h_gradients']+=g.numel()*4
                del params,output,weights,grads,state_values
            if self.device.type=='cuda':
                if self.async_mode:
                    self.release(slot)
                    with torch.cuda.stream(self.d2h):
                        self.d2h.wait_event(slot.done)
                        slot.grad_host[:total].copy_(flat_gradient,non_blocking=True)
                        copied=torch.cuda.Event();copied.record(self.d2h)
                    copied.synchronize()
                else:slot.grad_host[:total].copy_(flat_gradient)
                accumulated=[v.view_as(p) for v,(_,p) in zip(slot.grad_host[:total].split(sizes),self.specs[index])]
                self.transfer_bytes['d2h_gradients']+=total*4
            selected=[(self.canonical[id(p)],g) for ((_,p),g,used) in zip(self.specs[index],accumulated,seen) if used]
            names=[name for name,_ in selected if index==0 or name!=embedding_name]
            if update:self.store.prepare_stage(index,names)
            next_clocks={}
            for name,gradient in selected:
                if name==embedding_name and index!=0:
                    tied=gradient.clone();continue
                if name==embedding_name and tied is not None:
                    gradient.add_(tied);tied=None
                if update:
                    step=self.store.manifest.get('parameter_steps',{}).get(name,0)+1
                    self.store.update(name,gradient,step,self.lr,clip);next_clocks[name]=step
                else:
                    if gradient_callback is not None:gradient_callback(index,name,gradient)
                    for piece in gradient.flatten().split(1048576):norm_squared+=piece.double().square().sum().item()
            self.release(slot)
            if update:self.store.commit_stage(index,cotangents,tied,parameter_steps=next_clocks)
            del accumulated
        return math.sqrt(norm_squared),total_loss

    def resume_step(self):
        if self.store.manifest.get('transaction',{}).get('kind')=='stored-gradients-v1':
            raise ValueError('stored-gradient transaction requires SinglePassDiskDLS')
        return super().resume_step()
