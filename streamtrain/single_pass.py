"""One backward pass with bounded FP32 gradient storage and recoverable AdamW.

The complete immutable gradient stash is durable before canonical state changes.
Recovery applies saved gradients only; partially updated weights never participate
in another backward pass. This trades host/disk capacity for one less replay.
"""
import json
import math
import os
from pathlib import Path
import shutil
import torch
from .disk import durable_json,durable_torch
from .sparse_disk import SparseDiskDLS


class SinglePassDiskDLS(SparseDiskDLS):
    transaction_kind='stored-gradients-v1'

    def __init__(self,*args,stash_budget_bytes=4*2**30,**kwargs):
        if type(stash_budget_bytes) is not int or stash_budget_bytes<1:
            raise ValueError('positive integer gradient stash budget required')
        super().__init__(*args,**kwargs)
        self.stash_budget_bytes=stash_budget_bytes
        self.stash_path=self.store.root/'gradient-stash'
        self.last_stash_bytes=0
        if not self.store.manifest.get('transaction') and not self.store.manifest['dirty']:
            self._cleanup_stash()

    def _cleanup_stash(self):
        shutil.rmtree(self.stash_path,ignore_errors=True)
        shutil.rmtree(self.store.root/'gradient-stash.tmp',ignore_errors=True)

    def _routing_plan(self):
        getter=getattr(self.model,'get_routing_plan',None)
        return getter() if getter is not None else None

    def _preflight(self,checkpoints):
        # Upper bound before any gradient clone/allocation: deduplicate tied
        # parameters and include all active-stage tensors, even unused heads.
        for index in range(len(self.stages)):
            self.select(index,[saved[index] for saved in checkpoints])
        names={self.canonical[id(p)] for index,spec in enumerate(self.all_specs)
               if self.active_stages[index] for _,p in spec}
        required=sum(self.store.entries[name]['size']*4 for name in names)
        if required>self.stash_budget_bytes:
            raise MemoryError(f'gradient stash needs up to {required} bytes; budget {self.stash_budget_bytes}')
        return required

    def _persist(self,gradients,stages,norm,loss,clip):
        temporary=self.store.root/'gradient-stash.tmp'
        self._cleanup_stash();temporary.mkdir()
        records={}
        for name,gradient in gradients.items():
            filename=self.store.entries[name]['file']+'.pt'
            durable_torch(temporary/filename,gradient)
            records[name]=dict(file=filename,shape=list(gradient.shape),size=gradient.numel())
        metadata=dict(kind=self.transaction_kind,base_step=self.store.manifest['step'],
                      dtype=str(self.dtype),fp8_transfer=self.fp8_transfer,lr=self.lr,
                      norm=norm,loss=loss,clip=clip,loss_scale=self.loss_scale,
                      loss_scaler=self.scaler.state_dict(),
                      routing_plan=self._routing_plan(),records=records,
                      stages={str(index):names for index,names in stages.items()},
                      stage_count=len(self.stages),stash_bytes=self.last_stash_bytes)
        durable_json(temporary/'metadata.json',metadata)
        os.replace(temporary,self.stash_path)
        descriptor=os.open(self.store.root,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
        self.store.fault('gradient_stash_durable')
        return metadata

    def step(self,batches,max_norm=1.):
        if self.store.manifest['dirty'] or self.store.manifest.get('transaction'):
            raise RuntimeError('pending transaction: call resume_step before starting another step')
        if not batches or max_norm<=0 or not math.isfinite(max_norm):
            raise ValueError('nonempty microbatches and positive finite clipping threshold required')
        setter=getattr(self.model,'set_routing_step',None)
        if setter is not None:setter(self.store.manifest['step'])
        checkpoints=self.boundaries(batches)
        self._preflight(checkpoints)
        while True:
            gradients={};stages={};self.last_stash_bytes=0
            self.loss_scale=self.scaler.scale if self.dtype==torch.float16 else 1.
            def capture(index,name,gradient):
                if name in gradients:raise RuntimeError('canonical gradient captured more than once')
                size=gradient.numel()*4
                if self.last_stash_bytes+size>self.stash_budget_bytes:
                    raise MemoryError('gradient stash exceeded preflight budget')
                gradients[name]=gradient.detach().to(device='cpu',dtype=torch.float32).clone()
                stages.setdefault(index,[]).append(name);self.last_stash_bytes+=size
            norm,loss=self.reverse(batches,checkpoints,gradient_callback=capture)
            if math.isfinite(norm) and math.isfinite(loss):break
            del gradients
            if self.dtype!=torch.float16 or not math.isfinite(loss):
                raise FloatingPointError('nonfinite gradients/loss; canonical state unchanged')
            self.scaler.overflow()
        clip=min(1.,max_norm/(norm+1e-6))
        self._persist(gradients,stages,norm,loss,clip)
        # Release CPU clones before loading durable gradients for stage apply.
        gradients.clear();del gradients,checkpoints
        cursor='cursor-stored-initial.pt'
        durable_torch(self.store.root/cursor,dict(cotangents=[],tied=None))
        self.store.manifest.update(dirty=True,transaction=dict(kind=self.transaction_kind,
            cursor=cursor,next_stage=len(self.stages)-1,journal=None,
            gradient_stash='gradient-stash',routing_plan=self._routing_plan()))
        durable_json(self.store.root/'manifest.json',self.store.manifest)
        self.store.fault('stored_transaction_durable')
        return self._apply_stash()

    def _read_metadata(self):
        transaction=self.store.manifest.get('transaction',{})
        if transaction.get('kind')!=self.transaction_kind:
            raise ValueError('transaction is not a stored-gradient transaction')
        if transaction.get('gradient_stash')!='gradient-stash':
            raise ValueError('invalid gradient stash reference')
        metadata=json.loads((self.stash_path/'metadata.json').read_text())
        if (metadata['kind']!=self.transaction_kind or metadata['base_step']!=self.store.manifest['step']
            or metadata['dtype']!=str(self.dtype) or metadata['fp8_transfer']!=self.fp8_transfer
            or metadata['lr']!=self.lr or metadata['stage_count']!=len(self.stages)):
            raise ValueError('stored-gradient replay settings differ')
        if metadata['routing_plan']!=transaction.get('routing_plan'):
            raise ValueError('stored-gradient roster differs from committed transaction')
        # Validate the complete roster/schema before applying any gradient.
        stage_map=metadata.get('stages')
        records=metadata.get('records')
        if not isinstance(stage_map,dict):raise ValueError('invalid stored-gradient stage manifest')
        if not isinstance(records,dict):raise ValueError('invalid stored-gradient records manifest')
        owners={}
        for index,spec in enumerate(self.all_specs):
            for _,parameter in spec:owners.setdefault(self.canonical[id(parameter)],set()).add(index)
        embedding=getattr(getattr(self.model,'embedding',None),'weight',None)
        embedding_name=self.canonical.get(id(embedding)) if embedding is not None else None
        flattened=[]
        for key,names in stage_map.items():
            if not isinstance(key,str) or not key.isdecimal() or str(int(key))!=key:
                raise ValueError('invalid stored-gradient stage index')
            index=int(key)
            if index>=len(self.stages):raise ValueError('stored-gradient stage index is out of range')
            if not isinstance(names,list) or not all(isinstance(name,str) for name in names):
                raise ValueError('invalid stored-gradient stage roster')
            for name in names:
                allowed=owners.get(name,set())
                if name==embedding_name:allowed={0} if 0 in allowed else allowed
                if index not in allowed:
                    raise ValueError(f'stored-gradient parameter {name!r} does not belong to stage {index}')
                flattened.append(name)
        if len(flattened)!=len(set(flattened)) or set(flattened)!=set(records):
            raise ValueError('invalid stored-gradient stage manifest')
        for name,record in records.items():
            entry=self.store.entries.get(name)
            if (entry is None or record['file']!=entry['file']+'.pt' or record['shape']!=entry['shape']
                or record['size']!=entry['size'] or not (self.stash_path/record['file']).is_file()):
                raise ValueError('invalid stored-gradient tensor schema')
        if any(not math.isfinite(metadata[key]) for key in ('norm','loss','clip','loss_scale')):
            raise ValueError('nonfinite stored-gradient metadata')
        restore=getattr(self.model,'restore_routing_plan',None)
        if metadata['routing_plan'] is not None:
            if restore is None:raise ValueError('model cannot restore stored routing plan')
            restore(metadata['routing_plan'])
        self.last_stash_bytes=metadata['stash_bytes']
        return metadata

    def _apply_stash(self):
        metadata=self._read_metadata()
        self.store.recover_stage()
        transaction=self.store.manifest['transaction']
        for index in reversed(range(transaction['next_stage']+1)):
            names=metadata['stages'].get(str(index),[])
            self.report('stored_gradient_apply',index)
            self.store.prepare_stage(index,names)
            clocks={}
            for name in names:
                record=metadata['records'][name]
                gradient=torch.load(self.stash_path/record['file'],weights_only=True,map_location='cpu')
                if gradient.dtype!=torch.float32 or list(gradient.shape)!=record['shape'] or not torch.isfinite(gradient).all():
                    raise ValueError('invalid/nonfinite stored gradient')
                step=self.store.manifest.get('parameter_steps',{}).get(name,0)+1
                self.store.update(name,gradient,step,self.lr,metadata['clip']);clocks[name]=step
                del gradient
            self.store.commit_stage(index,[],None,parameter_steps=clocks)
        if self.dtype==torch.float16:
            self.scaler.load_state_dict(metadata['loss_scaler'])
            self.scaler.scale=metadata['loss_scale'];self.scaler.success()
            self.store.manifest['loss_scaler']=self.scaler.state_dict()
        try:self.store.finish_step()
        finally:
            if not self.store.manifest['dirty']:self._cleanup_stash()
        self.parameter_version=self.store.manifest['step']
        return dict(loss=metadata['loss'],grad_norm=metadata['norm'])

    def resume_step(self):
        # No activation checkpoints or backwards are needed or permitted here.
        return self._apply_stash()
