"""Bounded-memory two-pass disk AdamW. No optimizer approximation.

Canonical FP32 weight/m/v files; gradients exist for one stage at a time.
Recoverable stage transactions retain one undo window and a durable replay
cursor; legacy dirty stores without transaction state fail closed.
"""
import json
import math
import os
import shutil
import time
from contextlib import nullcontext
from contextvars import ContextVar
from pathlib import Path
import numpy as np
import torch
from torch.func import functional_call
from torch.nn import functional as F
from .engine import StreamTrainEngine,make_stages,unpack_batch,stage_loss


_profile_store=ContextVar('disk_profile_store',default=None)

def profile_fsync(descriptor):
    started=time.perf_counter()
    try:os.fsync(descriptor)
    finally:
        store=_profile_store.get()
        if store is not None:
            store.profile_seconds['fsync_wall']=store.profile_seconds.get('fsync_wall',0.)+time.perf_counter()-started
            store.profile_counts['fsync']=store.profile_counts.get('fsync',0)+1


def durable_json(path, data):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as handle:
        json.dump(data, handle)
        handle.flush()
        profile_fsync(handle.fileno())
    os.replace(temp, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        profile_fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_torch(path, value):
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as handle:
        torch.save(value,handle);handle.flush();profile_fsync(handle.fileno())
    os.replace(temp,path)
    descriptor=os.open(path.parent,os.O_RDONLY)
    try:profile_fsync(descriptor)
    finally:os.close(descriptor)


class DiskState:
    def __init__(self, root, model, manifest):
        self.root, self.manifest = Path(root), manifest
        self.profile_seconds={};self.profile_counts={};self.profile_bytes={}
        # Wrap virtual methods so packed/cache overrides are measured too.
        for operation in ('read_into','prepare_stage','update','commit_stage','recover_stage'):
            method=getattr(self,operation)
            def measured(*args,_method=method,_operation=operation,**kwargs):
                started=time.perf_counter();token=_profile_store.set(self)
                try:return _method(*args,**kwargs)
                finally:
                    _profile_store.reset(token)
                    self.profile_seconds[_operation+'_wall']=self.profile_seconds.get(_operation+'_wall',0.)+time.perf_counter()-started
                    self.profile_counts[_operation]=self.profile_counts.get(_operation,0)+1
                    byte_count=0
                    if _operation=='read_into':byte_count=self.entries[args[0]]['size']*4
                    elif _operation=='update':byte_count=self.entries[args[0]]['size']*12
                    elif _operation=='prepare_stage':byte_count=sum(self.entries[n]['size']*12 for n in args[1])
                    self.profile_bytes[_operation]=self.profile_bytes.get(_operation,0)+byte_count
            setattr(self,operation,measured)
        self.entries = {name: dict(shape=list(p.shape), size=p.numel(), file=f'{i:05}.bin')
                        for i, (name, p) in enumerate(model.named_parameters())}
        if self.entries != manifest['entries']:
            raise ValueError('disk state tensor schema differs')

    def profile_snapshot(self):
        """Flat cumulative counters. Fsync overlaps enclosing operation wall time."""
        return {**self.profile_seconds,
                **{name+'_count':value for name,value in self.profile_counts.items()},
                **{name+'_bytes':value for name,value in self.profile_bytes.items()}}

    @classmethod
    def create(cls, root, model, source=None, seed=42, chunk_elements=1048576, progress=None):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        entries = {name: dict(shape=list(p.shape), size=p.numel(), file=f'{i:05}.bin')
                   for i, (name, p) in enumerate(model.named_parameters())}
        manifest = dict(entries=entries, step=0, dirty=True, config=model.config.to_dict())
        durable_json(root / 'manifest.json', manifest)
        generator = torch.Generator().manual_seed(seed)
        original = dict(source.named_parameters()) if source is not None else None
        for name, entry in entries.items():
            size = entry['size']
            path = root / entry['file']
            # Reserve physical space, rather than promising sparse-file capacity.
            with path.open('wb') as handle:
                os.posix_fallocate(handle.fileno(), 0, size * 12)
            mapped = np.memmap(path, dtype='float32', mode='r+', shape=(3, size))
            for start in range(0, size, chunk_elements):
                end = min(start + chunk_elements, size)
                if original is not None:
                    value = original[name].detach().flatten()[start:end]
                elif name.endswith('dt_bias'):
                    value = torch.full((end-start,), -2.)
                elif name.endswith('a_log') or name.endswith('bias'):
                    value = torch.zeros(end-start)
                elif name.endswith('skip') or ('norm' in name and name.endswith('weight')):
                    value = torch.ones(end-start)
                else:
                    value = torch.randn(end-start, generator=generator) * .02
                mapped[0, start:end] = value.numpy()
                mapped[1:, start:end] = 0
            mapped.flush()
            del mapped
            if progress is not None:progress(name, sum(e['size'] for e in entries.values()), size)
        manifest['dirty'] = False
        durable_json(root / 'manifest.json', manifest)
        return cls(root, model, manifest)

    @classmethod
    def open(cls, root, model):
        root = Path(root)
        manifest = json.loads((root / 'manifest.json').read_text())
        if manifest.get('journal_format','legacy') != getattr(cls,'journal_format','legacy'):
            raise ValueError('disk journal format differs; use matching state class')
        if manifest['dirty'] and not manifest.get('transaction'):
            raise RuntimeError('incomplete disk update: state is not a valid resume point; restore a backup')
        if type(model.config)(**manifest['config']).to_dict() != model.config.to_dict():
            raise ValueError('disk model config differs')
        if not manifest['dirty']:
            for path in root.glob('cursor-*.pt'):path.unlink(missing_ok=True)
            (root/'transaction-input.pt').unlink(missing_ok=True)
            (root/'transaction-tied.pt').unlink(missing_ok=True)
            shutil.rmtree(root/'journal',ignore_errors=True)
        return cls(root, model, manifest)

    def fault(self,point):
        """Fault-injection hook; production default has no side effects."""
        return None

    def recover_stage(self):
        transaction=self.manifest.get('transaction')
        if not transaction:
            return
        journal=self.root/'journal'
        if transaction.get('journal') is not None:
            # A committed cursor always clears journal atomically. If journal is
            # present, this stage did not commit and its complete old state wins.
            for path in journal.glob('*.bin'):
                shutil.copyfile(path,self.root/path.name)
                descriptor=os.open(self.root/path.name,os.O_RDWR)
                try:profile_fsync(descriptor)
                finally:os.close(descriptor)
            transaction['journal']=None
            durable_json(self.root/'manifest.json',self.manifest)
        if journal.exists():
            shutil.rmtree(journal)
        for path in self.root.glob('cursor-*.pt'):
            if path.name!=transaction['cursor']:path.unlink()

    def prepare_stage(self,index,names):
        journal=self.root/'journal'
        if journal.exists():shutil.rmtree(journal)
        journal.mkdir()
        for name in names:
            file=self.entries[name]['file']
            shutil.copyfile(self.root/file,journal/file)
            descriptor=os.open(journal/file,os.O_RDWR)
            try:profile_fsync(descriptor)
            finally:os.close(descriptor)
        descriptor=os.open(journal,os.O_RDONLY)
        try:profile_fsync(descriptor)
        finally:os.close(descriptor)
        self.fault('journal_durable')
        self.manifest['transaction']['journal']=index
        durable_json(self.root/'manifest.json',self.manifest)

    def commit_stage(self,index,cotangents,tied,parameter_steps=None):
        transaction=self.manifest['transaction']
        previous=transaction['cursor']
        filename=f'cursor-{index}.pt'
        tied_ref=None
        if tied is not None:
            tied_ref=transaction.get('tied_ref')
            if tied_ref is None:
                # Only the first committed cursor publishes this immutable
                # payload. An orphan from an uncommitted attempt is replaced.
                tied_ref='transaction-tied.pt'
                durable_torch(self.root/tied_ref,tied.detach().cpu())
                self.fault('tied_file_durable')
        payload=dict(cotangents=[None if values is None else [v.detach().cpu() for v in values] for values in cotangents],
                     tied=None,tied_ref=tied_ref)
        if parameter_steps is not None:
            parameter_steps={**self.manifest.get('parameter_steps',{}),**parameter_steps}
            payload['parameter_steps']=parameter_steps
        durable_torch(self.root/filename,payload)
        self.fault('cursor_file_durable')
        # Publish clocks and the replay cursor in the same atomic manifest.
        transaction.update(cursor=filename,next_stage=index-1,journal=None)
        if tied_ref is not None:transaction['tied_ref']=tied_ref
        if parameter_steps is not None:self.manifest['parameter_steps']=dict(parameter_steps)
        durable_json(self.root/'manifest.json',self.manifest)
        self.fault('cursor_manifest_durable')
        shutil.rmtree(self.root/'journal',ignore_errors=True)
        if previous!=filename:(self.root/previous).unlink(missing_ok=True)

    def load_cursor(self):
        cursor=torch.load(self.root/self.manifest['transaction']['cursor'],weights_only=True,map_location='cpu')
        reference=cursor.get('tied_ref')
        if reference is not None:
            if reference!='transaction-tied.pt':raise ValueError('invalid tied-gradient reference')
            cursor['tied']=torch.load(self.root/reference,weights_only=True,map_location='cpu')
        return cursor

    def read_into(self, name, destination):
        entry = self.entries[name]
        mapped = np.memmap(self.root / entry['file'], dtype='float32', mode='r', shape=(3, entry['size']))
        # NumPy copyto writes directly to bounded destination without a tensor alias.
        np.copyto(destination.numpy().reshape(-1), mapped[0], casting='unsafe')
        del mapped

    def weight(self, name):
        """Materialize one parameter, for export or small correctness checks."""
        value = torch.empty(self.entries[name]['shape'])
        self.read_into(name, value)
        return value

    def begin_step(self):
        if self.manifest['dirty']:
            raise RuntimeError('incomplete prior update')
        self.manifest['dirty'] = True
        durable_json(self.root / 'manifest.json', self.manifest)

    def update(self, name, gradient, step, lr, clip, chunk_elements=1048576):
        entry = self.entries[name]
        path = self.root / entry['file']
        mapped = np.memmap(path, dtype='float32', mode='r+', shape=(3, entry['size']))
        for start in range(0, entry['size'], chunk_elements):
            end = min(start + chunk_elements, entry['size'])
            weight, first, second = [torch.from_numpy(mapped[i, start:end].copy()) for i in range(3)]
            g = gradient.flatten()[start:end] * clip
            first.lerp_(g, .1)
            second.mul_(.999).addcmul_(g, g, value=.001)
            denominator = second.sqrt().div_(math.sqrt(1 - .999**step)).add_(1e-8)
            weight.mul_(1-lr*.01).addcdiv_(first, denominator, value=-lr/(1-.9**step))
            mapped[:, start:end] = np.stack((weight.numpy(), first.numpy(), second.numpy()))
            self.fault('partial_state_write')
        mapped.flush()
        del mapped
        descriptor = os.open(path, os.O_RDWR)
        try:
            profile_fsync(descriptor)
        finally:
            os.close(descriptor)

    def finish_step(self):
        self.fault('before_final_manifest')
        self.manifest['step'] += 1
        self.manifest['dirty'] = False
        transaction=self.manifest.pop('transaction',None)
        durable_json(self.root / 'manifest.json', self.manifest)
        self.fault('final_manifest_durable')
        if transaction:
            (self.root/transaction['cursor']).unlink(missing_ok=True)
            (self.root/'transaction-input.pt').unlink(missing_ok=True)
            (self.root/'transaction-tied.pt').unlink(missing_ok=True)


class DiskDLS(StreamTrainEngine):
    def __init__(self, model, store, device='cpu', dtype=torch.float32, lr=.001, async_mode=False, fp8_transfer=False, cpu_prefetch=False):
        # Parent requires CPU parameters; meta definitions allocate no canonical model.
        # Build parent with a temporary metadata-only override of that check.
        self.store, self.lr = store, lr
        from .precision import DynamicLossScale
        self.scaler=DynamicLossScale()
        if store.manifest.get('loss_scaler'):self.scaler.load_state_dict(store.manifest['loss_scaler'])
        self.loss_scale=self.scaler.scale if dtype==torch.float16 else 1.
        self.model, self.device, self.dtype = model, torch.device(device), dtype
        self.async_mode = async_mode
        self.phase_seconds={}
        self.compute_events=[]
        for operation in ('step','resume_step'):
            method=getattr(self,operation)
            def measured_step(*args,_method=method,_operation=operation,**kwargs):
                token=_profile_store.set(self.store)
                wall_started=time.perf_counter();cpu_started=time.process_time()
                try:return _method(*args,**kwargs)
                finally:
                    self.timing(_operation+'_wall',time.perf_counter()-wall_started)
                    self.timing(_operation+'_cpu_process',time.process_time()-cpu_started)
                    _profile_store.reset(token)
            setattr(self,operation,measured_step)
        self.fp8_transfer=fp8_transfer
        self.cpu_prefetch=cpu_prefetch
        from .prefetch import SlotPrefetch
        self.prefetch=SlotPrefetch() if cpu_prefetch else None
        if fp8_transfer and dtype!=torch.float16:raise ValueError('FP8 temporary transport requires FP16 compute')
        if async_mode and self.device.type != 'cuda':
            raise ValueError('asynchronous disk fetch requires CUDA')
        from .engine import Stage, Slot
        self.stages=make_stages(model)
        self.specs = [list(stage.named_parameters()) for stage in self.stages]
        self.canonical = {id(p): name for name, p in model.named_parameters()}
        maximum = max(sum(p.numel() for _, p in spec) for spec in self.specs)
        self.slots = [Slot(maximum, self.device, dtype, async_mode) for _ in range(2)]
        self.h2d = torch.cuda.Stream(device=self.device) if async_mode else None
        self.d2h = torch.cuda.Stream(device=self.device) if async_mode else None
        self.compute = torch.cuda.current_stream(self.device) if self.device.type == 'cuda' else None
        self.parameter_version = store.manifest['step']
        self.transfer_bytes = dict(h2d_weights=0, d2h_gradients=0)
        # Buffers hold fixed graph topology and can be materialized independently.
        for stage in self.stages:
            for module in stage.modules():
                if hasattr(module, 'sources'):
                    capacity = module.sources.shape[1]
                    module.sources = torch.arange(capacity).view(1, -1, 1)
                    module.targets = torch.arange(capacity).roll(-1).view(1, -1)

    def profile_snapshot(self,previous=None):
        """Cumulative counters or deltas; overlapping wall/stream times are not additive."""
        self.collect_compute_timing()
        current=dict(seconds={**self.phase_seconds,**self.store.profile_seconds},
                     counts=dict(self.store.profile_counts),
                     bytes={**self.store.profile_bytes,**self.transfer_bytes})
        if previous is None:return current
        return {group:{key:value-previous.get(group,{}).get(key,0) for key,value in values.items()}
                for group,values in current.items()}

    def compute_start(self):
        started=time.perf_counter();event=None
        if self.device.type=='cuda':
            event=torch.cuda.Event(enable_timing=True);event.record(self.compute)
        return started,event

    def compute_end(self,phase,marker,expert=False):
        started,event=marker
        if event is not None:
            end=torch.cuda.Event(enable_timing=True);end.record(self.compute)
            self.compute_events.append((phase,event,end))
            if expert:self.compute_events.append(('expert_'+phase,event,end))
        elapsed=time.perf_counter()-started
        self.timing(phase+'_launch_wall',elapsed)
        if expert:self.timing('expert_'+phase+'_launch_wall',elapsed)

    def timing(self,name,elapsed):
        self.phase_seconds[name]=self.phase_seconds.get(name,0.)+elapsed

    def collect_compute_timing(self):
        if self.device.type=='cuda':
            torch.cuda.synchronize(self.device)
            for phase,start,end in self.compute_events:self.timing(phase+'_compute_stream_elapsed',start.elapsed_time(end)/1000)
            self.compute_events=[]
        return dict(self.phase_seconds)

    def load(self, index, slot):
        if self.prefetch is None:
            return self._load_sync(index,slot)
        self.prefetch.submit(slot,lambda:self._load_on_device(index,slot))

    def _load_on_device(self,index,slot):
        # CUDA device/stream context is thread-local, including allocations
        # inside FP8 packing and unpacking. Never rely on the worker default.
        with torch.cuda.device(self.device) if self.device.type=='cuda' else nullcontext():
            return self._load_sync(index,slot)

    def parameters(self,index,slot,grad):
        started=time.perf_counter()
        if self.prefetch is not None:self.prefetch.wait(slot)
        self.timing('prefetch_wait_wall',time.perf_counter()-started)
        return super().parameters(index,slot,grad)

    def close(self):
        try:
            if self.prefetch is not None:self.prefetch.close()
        finally:
            # Retain staging tensors through the last DMA completion.
            if self.async_mode:
                for slot in self.slots:
                    if slot.loaded is not None:slot.loaded.synchronize()
                    if slot.done is not None:slot.done.synchronize()

    def _load_sync(self, index, slot):
        read_started=time.perf_counter()
        if self.async_mode and slot.loaded is not None:
            slot.loaded.synchronize()
        offset = 0
        for _, parameter in self.specs[index]:
            size = parameter.numel()
            destination = slot.host[offset:offset+size]
            # BF16 numpy has no native representation; read an FP32 window then cast.
            if destination.dtype == torch.float32:
                self.store.read_into(self.canonical[id(parameter)], destination)
            else:
                work = torch.empty(size)
                self.store.read_into(self.canonical[id(parameter)], work)
                destination.copy_(work)
            offset += size
        self.timing('canonical_read_and_cast_wall',time.perf_counter()-read_started)
        transfer_started=time.perf_counter()
        # Retain pinned byte/scales buffers and device buffers through the H2D
        # completion event. The next load synchronizes it before reuse/deletion.
        slot.fp8_pending=[]
        with torch.cuda.stream(self.h2d) if self.async_mode else nullcontext():
            transfer_event=None
            if self.device.type=='cuda':
                transfer_event=torch.cuda.Event(enable_timing=True);transfer_event.record()
            if self.async_mode and slot.done is not None:
                self.h2d.wait_event(slot.done)
            from .precision import fp8_eligible,pack_fp8,unpack_fp8
            offset=0;transferred=0
            for name,parameter in self.specs[index]:
                size=parameter.numel();canonical=self.canonical[id(parameter)]
                if self.fp8_transfer and fp8_eligible(canonical):
                    pack_started=time.perf_counter()
                    encoded,scales=pack_fp8(slot.host[offset:offset+size])
                    self.timing('fp8_pack_cpu_wall',time.perf_counter()-pack_started)
                    if self.async_mode:encoded=encoded.pin_memory();scales=scales.pin_memory()
                    encoded_device=encoded.to(self.device,non_blocking=self.async_mode)
                    scales_device=scales.to(self.device,non_blocking=self.async_mode)
                    slot.weights[offset:offset+size].copy_(unpack_fp8(encoded_device,scales_device,self.dtype))
                    slot.fp8_pending.append((encoded,scales,encoded_device,scales_device))
                    transferred+=size+scales.numel()*4
                else:
                    slot.weights[offset:offset+size].copy_(slot.host[offset:offset+size],non_blocking=self.async_mode)
                    transferred+=size*slot.weights.element_size()
                offset+=size
            if transfer_event is not None:
                transfer_end=torch.cuda.Event(enable_timing=True);transfer_end.record()
                self.compute_events.append(('h2d_transfer',transfer_event,transfer_end))
            if self.async_mode:
                slot.loaded = torch.cuda.Event()
                slot.loaded.record(self.h2d)
        self.timing('transfer_prepare_and_schedule_wall',time.perf_counter()-transfer_started)
        slot.version = self.parameter_version
        self.transfer_bytes['h2d_weights'] += transferred

    def report(self,phase,index,micro=None):
        callback=getattr(self,'progress',None)
        if callback is not None:callback(phase,index,micro)

    def boundaries(self, batches):
        """Layer-major forward: one weight fetch per layer across all micros.

        Only one microbatch activation is on the compute device at a time.
        Current stages have no dropout/random forward operations; introducing
        those requires explicit per-stage/per-micro RNG replay state.
        """
        checkpoints=[[] for _ in batches];states=[]
        for batch in batches:
            tokens,proof_graph=unpack_batch(batch,'cpu')
            states.append((tokens if proof_graph is not None else tokens[:,:-1],None))
        self.load(0,self.slots[0])
        for index in range(len(self.stages)-1):
            slot=self.slots[index%2]
            if index+1<len(self.stages)-1:self.load(index+1,self.slots[(index+1)%2])
            params=self.parameters(index,slot,False)
            for micro,state in enumerate(states):
                self.report('forward',index,micro)
                checkpoints[micro].append(state)
                device_state=tuple(value.to(self.device) if value is not None else None for value in state)
                compute_started=time.perf_counter()
                if self.device.type=='cuda':
                    event_start=torch.cuda.Event(enable_timing=True);event_end=torch.cuda.Event(enable_timing=True);event_start.record()
                with torch.no_grad(),torch.autocast(self.device.type,dtype=self.dtype,enabled=self.dtype in (torch.bfloat16,torch.float16)):
                    result=functional_call(self.stages[index],params,device_state)
                if self.device.type=='cuda':event_end.record();self.compute_events.append(('forward',event_start,event_end))
                states[micro]=tuple(value.detach().cpu() if value is not None else None for value in result)
                self.timing('forward_compute_and_boundary_copy_wall',time.perf_counter()-compute_started)
                del result,device_state
            self.release(slot)
        for micro,state in enumerate(states):checkpoints[micro].append(state)
        return checkpoints

    def reverse(self, batches, checkpoints, update=False, clip=1.,initial_cursor=False):
        cotangents = [None] * len(batches)
        tied = None
        start_stage=len(self.stages)-1
        if update and self.store.manifest.get('transaction') and not initial_cursor:
            self.store.recover_stage()
            transaction=self.store.manifest['transaction']
            cursor=self.store.load_cursor()
            cotangents=[None if values is None else [v.to(self.device) for v in values] for values in cursor['cotangents']]
            tied=cursor['tied']
            start_stage=transaction['next_stage']
        norm_squared, total_loss = 0., 0.
        embedding_name = self.canonical[id(self.model.embedding.weight)]
        step = self.store.manifest['step'] + 1
        if start_stage<0:return 0.,0.
        self.load(start_stage, self.slots[start_stage % 2])
        for index in reversed(range(start_stage+1)):
            self.report('update_reverse' if update else 'norm_reverse',index)
            slot = self.slots[index % 2]
            if index:
                self.load(index-1, self.slots[(index-1) % 2])
            accumulated = [torch.zeros(p.shape) for _, p in self.specs[index]]
            for micro, batch in enumerate(batches):
                _,proof_graph=unpack_batch(batch,self.device)
                params = self.parameters(index, slot, True)
                state = tuple(v.to(self.device).detach().requires_grad_(bool(index)) if v is not None else None
                              for v in checkpoints[micro][index])
                state_values = [v for v in state if v is not None] if index else []
                weights = [params[name] for name, _ in self.specs[index]]
                values = state_values + weights
                phase='update_reverse' if update else 'norm_reverse'
                compute_started=time.perf_counter()
                if self.device.type=='cuda':
                    event_start=torch.cuda.Event(enable_timing=True);event_end=torch.cuda.Event(enable_timing=True);event_start.record()
                with torch.enable_grad(), torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype in (torch.bfloat16,torch.float16)):
                    output = functional_call(self.stages[index], params, (*state,proof_graph))
                    if cotangents[micro] is None:
                        loss = stage_loss(output,batch,self.device)
                        total_loss += loss.detach().item() / len(batches)
                        grads = torch.autograd.grad(loss * self.loss_scale / len(batches), values, allow_unused=True)
                    else:
                        grads = torch.autograd.grad([v for v in output if v is not None], values,
                                                    cotangents[micro], allow_unused=True)
                if self.device.type=='cuda':event_end.record();self.compute_events.append((phase,event_start,event_end))
                self.timing(phase+'_launch_wall',time.perf_counter()-compute_started)
                copy_started=time.perf_counter()
                grads = [torch.zeros_like(v) if g is None else g for v, g in zip(values, grads)]
                cotangents[micro] = grads[:len(state_values)] if index else None
                for destination, gradient in zip(accumulated, grads[len(state_values):]):
                    destination.add_(gradient.detach().float().cpu()/self.loss_scale)
                    self.transfer_bytes['d2h_gradients'] += gradient.numel() * 4
                self.timing('gradient_d2h_and_accumulation_wall',time.perf_counter()-copy_started)
                del output, params, weights, values, state_values, grads
            if update:
                names=[self.canonical[id(parameter)] for _,parameter in self.specs[index]
                       if index==0 or self.canonical[id(parameter)]!=embedding_name]
                self.store.prepare_stage(index,names)
            for (_, parameter), gradient in zip(self.specs[index], accumulated):
                name = self.canonical[id(parameter)]
                if name == embedding_name and index != 0:
                    tied = gradient
                    continue
                if name == embedding_name and tied is not None:
                    gradient.add_(tied)
                    tied = None
                if update:
                    self.store.update(name, gradient, step, self.lr, clip)
                else:
                    # Chunk reduction avoids a parameter-sized squared temporary.
                    for piece in gradient.flatten().split(1048576):
                        norm_squared += piece.double().square().sum().item()
            self.release(slot)
            if update:
                self.store.commit_stage(index,cotangents,tied)
            del accumulated
        return math.sqrt(norm_squared), total_loss

    def step(self, batches, max_norm=1.):
        if self.store.manifest['dirty'] or self.store.manifest.get('transaction'):
            raise RuntimeError('pending transaction: call resume_step before starting another step')
        if not batches or max_norm <= 0:
            raise ValueError('nonempty microbatches and positive clipping threshold required')
        self.report('forward_start',0)
        checkpoints = self.boundaries(batches)
        while True:
            self.loss_scale=self.scaler.scale if self.dtype==torch.float16 else 1.
            norm, loss = self.reverse(batches, checkpoints)
            if math.isfinite(norm) and math.isfinite(loss):break
            if self.dtype!=torch.float16 or not math.isfinite(loss):
                raise FloatingPointError('nonfinite gradients/loss; disk state unchanged')
            self.scaler.overflow()
        clip = min(1., max_norm / (norm + 1e-6))
        durable_torch(self.store.root/'transaction-input.pt',dict(batches=batches,checkpoints=checkpoints,clip=clip,loss=loss,norm=norm,loss_scale=self.loss_scale,dtype=str(self.dtype),fp8_transfer=self.fp8_transfer))
        durable_torch(self.store.root/'cursor-initial.pt',dict(cotangents=[None]*len(batches),tied=None))
        self.store.manifest.update(dirty=True,transaction=dict(next_stage=len(self.stages)-1,cursor='cursor-initial.pt',journal=None))
        durable_json(self.store.root/'manifest.json',self.store.manifest)
        self.reverse(batches,checkpoints,update=True,clip=clip,initial_cursor=True)
        if self.dtype==torch.float16:
            self.scaler.success();self.store.manifest['loss_scaler']=self.scaler.state_dict()
        self.store.finish_step()
        self.parameter_version=self.store.manifest['step']
        return dict(loss=loss, grad_norm=norm)


    def resume_step(self):
        if not self.store.manifest.get('transaction'):
            raise ValueError('no recoverable transaction')
        saved=torch.load(self.store.root/'transaction-input.pt',weights_only=True,map_location='cpu')
        if saved.get('dtype',str(self.dtype))!=str(self.dtype) or saved.get('fp8_transfer',False)!=self.fp8_transfer:
            raise ValueError('transaction precision/FP8 transport differs; resume original settings')
        self.loss_scale=saved.get('loss_scale',1.)
        if self.dtype==torch.float16:self.scaler.scale=self.loss_scale
        self.reverse(saved['batches'],saved['checkpoints'],update=True,clip=saved['clip'])
        if self.dtype==torch.float16:
            self.scaler.success();self.store.manifest['loss_scaler']=self.scaler.state_dict()
        self.store.finish_step()
        self.parameter_version=self.store.manifest['step']
        return dict(loss=saved['loss'],grad_norm=saved['norm'])


def estimate_disk_budget(model, batch_size, length, accumulation, dtype):
    """Conservative allocation envelope, not a measured peak-memory guarantee."""
    from .engine import Stage
    cfg = model.config
    stages=make_stages(model)
    stage_size = max(sum(p.numel() for p in stage.parameters()) for stage in stages)
    largest = max(p.numel() for p in model.parameters())
    count = sum(p.numel() for p in model.parameters())
    element = 2 if dtype in (torch.bfloat16,torch.float16) else 4
    graph = cfg.graph_nodes * cfg.d_graph if cfg.graph_enabled else 0
    boundaries = accumulation * (cfg.num_layers+2) * batch_size * length * (cfg.d_model+graph) * 4
    host = (2*stage_size*(element+4) + stage_size*4 + model.embedding.weight.numel()*4
            + largest*16 + boundaries + 2*2**30 + 64*2**20)
    ssm = batch_size*length*cfg.d_model*cfg.expand*cfg.d_state*4*8
    graph_work = batch_size*length*graph*cfg.reasoning_steps*4*16
    logits = batch_size*length*cfg.vocab_size*4*4
    cotangents = accumulation * batch_size * length * (cfg.d_model+graph) * 4 * 2
    cuda_reserve = 2*2**30 if stage_size >= 50_000_000 else 256*2**20+4*stage_size
    proof_work = batch_size*cfg.graph_nodes*cfg.statement_field_length*3*cfg.d_model*4*8 if cfg.proof_enabled else 0
    vram = proof_work + cotangents + 2*stage_size*(element+4) + stage_size*element + ssm + graph_work + logits + cuda_reserve
    return dict(parameters=count, state_disk_bytes=12*count, disk_reserve_bytes=2*2**30 +12*stage_size+3*boundaries+2*model.embedding.weight.numel()*4,
                estimated_host_bytes=host, estimated_cuda_bytes=vram,
                boundary_bytes=boundaries, largest_stage_parameters=stage_size)


def disk_train(cfg, tokens, root, steps, device='cpu', dtype=torch.float32, lr=.001,
               batch_size=1, sequence_length=8, accumulation=1, seed=42, resume=False, progress=None):
    import hashlib
    import time
    if cfg.proof_enabled:
        raise ValueError('use structured proof training entry point for joint proof batches')
    if steps < 0 or accumulation < 1 or batch_size < 1 or sequence_length < 1 or lr <= 0:
        raise ValueError('invalid disk training schedule')
    mapped = hasattr(tokens,'sample') and hasattr(tokens,'manifest')
    if mapped:
        if cfg.vocab_size < 256:
            raise ValueError('mapped byte shard needs vocabulary256')
        fingerprint=hashlib.sha256(json.dumps(tokens.manifest,sort_keys=True).encode()).hexdigest()
        shape=['mapped',len(tokens.records)]
    else:
        tokens = tokens.cpu().long()
        if tokens.ndim != 2 or tokens.shape[1] < sequence_length+1:
            raise ValueError('insufficient token data')
        if tokens.min() < 0 or tokens.max() >= cfg.vocab_size:
            raise ValueError('token IDs exceed vocabulary')
        fingerprint=hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
        shape=list(tokens.shape)
    with torch.device('meta'):
        from morph import MorphModel
        model = MorphModel(cfg)
    settings = dict(fingerprint=fingerprint,
                    shape=shape, lr=lr, seed=seed, device=torch.device(device).type,
                    dtype=str(dtype), batch_size=batch_size, sequence_length=sequence_length,
                    accumulation=accumulation)
    store = DiskState.open(root, model) if resume else DiskState.create(root, model, seed=seed)
    if resume and store.manifest.get('settings') != settings:
        raise ValueError('disk resume data or settings differ')
    if store.manifest['step'] > steps:
        raise ValueError('requested total steps precedes stored step')
    store.manifest['settings'] = settings
    durable_json(store.root / 'manifest.json', store.manifest)
    trainer = DiskDLS(model, store, device=device, dtype=dtype, lr=lr, async_mode=device.startswith('cuda'))
    start_step = store.manifest['step']
    if store.manifest.get('transaction'):
        trainer.resume_step()
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    final = None
    for step in range(store.manifest['step'], steps):
        # Stateless step seed makes completed-step resume deterministic without
        # storing huge activation checkpoints or a mutable sampler state.
        generator = torch.Generator().manual_seed(seed + step)
        batches = []
        for _ in range(accumulation):
            if mapped:
                batches.append(tokens.sample(generator,batch_size,sequence_length+1))
            else:
                rows = torch.randint(tokens.shape[0], (batch_size,), generator=generator)
                starts = torch.randint(tokens.shape[1]-sequence_length, (batch_size,), generator=generator)
                batches.append(torch.stack([tokens[r, s:s+sequence_length+1] for r, s in zip(rows, starts)]))
        final = trainer.step(batches)
        if progress is not None:
            progress(step+1, final['loss'])
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    import resource
    return dict(step=store.manifest['step'], final=final, state_dir=str(root),
                elapsed_seconds=elapsed, tokens_per_second=(steps-start_step)*batch_size*sequence_length*accumulation/max(elapsed,1e-9),
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else 0,
                transfer_bytes=trainer.transfer_bytes)
