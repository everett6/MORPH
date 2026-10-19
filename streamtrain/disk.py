"""Bounded-memory two-pass disk AdamW. No optimizer approximation.

Canonical FP32 weight/m/v files; gradients exist for one stage at a time.
Incomplete updates fail closed. Completed steps are durable resume points.
"""
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch.func import functional_call
from torch.nn import functional as F
from .engine import StreamTrainEngine


def durable_json(path, data):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as handle:
        json.dump(data, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class DiskState:
    def __init__(self, root, model, manifest):
        self.root, self.manifest = Path(root), manifest
        self.entries = {name: dict(shape=list(p.shape), size=p.numel(), file=f'{i:05}.bin')
                        for i, (name, p) in enumerate(model.named_parameters())}
        if self.entries != manifest['entries']:
            raise ValueError('disk state tensor schema differs')

    @classmethod
    def create(cls, root, model, source=None, seed=42, chunk_elements=1048576):
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
        manifest['dirty'] = False
        durable_json(root / 'manifest.json', manifest)
        return cls(root, model, manifest)

    @classmethod
    def open(cls, root, model):
        root = Path(root)
        manifest = json.loads((root / 'manifest.json').read_text())
        if manifest['dirty']:
            raise RuntimeError('incomplete disk update: state is not a valid resume point; restore a backup')
        if manifest['config'] != model.config.to_dict():
            raise ValueError('disk model config differs')
        return cls(root, model, manifest)

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
        mapped.flush()
        del mapped
        descriptor = os.open(path, os.O_RDWR)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def finish_step(self):
        self.manifest['step'] += 1
        self.manifest['dirty'] = False
        durable_json(self.root / 'manifest.json', self.manifest)


class DiskDLS(StreamTrainEngine):
    def __init__(self, model, store, device='cpu', dtype=torch.float32, lr=.001, async_mode=False):
        # Parent requires CPU parameters; meta definitions allocate no canonical model.
        # Build parent with a temporary metadata-only override of that check.
        self.store, self.lr = store, lr
        self.model, self.device, self.dtype = model, torch.device(device), dtype
        self.async_mode = async_mode
        if async_mode and self.device.type != 'cuda':
            raise ValueError('asynchronous disk fetch requires CUDA')
        from .engine import Stage, Slot
        self.stages = [Stage(model.embedding, 'embedding')]
        self.stages += [Stage(block, 'block') for block in model.blocks]
        self.stages += [Stage(model.norm, 'head', model.embedding)]
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

    def load(self, index, slot):
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
        with torch.cuda.stream(self.h2d) if self.async_mode else nullcontext():
            if self.async_mode and slot.done is not None:
                self.h2d.wait_event(slot.done)
            slot.weights[:offset].copy_(slot.host[:offset], non_blocking=self.async_mode)
            if self.async_mode:
                slot.loaded = torch.cuda.Event()
                slot.loaded.record(self.h2d)
        slot.version = self.parameter_version
        self.transfer_bytes['h2d_weights'] += offset * slot.weights.element_size()

    def boundaries(self, batches):
        checkpoints = []
        for batch in batches:
            state = (batch[:, :-1].to(self.device), None)
            saved = []
            self.load(0, self.slots[0])
            for index in range(len(self.stages)-1):
                saved.append(tuple(v.detach().cpu() if v is not None else None for v in state))
                slot = self.slots[index % 2]
                if index + 1 < len(self.stages)-1:
                    self.load(index+1, self.slots[(index+1) % 2])
                params = self.parameters(index, slot, False)
                with torch.no_grad(), torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype == torch.bfloat16):
                    state = functional_call(self.stages[index], params, state)
                self.release(slot)
            saved.append(tuple(v.detach().cpu() if v is not None else None for v in state))
            checkpoints.append(saved)
        return checkpoints

    def reverse(self, batches, checkpoints, update=False, clip=1.):
        cotangents = [None] * len(batches)
        tied = None
        norm_squared, total_loss = 0., 0.
        embedding_name = self.canonical[id(self.model.embedding.weight)]
        step = self.store.manifest['step'] + 1
        self.load(len(self.stages)-1, self.slots[(len(self.stages)-1) % 2])
        for index in reversed(range(len(self.stages))):
            slot = self.slots[index % 2]
            if index:
                self.load(index-1, self.slots[(index-1) % 2])
            accumulated = [torch.zeros(p.shape) for _, p in self.specs[index]]
            for micro, batch in enumerate(batches):
                params = self.parameters(index, slot, True)
                state = tuple(v.to(self.device).detach().requires_grad_(bool(index)) if v is not None else None
                              for v in checkpoints[micro][index])
                state_values = [v for v in state if v is not None] if index else []
                weights = [params[name] for name, _ in self.specs[index]]
                values = state_values + weights
                with torch.enable_grad(), torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype == torch.bfloat16):
                    output = functional_call(self.stages[index], params, state)
                    if cotangents[micro] is None:
                        loss = F.cross_entropy(output.float().flatten(0, 1), batch[:, 1:].to(self.device).flatten())
                        total_loss += loss.detach().item() / len(batches)
                        grads = torch.autograd.grad(loss / len(batches), values, allow_unused=True)
                    else:
                        grads = torch.autograd.grad([v for v in output if v is not None], values,
                                                    cotangents[micro], allow_unused=True)
                grads = [torch.zeros_like(v) if g is None else g for v, g in zip(values, grads)]
                cotangents[micro] = grads[:len(state_values)] if index else None
                for destination, gradient in zip(accumulated, grads[len(state_values):]):
                    destination.add_(gradient.detach().float().cpu())
                    self.transfer_bytes['d2h_gradients'] += gradient.numel() * 4
                del output, params, weights, values, state_values, grads
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
            del accumulated
        return math.sqrt(norm_squared), total_loss

    def step(self, batches, max_norm=1.):
        if not batches or max_norm <= 0:
            raise ValueError('nonempty microbatches and positive clipping threshold required')
        checkpoints = self.boundaries(batches)
        norm, loss = self.reverse(batches, checkpoints)
        if not math.isfinite(norm) or not math.isfinite(loss):
            raise FloatingPointError('nonfinite gradients; disk state unchanged')
        clip = min(1., max_norm / (norm + 1e-6))
        self.store.begin_step()
        self.reverse(batches, checkpoints, update=True, clip=clip)
        self.store.finish_step()
        self.parameter_version += 1
        return dict(loss=loss, grad_norm=norm)


def estimate_disk_budget(model, batch_size, length, accumulation, dtype):
    """Conservative allocation envelope, not a measured peak-memory guarantee."""
    from .engine import Stage
    cfg = model.config
    stages = [Stage(model.embedding, 'embedding')]
    stages += [Stage(block, 'block') for block in model.blocks]
    stages += [Stage(model.norm, 'head', model.embedding)]
    stage_size = max(sum(p.numel() for p in stage.parameters()) for stage in stages)
    largest = max(p.numel() for p in model.parameters())
    count = sum(p.numel() for p in model.parameters())
    element = 2 if dtype == torch.bfloat16 else 4
    graph = cfg.graph_nodes * cfg.d_graph if cfg.graph_enabled else 0
    boundaries = accumulation * (cfg.num_layers+2) * batch_size * length * (cfg.d_model+graph) * 4
    host = (2*stage_size*(element+4) + stage_size*4 + model.embedding.weight.numel()*4
            + largest*16 + boundaries + 2*2**30 + 64*2**20)
    ssm = batch_size*length*cfg.d_model*cfg.expand*cfg.d_state*4*8
    graph_work = batch_size*length*graph*cfg.reasoning_steps*4*16
    logits = batch_size*length*cfg.vocab_size*4*4
    cotangents = accumulation * batch_size * length * (cfg.d_model+graph) * 4 * 2
    cuda_reserve = 2*2**30 if count >= 100_000_000 else 256*2**20
    vram = cotangents + 2*stage_size*(element+4) + stage_size*element + ssm + graph_work + logits + cuda_reserve
    return dict(parameters=count, state_disk_bytes=12*count, disk_reserve_bytes=2*2**30,
                estimated_host_bytes=host, estimated_cuda_bytes=vram,
                boundary_bytes=boundaries, largest_stage_parameters=stage_size)


def disk_train(cfg, tokens, root, steps, device='cpu', dtype=torch.float32, lr=.001,
               batch_size=1, sequence_length=8, accumulation=1, seed=42, resume=False, progress=None):
    import hashlib
    import time
    if steps < 0 or accumulation < 1 or batch_size < 1 or sequence_length < 1 or lr <= 0:
        raise ValueError('invalid disk training schedule')
    tokens = tokens.cpu().long()
    if tokens.ndim != 2 or tokens.shape[1] < sequence_length+1:
        raise ValueError('insufficient token data')
    if tokens.min() < 0 or tokens.max() >= cfg.vocab_size:
        raise ValueError('token IDs exceed vocabulary')
    with torch.device('meta'):
        from morph import MorphModel
        model = MorphModel(cfg)
    settings = dict(fingerprint=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
                    shape=list(tokens.shape), lr=lr, seed=seed, device=torch.device(device).type,
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
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    final = None
    for step in range(start_step, steps):
        # Stateless step seed makes completed-step resume deterministic without
        # storing huge activation checkpoints or a mutable sampler state.
        generator = torch.Generator().manual_seed(seed + step)
        batches = []
        for _ in range(accumulation):
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
