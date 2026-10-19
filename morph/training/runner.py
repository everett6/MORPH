"""Resident AdamW training, deterministic resumption, and atomic checkpoints."""
import os
from pathlib import Path
import hashlib
import time
import torch
from morph import MorphModel


def train_run(cfg, tokens, steps, seed=42, lr=1e-3, checkpoint=None, resume=None,
              device='cpu', dtype=torch.float32, activation_checkpointing=False,
              batch_size=1, sequence_length=None, accumulation=1, engine='resident',
              optimizer_name='hybrid', muon_lr=None, checkpoint_every=0, progress=None):
    if steps < 0 or batch_size < 1 or accumulation < 1 or lr <= 0 or checkpoint_every < 0:
        raise ValueError('invalid training schedule')
    tokens = tokens.detach().cpu().long()
    if tokens.ndim != 2 or tokens.numel() == 0 or tokens.shape[1] < 2:
        raise ValueError('tokens must have shape [examples, sequence >= 2]')
    if tokens.min() < 0 or tokens.max() >= cfg.vocab_size:
        raise ValueError('token IDs exceed configured vocabulary')
    length = tokens.shape[1] if sequence_length is None else sequence_length + 1
    if length < 2 or length > tokens.shape[1]:
        raise ValueError('sequence length exceeds available data')
    fingerprint = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
    settings = dict(lr=lr, batch_size=batch_size, length=length, accumulation=accumulation,
                    dtype=str(dtype), device_type=torch.device(device).type, engine=engine, optimizer=optimizer_name, muon_lr=muon_lr, fingerprint=fingerprint, token_shape=tuple(tokens.shape))
    torch.manual_seed(seed)
    if engine not in ('resident', 'dls', 'async') or optimizer_name not in ('hybrid', 'adamw'):
        raise ValueError('unsupported engine or optimizer')
    model = MorphModel(cfg, activation_checkpointing).to(device=device if engine == 'resident' else 'cpu')
    # Keep FP32 master parameters and optimizer states with autocast BF16 compute.
    model = model.float()
    from streamtrain import StreamTrainEngine, HybridMuonAdamW
    optimizer = (HybridMuonAdamW(model, lr=lr, muon_lr=muon_lr) if optimizer_name == 'hybrid'
                 else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, foreach=False))
    streaming = None if engine == 'resident' else StreamTrainEngine(model, device, engine == 'async', dtype)
    losses = []
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        if saved['config'] != cfg.to_dict() or saved['settings'] != settings:
            raise ValueError('resume configuration, data, or training settings differ')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        losses = saved['losses']
        if streaming is not None:
            streaming.parameter_version = saved['parameter_version']
        torch.set_rng_state(saved['rng'])
        if device.startswith('cuda'):
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    if steps < len(losses):
        raise ValueError('steps is the total target; checkpoint is already beyond it')
    def save_checkpoint():
        if checkpoint is None:
            return
        path = Path(checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + '.tmp')
        torch.save(dict(config=cfg.to_dict(), settings=settings, model=model.state_dict(),
                        optimizer=optimizer.state_dict(), losses=losses, rng=torch.get_rng_state(),
                        parameter_version=len(losses),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else []), temp)
        os.replace(temp, path)
    initial_steps = len(losses)
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(len(losses), steps):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accumulation):
            rows = torch.randint(tokens.shape[0], (batch_size,))
            starts = torch.randint(tokens.shape[1] - length + 1, (batch_size,))
            batch = torch.stack([tokens[r, s:s + length] for r, s in zip(rows, starts)]).to(device)
            if streaming is not None:
                loss_value = streaming.backward(batch, 1 / accumulation)
            else:
                with torch.autocast(torch.device(device).type, dtype=dtype, enabled=dtype == torch.bfloat16):
                    loss = model.loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite loss; no optimizer update applied')
                (loss / accumulation).backward()
                loss_value = loss.detach().item()
            total += loss_value / accumulation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if streaming is not None:
            streaming.parameter_version += 1
        losses.append(total)
        if progress is not None:
            progress(len(losses), total)
        if checkpoint_every and len(losses) % checkpoint_every == 0:
            save_checkpoint()
    if checkpoint:
        save_checkpoint()
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    metrics = dict(elapsed_seconds=elapsed,
                   tokens_per_second=(len(losses) - initial_steps) * batch_size * (length - 1) * accumulation / max(elapsed, 1e-9),
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else 0,
                   transfer_bytes=streaming.transfer_bytes if streaming else {})
    return dict(model=model, losses=losses, streaming=streaming, metrics=metrics)
