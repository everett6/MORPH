"""Train native MORPH on local UTF-8 text or integer-token JSON data."""
import argparse
import json
from pathlib import Path
import torch
from morph import MorphConfig, MorphModel
from morph.training.runner import train_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof-traces', type=Path, help='structured teacher-forced proof JSONL; resident debug training')
    parser.add_argument('--config', default='configs/morph_debug.json')
    parser.add_argument('--engine', choices=['resident', 'dls', 'async', 'disk'], default='resident')
    parser.add_argument('--state-dir', type=Path, help='disk engine canonical weights/moments and completed-step state')
    parser.add_argument('--resume-store', action='store_true')
    parser.add_argument('--optimizer', choices=['hybrid', 'adamw'], default=None)
    parser.add_argument('--muon-lr', type=float)
    parser.add_argument('--inspect', action='store_true', help='count on meta device, allocate no weights')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--shard',type=Path,help='memory-mapped byte-token shard directory')
    source.add_argument('--text', type=Path, help='local UTF-8 text; deterministic byte IDs 0..255')
    source.add_argument('--tokens', type=Path, help='JSON list of token IDs or rectangular nested list')
    parser.add_argument('--steps', type=int, default=100, help='total optimizer steps including resumed steps')
    parser.add_argument('--sequence-length', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument('--activation-checkpointing', action='store_true')
    parser.add_argument('--checkpoint-every', type=int, default=100)
    parser.add_argument('--log-every', type=int, default=1)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.optimizer is None:args.optimizer='adamw' if args.proof_traces else 'hybrid'
    torch.set_num_threads(args.threads)
    cfg = MorphConfig.from_file(args.config)
    if args.proof_traces:
        if args.optimizer!='adamw':parser.error('structured proof mode supports AdamW only')
        if args.muon_lr is not None or args.activation_checkpointing:
            parser.error('proof mode currently uses AdamW and stage replay; Muon/checkpoint flags are unsupported')
        from dataclasses import replace
        proof_cfg=replace(cfg,proof_enabled=True)
        with torch.device('meta'):
            proof_probe=MorphModel(proof_cfg)
        proof_count=sum(p.numel() for p in proof_probe.parameters())
        from streamtrain.disk import estimate_disk_budget
        import shutil
        budget=estimate_disk_budget(proof_probe,args.batch_size,args.sequence_length,args.accumulation,
                                    torch.bfloat16 if args.precision=='bf16' else torch.float32)
        # JSON parsing, saved checkpoint loading, and proof-field activations need
        # their own host envelope even when canonical weights live elsewhere.
        data_bytes=args.proof_traces.stat().st_size*40 if args.proof_traces.exists() else 0
        resume_bytes=proof_count*16 if args.resume else 0
        field_bytes=args.batch_size*proof_cfg.graph_nodes*3*proof_cfg.statement_field_length*proof_cfg.d_model*4*8
        host_available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
        if args.engine=='resident':
            host_need=(proof_count*24+field_bytes+512*2**20 if not args.device.startswith('cuda') else 512*2**20)+data_bytes+resume_bytes
            cuda_need=proof_count*24+field_bytes+budget['estimated_cuda_bytes']
        else:
            host_need=budget['estimated_host_bytes']+data_bytes+resume_bytes+(proof_count*16 if args.engine!='disk' else 0)
            if not args.device.startswith('cuda'):host_need+=budget['estimated_cuda_bytes']
            cuda_need=budget['estimated_cuda_bytes']
        qualification=dict(estimated_host_bytes=host_need,estimated_cuda_bytes=cuda_need,disk_budget=budget)
        if args.inspect:
            print(json.dumps(dict(parameters=proof_count,proof_enabled=True,qualification=qualification)))
            return
        if host_need>host_available*.75:
            parser.error('structured proof host-memory estimate exceeds available RAM with headroom; use smaller config/batch/sequence')
        if args.device.startswith('cuda'):
            if not torch.cuda.is_available():parser.error('CUDA unavailable')
            if cuda_need>torch.cuda.mem_get_info(args.device)[0]*.8:
                parser.error('structured proof GPU-memory estimate exceeds free VRAM with headroom')
        if args.engine=='disk':
            if args.checkpoint or args.resume:parser.error('disk proof state is its checkpoint; use --resume-store')
            if args.state_dir is None:parser.error('disk proof mode requires --state-dir')
            parent=args.state_dir.parent
            while not parent.exists():parent=parent.parent
            needed=budget['disk_reserve_bytes']+(0 if args.resume_store else budget['state_disk_bytes'])
            if needed>shutil.disk_usage(parent).free:
                parser.error('insufficient disk for proof state and recovery journal, including resume reserve')
        from morph.training.proof_runner import run_proof_training
        _, losses = run_proof_training(cfg,args.proof_traces,args.steps,args.lr,args.device,args.checkpoint,sequence_length=args.sequence_length,engine=args.engine,
                                         batch_size=args.batch_size,accumulation=args.accumulation,
                                         dtype=torch.bfloat16 if args.precision=='bf16' else torch.float32,state_dir=args.state_dir,resume=bool(args.resume or args.resume_store),checkpoint_every=args.checkpoint_every,resume_path=args.resume,seed=args.seed)
        print(json.dumps(dict(structured_steps=len(losses),losses=losses)))
        return
    with torch.device('meta'):
        probe = MorphModel(cfg)
    count = sum(p.numel() for p in probe.parameters())
    # FP32 parameters, gradients, and both Adam moments; excludes activations/workspaces.
    from streamtrain.optimizer import partition
    muon, adam = partition(probe)
    muon_count = sum(p.numel() for p in muon)
    persistent = count * 16 if args.optimizer == 'adamw' else muon_count * 12 + (count - muon_count) * 16
    disk_budget = None
    if args.engine == 'disk':
        from streamtrain.disk import estimate_disk_budget
        disk_budget = estimate_disk_budget(probe, args.batch_size, args.sequence_length, args.accumulation,
                                            torch.bfloat16 if args.precision == 'bf16' else torch.float32)
        args.optimizer = 'adamw'
    print(json.dumps(dict(parameters=count, trainable_parameters=count,
                          persistent_optimizer_GiB=None if args.engine == 'disk' else persistent / 2**30, muon_parameters=0 if args.engine == 'disk' else muon_count,
                          optimizer=args.optimizer, engine=args.engine,
                          compute=args.precision, device=args.device, disk_budget=disk_budget)))
    if args.inspect:
        return
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            parser.error('CUDA is unavailable')
        available, _ = torch.cuda.mem_get_info(torch.device(args.device))
    else:
        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:'))) * 1024
    if args.engine != 'resident':
        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:'))) * 1024
    if args.engine == 'disk':
        import shutil
        if args.state_dir is None:
            parser.error('--engine disk requires --state-dir')
        parent = args.state_dir
        while not parent.exists():
            parent = parent.parent
        free_disk = shutil.disk_usage(parent).free
        needed_disk = disk_budget['disk_reserve_bytes'] + (0 if args.resume_store else disk_budget['state_disk_bytes'])
        if free_disk < needed_disk:
            parser.error(f'disk engine requires {needed_disk/2**30:.1f} GiB free; only {free_disk/2**30:.1f} GiB available')
        if disk_budget['estimated_host_bytes'] > available * .85:
            parser.error('disk host-memory estimate exceeds available RAM with headroom; reduce accumulation/sequence length')
        if args.device.startswith('cuda'):
            free_vram, _ = torch.cuda.mem_get_info(torch.device(args.device))
            if disk_budget['estimated_cuda_bytes'] > free_vram * .85:
                parser.error('disk VRAM estimate exceeds available GPU memory with headroom; reduce sequence length')
        if args.checkpoint or args.resume:
            parser.error('disk state is its checkpoint; use --state-dir and --resume-store')
    elif persistent > available * 0.75:
        parser.error(f'Optimizer persistent tensors require {persistent / 2**30:.1f} GiB before activations; '
                     f'only {available / 2**30:.1f} GiB currently available. Use debug config or larger hardware.')
    if args.engine == 'disk' and (args.text or args.tokens):
        corpus = args.text or args.tokens
        parse_envelope = corpus.stat().st_size * (40 if args.text else 12)
        if disk_budget['estimated_host_bytes'] + parse_envelope > available * .85:
            parser.error('corpus parsing plus disk windows exceeds RAM estimate; use a smaller local corpus shard')
    if args.shard:
        from data import MappedShard
        tokens=MappedShard(args.shard,'train')
    elif args.text:
        if cfg.vocab_size < 256:
            parser.error('byte encoding needs vocab_size >= 256')
        tokens = torch.tensor(list(args.text.read_bytes()), dtype=torch.long)[None]
    elif args.tokens:
        raw = json.loads(args.tokens.read_text())
        tokens = torch.tensor(raw)
        if tokens.dtype not in (torch.int64, torch.int32):
            parser.error('token JSON must contain integers')
        if tokens.ndim == 1:
            tokens = tokens[None]
    else:
        parser.error('provide --text or --tokens; no data is downloaded')
    if args.engine == 'disk':
        from streamtrain.disk import disk_train
        result = disk_train(cfg, tokens, args.state_dir, args.steps, args.device,
                            torch.bfloat16 if args.precision == 'bf16' else torch.float32, args.lr,
                            args.batch_size, args.sequence_length, args.accumulation, args.seed, args.resume_store,
                            lambda step, loss: print(json.dumps(dict(step=step, loss=loss)), flush=True)
                            if args.log_every > 0 and step % args.log_every == 0 else None)
        print(json.dumps(result))
        return
    result = train_run(cfg, tokens, args.steps, args.seed, args.lr, args.checkpoint,
                       args.resume, args.device,
                       torch.bfloat16 if args.precision == 'bf16' else torch.float32,
                       args.activation_checkpointing, args.batch_size, args.sequence_length,
                       args.accumulation, args.engine, args.optimizer, args.muon_lr,
                       args.checkpoint_every,
                       lambda step, loss: print(json.dumps(dict(step=step, loss=loss)), flush=True)
                       if args.log_every > 0 and step % args.log_every == 0 else None)
    print(json.dumps(dict(steps=len(result['losses']), initial_loss=result['losses'][0] if result['losses'] else None,
                          final_loss=result['losses'][-1] if result['losses'] else None,
                          checkpoint=str(args.checkpoint) if args.checkpoint else None, metrics=result['metrics'])))


if __name__ == '__main__':
    main()
