# Two-pass disk-backed AdamW

This mode keeps AdamW's update rule and moves its capacity cost to disk. It is
a custom MORPH execution schedule using known offloading/recomputation ideas,
not a claim of a new optimizer or throughput parity with resident AdamW.

For 6,995,110,080 parameters, canonical FP32 weights and both FP32 moments take
**83,941,320,960 bytes (78.18 GiB)** of disk. There is no full-model gradient
file, no full-model CPU tensor allocation, and no duplicate full checkpoint.
The initial files are physically allocated. Current development disk has about
102 GiB free; the CLI also reserves bounded transaction state and 2 GiB headroom. Leave additional space for your
corpus, other applications, and any independently retained backup.

## Why two backward passes

1. Save token and graph input boundaries for every layer and microbatch, with
   all model weights unchanged.
2. Reverse all stages, aggregating every microbatch's gradient for each stage.
   Sum the output/input uses of tied embeddings before counting that parameter
   once in the global gradient norm. Discard other gradients after reduction.
3. Compute the single global clipping multiplier.
4. Reverse the same saved boundaries again. Once a stage has produced all
   microbatch derivatives, update that stage's FP32 disk weight/m/v with AdamW.
   No later reverse stage reads those updated weights. Delay the tied embedding
   update until the input-embedding stage combines its gradient with the saved
   output-head contribution.

This schedule computes the same AdamW update as end-to-end training for the
current deterministic model, subject to floating-point reduction differences.
Tests compare three steps against PyTorch AdamW on CPU and CUDA with two
microbatches, graph-state gradients, tied embeddings, and global clipping.
BF16 computes gradients at BF16 precision with FP32 disk masters/moments;
it is not numerically identical to FP32 model execution.

Only one stage's accumulated gradients and the tied-head contribution persist
in RAM. File mappings are opened for one parameter and released; optimizer
updates operate in 1,048,576-element chunks. CUDA uses two pinned host/weight
slots and an H2D stream with load/computation events to prefetch the next stage.
Disk reads and gradient evacuation currently synchronize with the host, so
sustained IO/computation overlap is not claimed.

## Run a small validation first

```bash
python train.py --engine disk --optimizer adamw --text docs/smoke.txt \
  --device cuda --precision bf16 --sequence-length 8 --accumulation 2 \
  --steps 3 --state-dir /absolute/path/debug-disk-state
python train.py --engine disk --optimizer adamw --text docs/smoke.txt \
  --device cuda --precision bf16 --sequence-length 8 --accumulation 2 \
  --steps 5 --state-dir /absolute/path/debug-disk-state --resume-store
```

The directory must not exist for a new run. Each completed optimizer step is
its own durable state; `--checkpoint` and `--resume` are replaced by
`--state-dir` and `--resume-store` in this mode. Data, LR, precision, seed,
batch/window size, accumulation, device type, and configuration must match.

The 7B planning command allocates no model or state files:

```bash
python train.py --config configs/morph_7b.json --engine disk --device cuda \
  --precision bf16 --sequence-length 32 --accumulation 1 --inspect
```

The estimated envelope at B1/T32/K1 is about **9.98 GiB host RAM and 5.57 GiB
CUDA memory**, including buffer pools, a conservative layer-workspace estimate,
all accumulation boundaries, all microbatch cotangents, largest-parameter
mapping/work windows, logits, and runtime reserves. These are estimates, not
measured 7B peaks. Allocator/kernel behavior and corpus storage can add memory.
The CLI checks available RAM/VRAM/disk with headroom; reduce sequence length
or accumulation if rejected.

With prepared local token IDs, enough free disk, and those checks passing:

```bash
python train.py --config configs/morph_7b.json --engine disk --optimizer adamw \
  --tokens /absolute/path/corpus-shard.json --device cuda --precision bf16 \
  --sequence-length 32 --batch-size 1 --accumulation 1 --lr 0.0003 \
  --steps 10 --state-dir /absolute/path/morph7b-disk-state
```

No full 7B allocation or training run has been performed. Use a local shard
that fits in remaining RAM for text/JSON, or compile a memory-mapped shard
with `python -m data.prepare` and pass `--shard` instead of `--tokens`. This is full-parameter training, not adapters or frozen layers.

## Durability and cost

Initialization failures still fail closed. Training updates now use a durable
stage transaction: original forward boundaries, norm/clipping multiplier,
microbatches, cotangents, tied-head gradient and stage cursor are saved. Before
mutating a stage, its old FP32 weight/m/v files are copied into a bounded undo
journal and fsynced. A durable cursor commit either retains the complete new
stage or restores the journal and recomputes that uncommitted stage on restart.
Remaining reverse stages resume from the saved original boundaries/cotangents.
Six injected failure boundaries recover byte-identical weights and both Adam
moments, including a subsequent optimizer update. No whole-model duplicate is
required. `--resume-store` resumes recoverable in-progress transactions; legacy
dirty stores without transaction state remain rejected.

The disk guard now reserves the largest stage's 12-byte/parameter undo state,
three replay-boundary windows, two tied-gradient windows, and 2 GiB headroom in
addition to canonical state. These files add IO beyond the original two-pass
numbers below. See [audit implementation](audit_implementation.md) for the
recovery protocol and its tested limits.

The extra backward pass and disk IO reduce throughput. For K1, the 7B step
logically reads/writes roughly **254 GB** of weights/moments plus activation
traffic and filesystem overhead; caching changes physical device traffic.
BF16 parameter H2D is roughly **43 GB**, FP32 gradient D2H roughly **58 GB**
per microbatch across the two passes. Increasing accumulation changes those
costs; stage weights are reused across the backward microbatches. Async
prefetch hides some stalls, but cannot eliminate bytes or recomputation.

The reproducible debug-only comparison command is:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/disk_compare.py --device cuda
```

See [measured debug output](disk_cuda_measurements.json). The debug model has
37,672 parameters, so Python/CUDA startup, filesystem fsync, and runtime memory
dominate; these numbers must not be extrapolated into 7B throughput claims.

## Research basis

[Adafactor](https://arxiv.org/abs/1804.04235) reduces moment storage by factoring
matrix second moments; adopting it would change the optimizer approximation.
[Blockwise 8-bit optimizers](https://arxiv.org/abs/2110.02861) reduce optimizer
state through quantization. [ZeRO-Offload](https://arxiv.org/abs/2101.06840)
provides a primary reference for moving training state and work off GPU.
This implementation chooses ordinary FP32 AdamW states on disk to preserve
the optimizer baseline while eliminating the model-sized RAM gradient store.
Comparable update quality is checked numerically; comparable training speed
has not been demonstrated.
