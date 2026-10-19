# MORPH research implementation

Executable native causal selective-SSM/SwiGLU model with tied embeddings,
fixed neural graph memory, four shared graph refinement iterations, and
Dynamic Layer-Sequential Forward-Backward (DLS-FB) training. This is a tested
research reference, not a pretrained mathematical prover.

The 7B preset contains **6,995,110,080 trainable parameters**. Counting uses
PyTorch's meta device without allocating model weights:

```bash
python train.py --config configs/morph_7b.json --inspect --engine async --device cuda --precision bf16
```

## Train within the current RAM limit

The new [two-pass disk-backed AdamW mode](docs/disk_adamw.md) moves FP32 weights
and Adam moments to disk and holds only stage windows in RAM. The 7B state
requires 78.18 GiB disk; B1/T32/K1 BF16 has a conservative ~9.98 GiB host and
~5.57 GiB CUDA estimate. No full 7B run has been performed. It preserves
AdamW updates in small comparisons, at the cost of extra replay and disk IO.
See that guide for commands, guards, and the bounded transaction recovery protocol.

## Run now

Use the existing Python environment with PyTorch and NumPy; pytest runs checks.
No models, packages, or datasets are downloaded by the training code.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python train.py --text docs/smoke.txt --steps 20 --sequence-length 8 --checkpoint checkpoints/debug.pt
python train.py --text docs/smoke.txt --steps 40 --sequence-length 8 --resume checkpoints/debug.pt --checkpoint checkpoints/debug.pt
```

The same debug model runs through the synchronous streamed reference and CUDA
asynchronous ping-pong path:

```bash
python train.py --text docs/smoke.txt --steps 20 --sequence-length 8 --engine dls --device cuda --precision bf16
python train.py --text docs/smoke.txt --steps 20 --sequence-length 8 --engine async --device cuda --precision bf16 --checkpoint checkpoints/async.pt
```

`--text` uses exact UTF-8 byte IDs 0–255. This simple tokenizer works for smoke
training, but leaves most of a 65,536-entry vocabulary unused. For meaningful
7B corpus training, supply `--tokens corpus.json`, containing an integer ID
list or rectangular list of examples encoded by your chosen tokenizer. All
IDs must be below `vocab_size`; retain the tokenizer alongside the checkpoint.
The sampler draws reproducible random contiguous windows of
`sequence-length + 1` IDs. No training/validation splitting is automatic.

`--steps` is the total optimizer-step target, including resumed steps.
Checkpoints include model, optimizer/momentum, progress, CPU/CUDA RNG, data
fingerprint, and training settings. Resume rejects changed data or optimizer
settings. Writes use temporary-file replacement. `--checkpoint-every` saves every 100
optimizer steps by default when a checkpoint path is supplied, and at the end
of each successful run. `--log-every` controls JSON progress output (default 1).
Resume requires the same CPU/CUDA device type; device migration is not supported.

## Full 7B prerequisites

Detected development host: RTX 5070, 12,227 MiB VRAM, **29 GiB system RAM**.
RAM-backed modes cannot hold the 7B FP32 CPU master parameters, gradients, and optimizer
state. Layer streaming solves GPU residency, not the CPU storage requirement.
The CLI refuses this allocation before constructing the real model.

Hybrid Muon + AdamW assigns **6,759,841,792** hidden Linear-matrix parameters to
Muon, with FP32 masters/gradients/one momentum: 12 bytes each. The remaining
**235,268,288** embedding, norm, bias, graph seed/relation parameters use AdamW:
16 bytes each. Total persistent model training tensors: **79.05 GiB**, excluding
pinned buffers, activation boundaries, optimizer temporaries, corpus, runtime,
and checkpoint loading. Investigate at least 128 GiB RAM; 192 GiB provides
more headroom. This is an estimate, not a validated 7B memory benchmark.
AdamW for every parameter requires 104.24 GiB persistent tensors.

On a host with sufficient RAM, CUDA/BF16 support, free checkpoint storage,
and a prepared local token corpus, the full-model streaming command is:

```bash
python train.py --config configs/morph_7b.json --tokens /absolute/path/corpus.json \
  --engine async --device cuda --precision bf16 --optimizer hybrid \
  --lr 0.0003 --muon-lr 0.002 --sequence-length 128 --batch-size 1 \
  --accumulation 16 --steps 100 --checkpoint /absolute/path/checkpoints/morph7b.pt
```

This command is provided for adequately provisioned hardware; no full 7B run
has been performed. A pure-PyTorch sequential SSM and CPU Newton-Schulz optimizer
will be slow at this scale. Measure throughput and memory on intermediate
sizes before committing to substantial training.

## Execution and correctness

Resident mode is ordinary end-to-end autograd. DLS-FB keeps authoritative FP32
weights/gradients/optimizer on CPU, saves layer token and graph input boundaries
to CPU, reloads layers in reverse, and recomputes exact local derivatives.
Graph-state cotangents propagate across exchange blocks. Weights stay unchanged
until all accumulated microbatches complete; clipping uses one global norm.
Tied embedding gradients from input and output uses sum into one CPU parameter.

Async mode has two reusable flat CUDA weight buffers, two pinned host staging
buffers, pinned FP32 gradient staging, H2D/D2H streams, and completion events.
Before buffer reuse, computation and outstanding gradient evacuation complete;
pinned source storage is retained until DMA completes. Parameter versions
are checked during replay. BF16 mode transfers BF16 weights from FP32 CPU
masters, computes with BF16 autocast, and offloads gradients as FP32.

Prefetch can overlap transfers with computation; this hides stalls and does
**not** by itself reduce PCIe bytes. BF16 weight transfer halves weight traffic
relative to FP32. The final vocabulary head runs only during backward replay.
For this preset, weight H2D traffic is approximately 28.45 GB and parameter
gradient D2H traffic 28.92 GB per microbatch, before activation/graph transfers.
CPU checkpoint copies synchronize, so sustained transfer overlap and 7B token
throughput are not claimed. Each run reports elapsed time, tokens/second,
peak CUDA allocated bytes, and tracked weight/gradient transfer bytes (excluding
activation traffic). The separate disk mode is documented above.

Tests cover analytic SSM updates, FP64 finite-difference gradients including
initial state, causal/chunk equivalence, all-parameter backward, graph masked
aggregation, shared refinement gradients, activation-checkpoint equivalence,
CPU/CUDA streamed-vs-resident loss/gradients/updates, repeated buffer reuse,
BF16 CUDA tolerances/transfer accounting, overfit, auxiliary AdamW equivalence,
nonfinite optimizer rejection, and exact checkpoint/resume with accumulation.

## Architectural scope and roadmap

The SSM now defaults to a bounded chunk-parallel affine scan with a scalar
reference and forward/backward equivalence checks. It is not a fused Mamba-2
kernel or compatible pretrained checkpoint implementation; causal convolution
is not implemented.

The causal language-model baseline retains neural ring memory. Opt-in
`proof_enabled` adds typed symbolic graph inputs, external deterministic graph
actions/replay, a separate shared post-backbone reasoning core, action-conditioned
node selection, dependency/value heads, and joint teacher-forced losses.
Structured proof training currently requires resident FP32 execution; streaming
rejects that mode rather than omit its parameters.

[The audit implementation guide](docs/audit_implementation.md) contains commands,
actual 100M CPU timing evidence, typed proof/data/Lean interfaces, transactional
disk recovery, and remaining empirical gates. Real Lean integration is skipped
when Lean is absent. No verified proof success, architectural superiority, or
validated full 7B/15B training throughput is claimed.

Muon follows the algorithm in the author's [primary implementation](https://github.com/KellerJordan/Muon/blob/master/muon.py),
using FP32 Newton-Schulz iterations as a reproducible CPU baseline instead of
the original BF16 orthogonalization. Muon and Adam learning rates are separate
experimental settings, not tuned recommendations.
