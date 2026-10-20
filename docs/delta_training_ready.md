# Native MORPH-Delta training and operation

The implemented model has **4,981,264,481 trainable parameters**, including
4,681,334,784 in 64 expert bundles. Shared capacity plus one bundle is
373,075,553 parameters; actual FLOPs and the set touched by a full batch differ.
This is a newly initialized architecture, separate from the retired dense run.

## What is connected

- Shared causal SSM backbone with the local Triton backend and a 256-byte tied
  embedding/output vocabulary.
- Dropless, token-conditioned top-1 sparse experts, differentiable router gates,
  and load-balancing loss. Each bundle is an independently loadable stage.
- Optional per-expert latent bottlenecks. These are a native research variant;
  they do not reproduce NVIDIA LatentMoE's shared projection architecture.
- Differentiable FP32 session fast memory, causal reads/writes, reset, branch
  cloning and rollback. Training differentiates through memory updates.
- Typed proof-graph conditioning of memory, routing and final proof logits;
  action, target, dependency and value supervision in the proof stage.
- Frozen inference caches that re-encode changed proof nodes and invalidate
  dependent aggregation, weight versions, precision/device and memory epochs.
  Training retains full gradient paths instead of detached cache reuse.
- Layer-major forward and two-pass exact global-norm replay; FP32 conditional
  AdamW masters/moments and per-parameter activation clocks. Inactive experts
  have no fetch or optimizer update. This follows absent-gradient AdamW semantics,
  rather than treating inactivity as dense zero-gradient updates.
- One CPU preparation worker, two pinned/device stage buffers, CUDA events and
  separate transfer streams. CPU reads can overlap current-stage computation.
  Stage gradients accumulate in the preallocated FP32 GPU window before a single
  host transfer, reducing repeated PCIe traffic across microbatches.
- Recoverable stage journals, atomic optimizer clocks/cursors, and one immutable
  tied-gradient record per transaction. Canonical state is the checkpoint.
- Bounded FP16 shard exports after the first update, every 1,000 updates and on
  a clean stop/completion. The previous export is retained until replacement
  completes, then removed. FP32 training state remains authoritative.

## Data streaming and token accounting

`configs/hf_math_stream.json` selects OpenWebMath `default/train`, field `text`,
pinned to commit `fde8ef8de2300f5e778f56261843dab89f230815`.
The stdlib HTTP reader receives small Hugging Face row pages into bounded RAM.
It creates no dataset shard/download cache and installs no packages. Network
transfer is necessary to stream the text. Small current-page/carry and pending
batch bytes are saved with the training cursor for exact crash recovery; this
is not a persistent corpus copy.

Every nonpadding next-byte target contributes to loss. Adjacent windows overlap
by one context byte so targets are not lost between windows. The final batch
can be shorter to meet the exact target budget. Source revision changes,
truncated text fields, missing/discontinuous rows and exhausted data fail
explicitly. Dataset-server availability and access remain runtime dependencies.
Byte counts are not interchangeable with other papers' subword token counts.

The cycle saves a pending source cursor before the optimizer starts. After an
optimizer commit, it commits that cursor exactly once. Recovery also handles
an interruption between these two commits. The first budget is **1 billion
processed byte targets**; it is not a claim of model quality or unique data.

## Start and resume

Use the existing Python installation from the repository root:

```bash
/home/everett/miniconda3/bin/python -m streamtrain.delta_cycle --preflight
/home/everett/miniconda3/bin/python -m streamtrain.delta_cycle \
  --config configs/morph_delta_5b.json \
  --source-config configs/hf_math_stream.json \
  --run-dir runs/morph-delta5b-hf-math \
  --target-tokens 1000000000 --device cuda --precision fp16 \
  --batch-size 4 --sequence-length 128 --accumulation 8 --threads 4
```

There is no automatic calibration-step cap in this command. `--max-steps` is
available only when an explicitly bounded experiment is desired. The separate
`runs/delta-hf-qualification` experiment used a tiny validation model and
completed 128 targets; those targets are not credited to the full model.

To resume, repeat the full command with `--resume`. Settings and source
fingerprints must match. A run lock prevents two writers. To stop safely, send
SIGTERM to the PID in `process.json`; it completes its current optimizer update
and exports before exiting. A forced interruption requires transaction recovery.
Never delete a live transaction or start a second direct writer.

Check `status.json`, `metrics.jsonl`, `process.log` and `fp16-latest.json` inside
the run directory. Only committed optimizer updates increase the processed count.
Stage status shows progress within a long update. Metrics include full-step
seconds, actual targets, active experts, transfers and physical storage IO.
Actual 5B throughput must be measured; the tiny test is not a speed prediction.

## Precision choice

FP16 compute/transfers and FP32 weights, Adam moments, clipping and gradient
accumulation are enabled. The new model's optional FP8 qualification **failed**:
loss drift was small but Adam update drift exceeded the chosen 10% threshold.
The launch command therefore does not enable FP8. See
`docs/delta_fp8_precision_gate.json`. FP8 transport dequantizes before GEMMs;
it must not be described as native FP8 compute. No FP4 optimizer or master
weights are used.

## Formal proof curriculum

Stage A learns next-byte mathematical text. It does not train the unused proof
heads from OpenWebMath or certify those web documents. A subsequent explicit
same-state transition can train local structured proof records without creating
a second 5B optimizer checkpoint:

```bash
/home/everett/miniconda3/bin/python -m streamtrain.delta_proof \
  --config configs/morph_delta_5b.json \
  --traces .generated/verified_declarations.jsonl \
  --run-dir runs/morph-delta5b-hf-math --continue-lm-state --steps 1 \
  --device cuda --precision fp16 --sequence-length 256 --export --preflight
```

First stop the language-model cycle cleanly. Inspect the proof preflight before
running without `--preflight`; a long proof sequence has a separate memory
budget and is not guaranteed to fit. `--continue-lm-state` explicitly preserves
FP32 masters/moments/clocks and records the phase change. Its `--steps` counts
additional updates; ordinary proof resume uses absolute total steps. Do not
resume the old language-model cursor after proof updates without a deliberate
new phase policy.

Local declaration records are verified reference-check examples, not multi-step
proof discovery. `.generated/proof_loop_training.jsonl` contains small verified
toy/seeded-repair examples, and `.generated/search_values.jsonl` contains search
value examples. Held-out theorem families, Lean certificates, repair success and
matched search/checker budgets remain necessary evaluations. Architecture and
unit tests alone do not establish mathematical ability.

## Storage and retained work

The user authorized retirement of the old dense FP32 training state to make room
for the new bank. Its 544 FP16 shards were rehashed before deletion; the FP16
model remains at `runs/morph5b-math-stage-a/fp16-step-000000001`. The deletion
receipt is `runs/morph5b-math-stage-a/retired-state.json`; that old optimizer run
cannot resume. New preflight checks state, recovery data, replacement exports,
RAM and VRAM before allocating the full bank. Streaming saves corpus space;
it does not eliminate optimizer IO or storage requirements.
