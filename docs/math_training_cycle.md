# Local 5B math-source training cycle

**Historical dense calibration:** its FP32 training state was deleted with user
approval to make room for MORPH-Delta. The verified FP16 export was retained.
This old run can no longer resume. Use [the Delta training guide](delta_training_ready.md)
for the active Hugging Face streamed training pipeline. Details below record the
earlier experiment and are not current launch instructions.

The run is currently a one-step calibration segment toward the retained one-billion
target, followed by a safe pause for measured throughput and storage review.
The original BF16 process initialized its store but completed zero updates.

Stage A trains the full native LM baseline:5,034,142,016 parameters,42 layers,
width3072. It uses only existing local Mathlib sources. No installer, network
request or download is part of compilation/training. This is Lean/math-source
next-byte training, not a verified multi-step proof trajectory stage.

The initial local corpus contains6305 unique `.lean` files after exact SHA256
content deduplication, pinned Mathlib revision
`c44e0c8ee63ca166450922a373c7409c5d26b00b`, Apache2.0. Top-level module families
are deterministically assigned to train/validation/test, including submodules:
4720train files/52,824,777 byte tokens;778validation/7,622,857;807test/12,730,281.
Semantic duplicates and imported dependencies can cross module families; this is
not a theorem-generalization benchmark. Only train records enter the sampler.
Validation uses a fixed held-out window every100 optimizer steps.

The requested1,000,000,000 budget counts actual nonpadding next-byte targets
processed, including repetition of local data. It does not mean a billion unique
math tokens. Windows are sampled uniformly over eligible source files, then
uniformly within the chosen file; this is not a complete-pass/epoch traversal.

## Run and control

```bash
python -m data.local_math --source .tools/mathlib4/Mathlib --output .generated/mathlib-source-shards
python -m streamtrain.cycle --config configs/morph_5b.json \
  --shard .generated/mathlib-source-shards --run-dir runs/morph5b-math-stage-a \
  --target-tokens 1000000000 --device cuda --precision fp16 \
  --sequence-length 128 --accumulation 8 --lr 0.0001 \
  --validation-every 100 --threads 2 --backend triton --fp8-transfer \
  --max-steps 2 --export-every 1000
```

The launched process is detached with a new session; exact invocation/PID are
in `runs/morph5b-math-stage-a/process.json`. It keeps running after the chat.
Read `status.json`, append-only `metrics.jsonl`, or `process.log` there for
progress. They are local ignored runtime artifacts. `state/` is the canonical
checkpoint, including stage recovery journal/cursor when a step is in progress.
No second whole-model checkpoint is created. A run lock prevents concurrent
writers using this entrypoint. Do not run another trainer directly on its store.

Request a safe stop with `kill -TERM PID` using the recorded PID. The process
finishes its current initialization/optimizer step, writes durable progress and
exits. Resume by repeating the exact command with `--resume`; do not delete the
store. Forced termination during a committed transaction resumes using the
existing stage journal/cursor. Initialization interruption remains fail-closed
and requires deliberate recreation of incomplete state. A signal before the
transaction starts may discard that unfinished step, with weights unchanged.
Sampler steps are deterministic and checkpoint counts derive from completed
steps, so a crash between store commit and progress logging cannot double-count.

The selected schedule is B1/T128/K8 (1024 targets per ordinary optimizer step),
FP32 disk masters and AdamW moments, FP16 compute with dynamic loss scaling,
FP32 recurrence and accumulated/clipped parameter gradients,
learning rate1e-4, global norm clipping1.0, weight decay0.01, seed42. No warmup,
quality-tuned learning-rate schedule or mathematical ability claim is implied.
Optional FFN gate/up/down temporary copies use block-scaled E4M3FN FP8
(256 values per FP32 scale) only after the numerical gate passes. Everything
else transfers as FP16; FP8 copies are dequantized to FP16 before computation.
Canonical optimizer arithmetic remains FP32; FP8 forward rounding changes
gradients, so it is not bitwise AdamW/full-FP16 equivalence. The small complete
model gate measured0.00962% relative loss drift and1.94849% gradient L2 drift;
quantized resident/disk gradient discrepancy was0.02034%. This does not prove
5B convergence or speed. See `fp8_precision_gate.json`. Failed/missing gate
automatically chooses FP16 transfer instead.

Triton is already installed locally. Fallback to a smaller model is prohibited.

Preflight estimates canonical state56.26GiB, additional recovery/storage
reserve9.36GiB, host10.30GiB and CUDA6.64GiB including FP8 temporary workspace for this schedule. These estimates
include bounded stage windows, optimizer temporaries and accumulation boundaries;
actual measured values appear in the runtime metrics. The1B budget may take a
long time: measure complete steps before estimating a finish date. Disk replay,
state writes and checkpoint traffic can dominate throughput. This is a real
training start, not evidence that full5B pretraining is economical on one GPU.


FP16 weights export as bounded raw shards with hashes, config and shared-parameter
alias metadata, separate from FP32 optimizer state. The cycle exports after its
calibration segment and every1000 completed steps, retaining the latest completed
export. Space checks include the new export plus journal reserve. No whole-model
RAM copy is required. `fp16-latest.json` points to the latest complete export.

Each optimizer step logically copies old weight/m/v into its undo journal and
writes new state: approximately24 bytes/parameter,120.82GB for this5B model,
before activation/cursor writes and filesystem amplification. At1024 targets
per step, extrapolating to1B would imply approximately118PB logical writes.
That is why this segment stops after one real step for review. FP8 transfer
does not reduce FP32 optimizer/journal disk traffic. Runtime metrics record
actual `/proc/self/io` read/write bytes and step seconds; no SSD endurance
rating or sustained throughput is assumed.


Load the delivered FP16 model without a second FP32 initialization:

```python
import torch
from streamtrain.export import load_fp16
model = load_fp16("runs/morph5b-math-stage-a/fp16-step-000000001")
# CPU tensors map private FP16 shards; parameters stay FP16. Recurrence is FP32.
with torch.inference_mode(), torch.autocast("cpu", dtype=torch.float16):
    output = model(torch.tensor([[71, 111, 97, 108]]))
```

`load_fp16(..., device="cuda")` additionally guards free memory before copying
mapped FP16 weights. It rejects invalid schema, hashes, alias metadata,
nonfinite values and escaping paths. CPU inference may be slow. The small
roundtrip test matches a directly rounded FP16 model under autocast.

## Revised scheduling calibration

The initial real step used microbatch-major forward replay. The revised scheduler
loads a stage once, runs all accumulated microbatches, and retains each `(X,G)`
boundary on CPU. It does not retain all microbatch activations on GPU. Existing
stages have no random/dropout forward operations; adding those requires explicit
per-stage/microbatch RNG replay. Distinct K3 forward states and complete FP32
weight/m/v updates match the old schedule exactly in tests.

Optional `--pack-factor 2` groups the same eight sampled B1 windows into four B2
microbatches. Effective tokens, sampled window order, loss weighting, learning
rate and optimizer update count stay fixed. Incomplete/uneven groups fall back
without incorrect loss weighting. B4 is rejected by the conservative GPU budget;
B2/K4 has an estimated8.39GiB CUDA envelope including FP8 temporaries.
Clean resume records an explicit `--execution-transition`; the old/new policy
is saved. Pending transactions retain their original packed batches/settings.

A small forward-only GPU experiment (1,682,176params,T128,effectiveB8,
FP16+FFNFP8,one warmup,three measured forwards) recorded medians2.219s old,
2.124s layer-major,1.074s layer-major+B2. Layer-major reduced weight-transfer
bytes8×. This is not a full5B throughput comparison; model size, cache effects
and concurrent host/GPU activity differ. Details are in
`layer_major_forward_profile.json`.

New processes emit throttled phase/layer status and cumulative phase timings for
canonical read/cast, FP8 CPU packing, transfer preparation/scheduling, forward
compute/boundary copy, reverse launch and gradient D2H/accumulation. CUDA events
measure elapsed compute-stream time, including launch/synchronization gaps;
that is not isolated active-kernel time. Timing categories overlap and must not
be summed. CPU preparation remains synchronous; CUDA copy streams alone do not
hide CPU read/packing work. No CPU prefetch worker is claimed.

The current recovery cursor also serializes the immutable tied-embedding head
gradient (approximately805MB for this vocabulary/width) at each reverse stage.
This adds substantial repeated writes beyond the24N state/journal estimate.
Persisting that gradient once per transaction and referencing it from cursors is
remaining work. CPU-prefetch preparation is also not implemented. The prototype
therefore does not yet feed the GPU efficiently; the small scheduling improvement
does not establish that low utilization or full5B throughput is solved.

Ready optimized bounded resume command (do not start a concurrent writer):

```bash
python -m streamtrain.cycle --config configs/morph_5b.json \
  --shard .generated/mathlib-source-shards --run-dir runs/morph5b-math-stage-a \
  --target-tokens 1000000000 --device cuda --precision fp16 \
  --sequence-length 128 --accumulation 8 --pack-factor 2 --lr 0.0001 \
  --validation-every 100 --threads 2 --backend triton --fp8-transfer \
  --resume --execution-transition --max-steps 1 --export-every 1000
```

This runs at most one additional optimizer step, keeps the original1B target and
sampler schedule, and pauses again with a clean FP16 export. It has not been
run at full5B; the only measured full5B step is the original schedule.

## Actual first 5B result

The original schedule completed one full optimizer step over1024 actual targets,
then safely stopped. Loss11.64959; unscaled norm708.164 was clipped to1.0;
loss scale128. All5,034,142,016 parameters belonged to the real trainable model;
no smaller fallback ran. The step took915.52s (1.11848tokens/s), with peak CUDA
allocated3.52GiB and peak process RSS5.67GiB. Low CUDA residency is intentional
stage buffering; low utilization remains a host/transfer/submission bottleneck.

Measured process IO during the step:127.69GB physical reads and156.65GB physical
writes, excluding initialization and FP16 export. Tracked H2D weights78.08GB and
D2H parameter gradients335.07GB exclude activation traffic. At this single-step
rate, extrapolating1B targets gives approximately28.3years and153PB writes.
These are arithmetic projections, not sustained-performance/endurance forecasts.
The run is paused for practical engineering review, not running indefinitely.

The clean FP16 export has544 hashed shards,10,068,284,032 weight bytes
(9.37682GiB), at `runs/morph5b-math-stage-a/fp16-step-000000001`.
Every shard hash was independently re-read/verified after export. Manifest SHA256:
`1cf2c88eba171b5ecf281934cc84d3f34268a5d59f4861456959e27232179083`.
Status is `stopped`, step1, processed1024/1,000,000,000, `safe_to_resume=true`.
No optimized full5B step was run. See `5b_math_calibration.json` for exact bytes
and settings, and the bounded optimized resume command above. One step does not
establish trained mathematical capability or a useful proof model.
