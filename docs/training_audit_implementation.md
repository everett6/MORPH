# MORPH training-speed audit: implementation and limits

The isolated experiment retains all 4,981,264,481 parameters and all 64 experts. Defaults remain compatible with the original trainer. The original training directory has not been opened for mutation, migrated, resumed or deleted during this work.

## Implemented milestones

1. Full-step profiling now covers the sparse execution path, including CPU preparation, canonical reads and casts, prefetch waits, forward/backward launch wall time, expert-specific intervals, H2D/D2H CUDA stream intervals, optimizer updates, journals and synchronization counts. `metrics.jsonl` records per-step deltas. CUDA intervals include stream idle and overlap CPU wall time; adding them does not produce a valid total GPU-busy estimate.
2. Optional packed stage undo journals use bounded streaming copies and checksums. The complete journal is validated before recovery changes any canonical file. Wrong-format recovery fails closed. Packing reduces per-tensor synchronization overhead; it still writes the same undo payload and canonical FP32 arrays. Cache write-through durability remains intact.
3. `--chunk-microbatches` bounds live activation windows independently of `--accumulation`. Gradients are unscaled, accumulated in FP32 across windows, globally clipped once, then applied in a single AdamW transaction. The expert roster is fixed for the transaction. Overflow retries replay all windows before publishing gradients. Recovery uses durable gradients rather than backward through partially updated weights. Chunk reduction order may introduce normal floating-point differences.
4. `expert_group_size` optionally groups experts using padded batched matrix multiplications. Parameter selection fetches only experts actually routed in the window. Default size one preserves the existing stages; grouping changes the stage layout and is part of persisted settings. Padding and transient stacked weights cost extra memory and are included in preflight.
5. `memory_projection_backend="batched"` batches token-local key/value/query/gate projections and graph conditioning. The causal FP32 fast-memory recurrence and its derivative remain sequential and unchanged. This is a projection optimization, not a fused recurrence kernel.
6. Small real-data qualification, full-step profiles, held-out row separation, recovery faults, bounded memory checks and FP16 CUDA comparisons are recorded separately from full-size capacity checks. The 2,000 byte-target/s threshold remains a measured acceptance criterion; a small-model result cannot qualify the 5B model or prove a seven-day deadline.

## Prepared full-size configuration

From the isolated worktree, this performs only a preflight:

```bash
python -m streamtrain.delta_cycle \
  --config configs/morph_delta_5b_audit.json \
  --engine single-pass --journal-format packed \
  --chunk-microbatches 2 --accumulation 64 \
  --batch-size 4 --sequence-length 128 \
  --cache-bytes 1073741824 --stash-budget-bytes 4294967296 \
  --experiment-run-dir runs/delta-audit-5b \
  --target-tokens 1000000000 \
  --device cuda --precision fp16 --backend triton \
  --preflight
```

This describes 32,768 next-byte targets per optimizer update with at most two microbatches' activation boundaries live. It retains every target. Fewer optimizer commits per target change the optimization schedule; learning-rate and convergence validation are still needed. The preset keeps group size one because grouping should be enabled only after checking its resource report and measured benefit.

The [current capacity report](delta_audit_preflight.json) is a snapshot, not a permanent hardware guarantee. Creating a second full-size bank still needs more free SSD space. No second 5B bank was allocated and no full-size speedup is claimed.

## Existing run status

On October 21, 2026, PID 64808 was absent. The last committed update was step 26 with 106,496 byte targets. The manifest indicates a pending update, with a journal and reverse cursor; the source has 4,096 pending targets. These records require the original compatible recovery path before continuing that run. Presence of recovery records is not a completed recovery test of that particular checkpoint. No automatic restart was performed.

The original state and source manifests had these SHA-256 digests when inspected:

- State manifest: `c2dafab247f0f7fb3f0fcffe8e8f8d16aa0246d1100de4d9aca80289e19dbd20`
- Source manifest: `4497f38598780b199ee39a1e1d8a1b0c3508402712e1aa76065ce47eac334114`

## Results

See [real-data qualification](audit_qualification.md), its [numeric report](audit_qualification.json), and the full-size capacity report. The initial qualification helper prebuffers a bounded small RAM sample to give baseline and optimized engines identical inputs; its optimizer-step throughput excludes fetching. An additional streamed-cycle measurement includes fetching and source commits. Three updates are short mechanical qualification, not sustained 5B convergence evidence.

## Verified measurements on October 21

The real-data benchmark used an **81,910-parameter test model**, not the full 5B bank. Three controlled GPU updates on identical inputs measured 558.64 byte targets/s for two-pass execution and 822.04 for the combined optional paths. A separate 32,768-target/update trial measured 3,458.83 targets/s during optimizer steps.

The [direct streaming-cycle report](audit_streaming_cycle.json) includes source resolution/fetching, initialization, source cursor commits and all three optimizer updates: 98,304 targets in 34.36 seconds, or **2,861.28 targets/s end to end**. Its CUDA peak allocation was 70,194,688 bytes. Imports and CUDA context warmup are excluded. The temporary test state was removed; no bulk dataset cache was created. Reproduce with `python -m benchmarks.audit_streaming_cycle`.

Held-out rows were disjoint from training rows in the qualification helper. Held-out loss decreased from 5.54268 to 5.42390 in the larger-batch trial. Three commits do not establish convergence or mathematical reasoning quality.

The [FP16 precision gate](audit_precision.md) measured multi-step update differences of 1.83%, 1.16% and 0.89% versus the reference, below the declared 10% bound. Both group sizes one and two passed; this is numerical tolerance evidence, not a guarantee of equal final model quality.

Final checks: **177 CPU tests passed, 22 skipped; 8 focused CUDA tests passed**. Recovery, packed export and wrong-format rejection are included. The original run's two manifest hashes remained unchanged.

The full-size preset requires approximately **87.83 GB free SSD**, versus **22.13 GB available** in the current snapshot. Its estimated host envelope is 9.53 GB and its estimated GPU envelope is 7.63 GB. Those memory checks pass; allocating a separate full-size bank is blocked by disk capacity. The full 5B model has not passed the 2,000-target/s threshold, and no seven-day completion claim is supported.
