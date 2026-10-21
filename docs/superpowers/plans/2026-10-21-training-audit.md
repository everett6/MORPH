# MORPH training-speed audit implementation plan

Goal: implement the supplied audit in the isolated 5B experiment while preserving existing checkpoints and exact conditional FP32 AdamW semantics.

Architecture: retain all 64 experts and the causal SSM/proof model. Add optional packed undo journals, bounded activation-window gradient accumulation, and optimized expert/memory execution. Profile actual full optimizer steps and qualify on small real RAM-streamed data before making throughput projections.

Constraints: no installations or model/dataset downloads; HF rows may stream into bounded RAM. Never open the original run for mutation, resume, delete or migrate it. Default behavior remains compatible. The 2,000 byte-target/s sustained goal is a qualification threshold, not a promised result.

Decisions: packed stage journals reduce per-tensor durability overhead while retaining undo protection; they do not eliminate canonical writes. Chunk accumulation sums all unscaled gradients before a single global clipping/update; roster stays fixed per transaction. Optimized execution must preserve causal outputs, gradients, tied contributions, None-gradient clocks and FP32 memory recurrence. If a proposed grouping would fetch dormant experts or exceed capacity, reject that configuration rather than silently changing semantics.

Tasks:
- [x] Profiling: per-step wall/CPU/CUDA stream, preparation, transfers, optimizer/journal/sync timing, counts, bytes, expert work; overlapping timings explicitly labeled.
- [x] Transactions: optional bounded packed journal, durability/recovery/schema/truncation tests, skip empty saved-gradient stages where safe.
- [x] Accumulation: configurable maximum microbatches with live activation boundaries, same target budget, FP32 global gradient sum/clip, overflow and replay tests.
- [x] Execution: optimized batched projections for exact differentiable session memory; investigate/group expert execution without changing routing or inactive clocks. Verify full outputs/gradients.
- [x] Integration: explicit CLI options and persisted compatibility, resource checks reflecting bounded windows, profile and benchmark reports.
- [x] Qualification: CPU suite, FP16 CUDA parity, short real HF RAM-streamed train/eval, interruption recovery and measured memory. Report if throughput threshold is unmet. No full duplicate 5B bank allocation without capacity.

Review focus: interrupted packed journals fail closed; uneven final batches preserve objective weights; FP16 overflow replays a clean optimizer transaction; inactive experts never get decay/clocks; reports do not add overlapping times or extrapolate small-model speeds to 5B.

Completed implementation and small-model qualification. Full-size sustained qualification remains blocked by separate-bank disk capacity; the seven-day goal is not established. Results: docs/training_audit_implementation.md.
