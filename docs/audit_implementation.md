# Audit implementation and remaining empirical gates

This update implements the main proof-state, supervised-training, and recovery
foundations; the table identifies partial integrations and empirical gates.
This does not establish that MORPH outperforms an SSM, proves new theorems, or trains
productively at 7B/15B. Existing 7B parameter count is unchanged unless the
new proof mode is explicitly enabled.

## Implementation status

| Audit item | Code delivered | Evidence and remaining gate |
|---|---|---|
| 100M performance benchmark | Resource-qualified scaling/component harness | CPU measurements below; occupied GPU safely skipped |
| Python SSM bottleneck | Stable parallel affine scan inside bounded chunks; scalar reference retained | Forward/all-input/state gradients at multiple chunk boundaries; not a fused CUDA kernel |
| Symbolic proof state | Typed nodes, independent status, exact assumptions/provenance, directed hyperedges | Metadata/tensor integration tests; status never inferred from confidence |
| Dynamic actions | External create/connect/decompose/revise/prune/verify requests, masks, hash-checked replay | Invalid action-target pairs and stale dependency revisions rejected |
| Separate reasoning core | Opt-in shared post-backbone refinement, action-conditioned targets, value and dependency heads | Joint losses train backbone and every refinement iteration |
| Math/Lean pipeline | Fixed versioned UTF-8 byte tokenizer, streamed JSONL compiler, mmap theorem records, family-held-out splits, real Lean client | Synthetic fixture pipeline works; Lean/Mathlib not installed, real integration test skipped |
| Recoverable disk updates | Durable one-stage undo journal plus saved boundaries/cotangents/tied-gradient cursor | Six interruption boundaries recover exact weight/m/v and a subsequent step |
| Graph-aware objectives | Token + typed dependency + action + target + value teacher-forced loss | Structured resident CLI trained/evaluated on fixture traces |
| Ablations | SSM, frozen neural graph, recurrent graph component/LM timing switches | Timing harness only; no matched-compute mathematical-quality conclusion |
| Scale qualification | 100M/300M/1B size presets, exact meta count, RAM/VRAM/disk guards | Only 100M short CPU steps measured; 300M/1B/7B/5B/15B training and proof quality remain unvalidated |

## Measured CPU runs

AMD Ryzen 9 7950X, PyTorch 2.14.0+cu130, two CPU threads, FP32, batch1,
sequence8, accumulation1, one optimizer step, no warmup or statistical repeats.
Host load was not isolated. Timings exclude model/store initialization but
include first optimizer state allocation or disk transaction work.

| Path | Parameters | Time | Tokens/sec |
|---|---:|---:|---:|
| Resident reference scan | 102,402,048 | 0.730 s | 10.96 |
| Resident chunk-parallel scan | 102,402,048 | 0.654 s | 12.23 |
| Disk two-pass/journaled parallel scan | 102,402,048 | 2.424 s | 3.30 |
| Resident neural graph/refinement | 103,994,368 | 1.089 s | 7.34 |

Disk peak process RSS was approximately 0.99 GiB; graph resident RSS was
approximately 2.62 GiB. These single short measurements establish executable
100M paths, not sustained speed, a matched compute comparison, or scale trends.
The CPU SSM component and graph component were also timed separately in the
corresponding JSON files; those report component sizes, not 100M throughput.
CUDA 100M qualification was skipped because free VRAM was insufficient; other
GPU processes were not modified.

Reproduce or extend, with resource checks before allocation:

```bash
python benchmarks/scaling.py --config configs/morph_100m.json --backend reference
python benchmarks/scaling.py --config configs/morph_100m.json --backend parallel
python benchmarks/scaling.py --engine disk
python benchmarks/scaling.py --ablation static_graph
python benchmarks/scaling.py --ablation graph_recurrence
python benchmarks/scaling.py --component ssm --sequence-length 64 --steps 3
python benchmarks/scaling.py --component graph --sequence-length 16 --steps 2
```

300M/1B JSON files are **planning presets**, not trained/validated checkpoints.
No 5B/15B run configuration is described as validated.

## Structured proof workflow

```bash
python train.py --proof-traces docs/proof_examples.jsonl --steps 3 --checkpoint /tmp/proof.pt
python evaluate.py --checkpoint /tmp/proof.pt --proof-traces docs/proof_examples.jsonl
python -m data.prepare --input docs/proof_examples.jsonl --output /tmp/proof-shard
python train.py --shard /tmp/proof-shard --steps 2 --sequence-length 8
```

Proof mode supports resident FP32, batch1/accumulation1 AdamW. It uses the
configured sequence limit and rejects unsupported streaming/resume options;
the streaming stage registry does not yet include the shared proof core.
`proof_enabled` adds parameters only when explicitly requested. The existing
language model's causal ring memory remains available as a separate baseline.

The fixture has identity/truth train families and a conjunction validation
family. Three supervised steps reduced joint loss from 10.178 to 10.024;
held-out fixture loss was 9.972 and action prediction incorrect. This is a
functional pipeline check, not a generalization or proof success result.
Value labels are fixture targets, not a claim of formally verified outcomes.

The exported trajectory format includes graph/action/value/text/family/split;
`data.prepare` validates deterministic transitions and optionally invokes real
Lean for supplied candidate proofs. It consumes exported trajectories; it does
not implement LeanDojo repository tracing or extract arbitrary Lean tactics
without a toolchain. Byte tokenization is a reproducible baseline, not a trained
mathematics-specific subword tokenizer. Mapped shards validate manifest schema,
family splits, array dtype/length/vocabulary/content hash before sampling.

`verifier.LeanClient(project=..., imports=('Mathlib',))` runs a supplied existing
Lake project with a timeout, compiles an exact constructed statement with its
explicit single-line type/binder fragments, rejects declaration separators and
sorry/admit/declaration injections, and requires a
recognized [Lean axiom audit](https://lean-lang.org/doc/reference/latest/Axioms/). Accepted kernel axioms are propext, Classical.choice,
and Quot.sound. Unknown audit formats fail closed. The checker writes JSONL
receipts and records Lean version and proof hash. It is **not an OS sandbox**;
untrusted Lean execution should run in a separately isolated environment.
No Lean binary or Mathlib download was installed, and no real proof success
was claimed in this environment.

The symbolic controller's VERIFY action requests checking; only a matching
external checker receipt can promote a node to VERIFIED. Restoring verified
nodes also requires explicit checker evidence. Rollouts use joint action/target
masks and caller-provided payloads; no arbitrary topology mutation is treated
as differentiable. A verifier receipt should be applied through
`ProofGraph.accept_verification`, not by editing status labels.

## Recovery and CI

Disk transactions persist original boundaries, microbatches, clipping norm and
multiplier, stage cursor, cotangents, and tied-head gradient. Each stage's old
weight/m/v state is journaled and fsynced before mutation. A committed cursor
selects either the new complete stage or rollback of the uncommitted stage;
restart continues remaining reverse stages without reusing updated weights in
old forward computations. Completed-state cleanup is idempotent. Initialization
failures or legacy dirty stores without a transaction still fail closed.

The disk guard includes canonical 12 bytes/parameter plus largest-stage undo
state, replay/cursor windows, tied-gradient windows, and reserve. This avoids a
second whole-model checkpoint but adds disk IO and checkpoint traffic. Power
loss/file corruption outside the tested fsync protocol and arbitrary external
file edits are not claimed recoverable.

GitHub CPU CI runs correctness and structured train/evaluate smoke commands.
CUDA and real Lean integration tests explicitly skip when unavailable. A real
GPU runner and Lean/Mathlib environment still need external provisioning.
