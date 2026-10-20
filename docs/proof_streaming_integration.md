# Proof streaming integration: measured scope

Proof mode now follows the same registered stages in resident, synchronous DLS,
asynchronous ping-pong DLS, and disk-backed AdamW. Every token, dependency,
action, action-conditioned target and value objective trains the shared core and
backbone. Shared embedding uses are summed before global clipping/update. Disk
recovery persists structured microbatches and resumes interrupted stage updates;
small fault tests compare weights and both moments. Standard LM 7B remains
6,995,110,080 parameters; proof mode is opt-in and changes that count.

Node text uses ordered positional/convolution encoding of natural statement,
formal statement and assumptions. The configured per-field byte limit rejects
overflow. Graph-only memory conditions causal token logits; full-prompt context
is used only for auxiliary decisions. Tests cover permutation distinctions,
formal/assumption changes, future-token independence, cached generation,
padding invariance, distinct accumulated microbatches and BF16 CUDA gradients.

## Reproduce

The explicit installer is local to ignored `.tools`, pins Lean4.19.0 and Mathlib
v4.19.0 revisions/checksum, and fetches only two Mathlib module cache roots.
It requires network, git and tar with zstd support; no system installation is performed.

```bash
python -m verifier.toolchain --install
python -m data.collect_lean --limit 1000 --workers 4
python benchmarks/proof_loop.py --steps 400
python benchmarks/proof_loop.py --steps 400 --engine disk --state-dir .generated/reproduced-proof-disk --output .generated/reproduced-disk-results.json
python train.py --proof-traces docs/proof_examples.jsonl --engine async --device cuda --steps 2 --checkpoint /tmp/proof-async.pt
python train.py --proof-traces docs/proof_examples.jsonl --engine disk --state-dir .generated/proof-disk --steps 2
python train.py --proof-traces docs/proof_examples.jsonl --engine disk --state-dir .generated/proof-disk --resume-store --steps 3
python evaluate.py --checkpoint .generated/proof_loop_model.pt --search-statement 'True ∧ True' --lean-project .tools/mathlib4 --search-policy learned --attempts 3
python evaluate.py --proof-disk-state .generated/complete_proof_disk --search-statement 'True ∧ True' --lean-project .tools/mathlib4 --search-policy verify --attempts 3
python benchmarks/scaling.py --config configs/morph_100m.json --engine async --device cuda --backend triton --sequence-length 128 --steps 10
python -m pytest -q
```

`learned` uses learned action/target heads and model-generated schema-checked
JSON payloads without a provider. `verify` deliberately fixes the schedule to
VERIFY for bounded diagnostic-conditioned proof generation. Neither silently
inserts tactics. JSON permits Unicode escapes. The byte tokenizer is fixed,
not a trained mathematical tokenizer. Checkpoints validate exact data hash and
training settings; disk state must use `--resume-store`. All proof CLI engines
perform meta-only resource qualification before actual weight allocation.

## Actual evidence

On AMD Ryzen9 7950X, two CPU threads, FP32, a tiny complete model trained400
steps on six checked toy references and one seeded rejected-proof/repair
training example. Loss fell10.7453→0.006435. The model generated `by simp` for
held-out toy `True ∧ True`; real isolated Lean certified it. A learned-policy
rollout also selected VERIFY/target0 and generated that proof without a provider
(`learned_policy_search.json`). These tiny fixtures do not demonstrate
mathematical generalization. The complete disk-backed path separately trained400steps with loss10.9913→
0.005706 and bounded disk inference generated the same checked toy proof
(`complete_disk_proof_results.json`,183.58s total CPU).
The seeded training rejection is not a model
runtime rejection. In a further bounded six-goal search, five succeeded first
try and one failed all three attempts (`bounded_repair_search.json`). A natural
model rejection followed by a successful model repair remains unproven.

`verified_data_report.json` records exactly1000 individually compiled/audited
declaration-reference checks (813train/94validation/93test),205 rejected
candidates and pinned provenance. Prompts exclude target declaration names.
Splits group whitespace-normalized identical types, including exact aliases;
this is not universal semantic equivalence detection. Reference proofs may cite
an already imported theorem. These are genuine kernel checks, not novel theorem
proofs, LeanDojo traces or extracted multi-step proof-state trajectories.

The optional value-data compiler `data.search_values` derives explicitly labeled
value-only examples from observed search success, compiler seconds and attempt
counts. Static reference value1 labels remain distinct. No learned search-cost
optimization result is claimed.

Lean runs through actual rootless bwrap namespaces with no network, read-only
toolchain/project, private writable source/temp directories, external prlimit
caps and timeout/process-group termination. Unsupported namespaces fail closed.
Exact statements/assumptions use autoImplicit=false; sorry/admit and unexpected
axioms are rejected. Speculative edges cannot certify conclusions; certified
closure needs a matching checker receipt and compatible assumptions, with
cycles rejected. Original goals are immutable during search: REVISE is masked
and rejected except for conjectures/lemmas; success anchors exact original
formal statement and assumptions. This is not a general proof of sandbox security.

## GPU measurements and limits

RTX5070, PyTorch2.14.0+cu130, other GPU workloads left untouched. The100M model
has102,402,048 parameters. FP32 asynchronous Triton training, B1/T128/K1,
10steps took4.2959s (297.96tokens/s), peak allocated CUDA127.96MiB,
peak process RSS3290.83MiB, tracked H2D weights7817.64MiB and D2H gradients
3911.33MiB. The model timer includes first kernel compilation/allocation;
no separate warmup was excluded. These bytes exclude activation/checkpoint
traffic, and wall time includes transfer/optimizer work without separate phase
profiling. `100m_triton_async_gpu.json` retains exact byte totals/settings.

A separate scan primitive experiment B1/H4/P8/N16/T1024, FP32, five measured
forward+backward iterations after warmup, reports parallel0.2086s versus
Triton0.02516s (`triton_scan_gpu.json`). Triton fuses recurrence and chunk
recomputation backward; reduction atomics allow small nondeterminism. This
primitive comparison is not an architecture or end-to-end speed claim.
The earlier100M parallel model used T32 and must not be compared as matched
work. GPU numbers establish short executable paths, not sustained throughput,
quality scaling or practical full7B training. No300M/1B/5B/7B/15B training was run.

Remaining gates: multi-step real proof-state extraction and training,
independent theorem-family evaluation without imported-reference leakage,
natural rejection→repair success, broad sandbox adversarial testing,
matched-compute quality ablations and larger resource-qualified scaling.
