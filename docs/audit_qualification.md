# Small-model audit qualification

Short three-commit small-model full-step qualification, not evidence of convergence or sustained 5B throughput. No 5B throughput or deadline projection.

Data: revision-pinned Hugging Face rows.

| Variant | Byte targets/s | >2,000 status | Heldout initial → final | Improved |
|---|---:|---|---|---|
| reference_two_pass | 558.6 | NOT MET | 5.54268 → 5.43275 | True |
| packed_single_pass_chunk2_group2_batched | 822.0 | NOT MET | 5.54268 → 5.43275 | True |
| large_accumulation_32768_targets_chunk2 | 3458.8 | MET | 5.54268 → 5.42390 | True |

Optimizer-step-only timing on prebuffered RAM inputs; excludes source fetch, initialization, heldout evaluation and export.
Source fetch wall time: 7.706 seconds.

Wall phases and CUDA stream intervals overlap and must not be added. Stream elapsed includes kernels and stream idle; it is not pure GPU busy time. Process I/O is process-accounted, not device-wide traffic.

tests/test_packed_state.py: multi-step partial-state-write recovery parity, truncated journal fail closed, wrong-format guards, cached recovery.

Configuration: 81,910 parameters, FP16 CUDA, CPU prefetch enabled, three commits per variant.

Controlled final weight maximum absolute difference: 0.00061552; per-parameter optimizer clocks match. This comparison includes FP16 numerical differences.

Allocated CUDA memory stayed exactly constant across all three commits within each variant:
- reference_two_pass: 67,330,048 allocated bytes; 70,194,688 peak allocated bytes.
- packed_single_pass_chunk2_group2_batched: 67,551,232 allocated bytes; 70,415,872 peak allocated bytes.
- large_accumulation_32768_targets_chunk2: 67,772,416 allocated bytes; 70,637,056 peak allocated bytes.

Model seed: 4201. Torch CPU threads: 2. Artifact written: 2026-10-21T22:14:19.224116+00:00. Exact process start timestamp was not captured for this completed trial.

A separate synthetic fallback artifact preserves a transient HF failure; it does not replace this successful revision-pinned qualification.
