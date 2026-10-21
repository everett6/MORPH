# Small-model audit qualification

Short three-commit small-model full-step qualification, not evidence of convergence or sustained 5B throughput. No 5B throughput or deadline projection.

Data: synthetic fallback.

Real HF qualification failed: RuntimeError: Hugging Face request failed after 1 attempts

| Variant | Byte targets/s | >2,000 status | Heldout initial → final | Improved |
|---|---:|---|---|---|
| reference_two_pass | 567.3 | NOT MET | 5.55961 → 5.56014 | False |
| packed_single_pass_chunk2_group2_batched | 842.9 | NOT MET | 5.55961 → 5.56014 | False |
| large_accumulation_32768_targets_chunk2 | 3497.9 | MET | 5.55961 → 5.55877 | True |

Optimizer-step-only timing on prebuffered RAM inputs; excludes source fetch, initialization, heldout evaluation and export.
Source fetch wall time: 0.407 seconds.

Wall phases and CUDA stream intervals overlap and must not be added. Stream elapsed includes kernels and stream idle; it is not pure GPU busy time. Process I/O is process-accounted, not device-wide traffic.

tests/test_packed_state.py: multi-step partial-state-write recovery parity, truncated journal fail closed, wrong-format guards, cached recovery.
