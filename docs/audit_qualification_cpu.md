# Small-model audit qualification

Measured small-model full optimizer steps only; no 5B throughput or deadline projection.

Data: revision-pinned Hugging Face rows.

| Variant | Byte targets/s | >2,000 status | Heldout initial → final | Improved |
|---|---:|---|---|---|
| reference_two_pass | 993.5 | NOT MET | 5.54268 → 5.43275 | True |
| packed_single_pass_chunk2_group2_batched | 984.2 | NOT MET | 5.54268 → 5.43275 | True |
| large_accumulation_32768_targets_chunk2 | 8170.5 | MET | 5.54268 → 5.42385 | True |

Wall phases and CUDA stream intervals overlap and must not be added. Stream elapsed includes kernels and stream idle; it is not pure GPU busy time. Process I/O is process-accounted, not device-wide traffic.

tests/test_packed_state.py: multi-step partial-state-write recovery parity, truncated journal fail closed, wrong-format guards, cached recovery.

Earlier CPU trial: telemetry predates expert alias counters. Timing covers prebuffered optimizer steps and excludes source fetch, initialization, evaluation and export. This is a short small-model trial and provides no sustained 5B throughput or convergence evidence.
