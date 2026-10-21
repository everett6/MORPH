# Optional execution precision gate

Three-step small-model numerical accuracy gate only; no convergence, full5B, or throughput inference.

Data: revision-pinned Hugging Face rows. FP16 CUDA compute with FP32 canonical weights and Adam moments.

sqrt(sum((W_reference-W_optimized)^2)/sum((W_reference-W_initial)^2)), over all canonical parameters after each committed step.

| Group size | Step | Relative update error | First moment error | Second moment error | Gate |
|---|---:|---:|---:|---:|---|
| 1 | 1 | 0.018312 | 0.000122 | 0.000189 | PASS |
| 1 | 2 | 0.011603 | 0.000105 | 0.000111 | PASS |
| 1 | 3 | 0.008927 | 0.000103 | 0.000122 | PASS |
| 2 | 1 | 0.018312 | 0.000122 | 0.000189 | PASS |
| 2 | 2 | 0.011603 | 0.000105 | 0.000111 | PASS |
| 2 | 3 | 0.008927 | 0.000103 | 0.000122 | PASS |

Limits: update error ≤10%, loss drift ≤2%, global gradient norm drift ≤10%; finite state and matching per-parameter clocks are required. Moment errors are reported separately.

Overall gate: PASS. Failure keeps the affected optimization optional and calls for the reference fallback.
