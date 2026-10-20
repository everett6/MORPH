# Existing live run: read-only profile

Snapshot: 2026-10-20T05:05:23.821547+00:00. The original run was not stopped or reconfigured.

| Committed step | Cumulative byte targets | Seconds | Targets/s | Active experts | Kernel read GB | Kernel write GB |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 4,096 | 482.28 | 8.493 | 64 | 77.63 | 121.42 |
| 2 | 8,192 | 527.50 | 7.765 | 64 | 67.00 | 121.43 |
| 3 | 12,288 | 581.74 | 7.041 | 64 | 67.19 | 121.43 |

A separate 30-sample GPU window during optimizer work averaged 2.53% utilization (maximum 7%), using about 2566 MiB. The preceding NVMe window averaged 82.70% device utilization, 103.47 ms write wait, and 109.89 queued requests. These partial, separate windows support investigating storage costs; they do not constitute a full-step GPU/CPU/SSD trace.

Raw numeric summaries and committed metrics are retained in [live_training_profile.json](live_training_profile.json). No new training-state writer or dataset cache was added to the original run.

## Capacity arithmetic for the isolated eight-expert experiment

The full model remains 4,981,264,481 parameters. Shared parameters plus eight complete expert bundles give an upper bound of 885,096,545 active parameters per step: 3.297 GiB for an FP32 gradient stash and 9.892 GiB for all three active FP32 optimizer arrays. Canonical full-model FP32 weights and moments still occupy 59.775 GB. Inactive expert parameters are retained, not deleted.

The ratio of total parameters to this active bound is 5.63×. It is a parameter-traffic ratio, not an achieved training speedup. Exact conditional AdamW still writes updated canonical state and recovery journals; the cache does not remove those writes. Persisted FP32 gradient storage adds its own writes.

The rotating roster is content independent to preserve causal language modeling. It is not the batch-score TopK proposal or a learned graph-conditioned scheduler. Every training target is retained, but expert update frequency changes, so convergence and held-out quality still require separate evaluation.
