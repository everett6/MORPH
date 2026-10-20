# Step-local training benchmark

Actual isolated CPU experiment, not a full 5B throughput or convergence result. All four variants started from the same FP32 random weights and processed the same synthetic byte targets. Each variant ran 3 updates, B2 × T16 × K2 = 64 targets/update (192 total). Model: 879,532 parameters, 64 experts, quota 8, FP32, parallel SSM, 2 CPU threads, seed 42. CPU: AMD Ryzen 9 7950X 16-Core Processor. Torch 2.14.0+cu130.

Temporary canonical stores lived on ext4 (`/dev/nvme0n1p5`) inside the isolated checkout, not `/tmp` tmpfs. Stores ran sequentially and were deleted; peak observed temporary footprint was 12 MiB. The measured step write counters total 205.05 MiB, excluding initialization. Timing excludes initialization and snapshot comparison.

| Variant | 3-step seconds | Byte targets/s | Active expert union per step | Physical writes MiB | Max gradient stash bytes |
|---|---:|---:|---|---:|---:|
| unrestricted_two_pass | 67.710 | 2.84 | [34, 34, 27] | 74.79 | 0 |
| quota_two_pass | 43.875 | 4.38 | [8, 8, 8] | 44.49 | 0 |
| quota_single_pass | 49.883 | 3.85 | [8, 8, 8] | 42.88 | 521904 |
| quota_single_pass_cache | 25.692 | 7.47 | [8, 8, 8] | 42.88 | 521904 |

Quota two-pass, quota single-pass, and quota single-pass+cache finished with **bitwise identical canonical FP32 weights, first moments, second moments, and conditional parameter clocks** (all maximum absolute errors 0). Unrestricted routing is a different computation, so its output is not expected to match. Every variant counted exactly 192 nonpadding target bytes; no tokens were dropped.

Cache case: 648 hits, 279 misses, 0 evictions, 4,133,904 resident bytes in a 32 MiB budget; estimated logical read payload avoided 6,548,512 bytes. Cache remains write-through and does not remove journal durability. The gradient stash was 521,904 FP32 payload bytes at peak; serialization framing adds disk bytes.

`process_io_delta.read_bytes/write_bytes` are Linux process-accounted physical I/O, not complete device traffic. `rchar/wchar` are syscall byte counters. Instrumented logical payload counters separately count requested weight reads, canonical array writes and journal array writes; they exclude metadata/cursors/stash framing and are not actual physical reads. Initial state creation is outside step measurements. OS caching means very low physical reads do not mean there was no parameter access.

Single-pass without cache was slower than quota two-pass in this run (about 12% lower throughput). The cached variant was faster, but one sequential short sample cannot isolate cache causality: fsync latency, warm filesystem pages, production training I/O and one overlapping tiny CPU test were uncontrolled. These measurements are dominated by small-model transaction/serialization overhead; there is no justified full 5B speedup, GPU throughput, loss-quality or five-day projection.

Separate verification passed on the RTX 5070: FP16 with asynchronous fetching, CPU prefetch and the bounded cache produced byte-identical canonical weights and Adam moments, with equal parameter clocks, versus two-pass updates across two changing rosters (`tests/test_single_pass_cuda.py`: 1 passed). The complete CPU suite passed with CUDA hidden: 143 passed, 19 skipped. This GPU check measures correctness, separately from the CPU timing experiment.

Reproduce from the isolated checkout:

```bash
OMP_NUM_THREADS=2 python -m benchmarks.step_local_training
OMP_NUM_THREADS=2 python -m pytest tests/test_step_local_benchmark.py -q
```

Optional CLI controls: `--config`, `--steps`, `--length`, `--batch-size`, `--accumulation`, `--quota`, `--threads`, `--output`. The harness rejects larger configurations using a conservative cumulative payload envelope; do not use it to allocate full 5B. Detailed per-step results and metadata are in `docs/step_local_benchmark.json`.
