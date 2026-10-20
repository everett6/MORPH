# Separate step-local training experiment

This work lives only in `/home/everett/.codex/worktrees/delta-training-speed/MORPH`. The existing production run and its writer are unchanged. The commands below are prepared entry points; no full-size run was launched as part of this implementation.

The experimental configuration retains the full parameter bank but permits a deterministic roster of eight out of 64 experts during each optimizer step. All microbatches use that roster. The next step rotates the roster; eight consecutive steps cover the bank. This changes the routing/training policy and requires its own quality comparison. It is not numerically equivalent to unrestricted routing.

The single-pass engine computes forward boundaries and one backward pass, accumulating the same unscaled FP32 gradients and tied embedding contributions used by the existing two-pass implementation. It obtains the full global norm, applies the same clipping rule and conditional AdamW clocks, and keeps masters and Adam moments in FP32. Before changing any master, it durably stores all active gradients plus loss, norm, scaling, clipping and the roster. Recovery reads those gradients and applies journaled stage updates; it never reruns backward through partly updated weights.

A bounded optional state cache stores FP32 weight/moment entries. Disk files remain authoritative and updates are written and synchronized before returning. Cache eviction changes traffic, not optimizer arithmetic. The gradient stash and cache consume additional host RAM. The cycle refuses a run when their combined upper bound plus the existing execution envelope exceeds 80% of available RAM, when the stash budget is too small, or when disk cannot accommodate the canonical state, recovery, durable gradients and requested exports.

For the local-eight preset, the full bank contains 4,981,264,481 parameters. Its metadata-only count is 299,929,697 shared parameters plus eight bundles of 73,145,856 parameters: an upper bound of 885,096,545 active parameters, or 3,540,386,180 bytes (about 3.297GiB) of FP32 gradients. This is a conservative parameter bound, not a measured allocation or speed estimate. The default stash budget is 4GiB; larger cache budgets must also fit the combined memory check.

Run commands from the isolated worktree. First perform a preflight using a separate directory:

```bash
cd /home/everett/.codex/worktrees/delta-training-speed/MORPH
python -m streamtrain.delta_cycle \
  --config configs/morph_delta_5b_local8.json \
  --engine single-pass \
  --cache-bytes 1073741824 \
  --stash-budget-bytes 4294967296 \
  --experiment-run-dir runs/delta-local8-single-pass \
  --target-tokens 1000000000 \
  --sequence-length 128 --batch-size 4 --accumulation 8 \
  --device cuda --precision fp16 --backend triton \
  --preflight
```

Preflight constructs a model on the metadata device; it does not allocate the 5B parameter bank. It makes a bounded access check against the configured row-stream source and restores the source cursor. It does not download a model or persist a dataset cache. The example source configuration connects to Hugging Face; the implementation tests use a local fake HTTP source.

If preflight passes and resource capacity is available, the same command with `--preflight` replaced by `--max-steps 2 --no-export` starts a bounded experiment. This allocates the full FP32 disk state for the selected preset. To test the mechanics with a small model first, replace the config with `configs/morph_delta_tiny_local2.json` and use a new experiment directory.

Resume with the identical config, source, token target, precision, engine and cache/stash settings, adding `--resume`. Completed optimizer steps reconcile the pending source batch exactly once. The run lock prevents another writer for the same directory. Step-local configurations are rejected by the cycle with `--engine two-pass` until durable roster recovery for that engine is separately implemented.

The ordinary defaults remain `two-pass`, zero cache and unrestricted routing. Their settings file does not acquire experimental fields, and old settings lacking the new default-valued roster fields remain resumable. Experimental options choose a separate default directory when one is omitted; `--experiment-run-dir` makes the separation explicit. The ordinary production directory name is rejected for experimental runs.

`metrics.jsonl` adds the selected engine, routing plan, active-parameter upper bound, actual captured gradient bytes, bounded-cache counters and existing timing/I/O/transfer measurements. These measurements are required before any throughput claim. The optional FP8 transport gate previously failed the update-drift threshold; these entry points retain ordinary FP16 transfers.

CPU qualification covers clipped/tied gradients, conditional optimizer clocks, proof and language-model objectives, overflow retry, interruption recovery without backward replay, bounded token consumption, pending-source reconciliation, memory/disk refusal and old-settings compatibility. GPU throughput and learning-quality comparisons remain separate experiments.

## Current capacity snapshot

The [local-eight preflight report](delta_local8_preflight.json) estimates 13.682GB of host memory against 21.150GB available, and 7.653GB of GPU memory against 9.676GB free. Those checks pass. It requires 92.430GB of free disk for a new full-size bank and reserves, against 24.510GB free, so disk capacity blocks that allocation. These are a dated resource snapshot and conservative estimates, not measured full-size training peaks.

This blocks a new full-size bank, not the completed tiny-model benchmarks. No full 5B training throughput or convergence claim follows from those benchmarks or the preflight. No full 5B training was launched for this experiment.
