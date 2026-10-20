# Bounded FP32 disk-state cache

`streamtrain.state_cache.CachedDiskState` is a `DiskState`-compatible option for
benchmarking repeated CPU reads of canonical FP32 optimizer state. It keeps a
bounded LRU of complete parameter state arrays (`weight`, first moment, second
moment), with an exact capacity cost of `12 * parameter_numel` bytes per entry.
The default budget is zero, so the ordinary disk path remains the default.

```python
from streamtrain.state_cache import CachedDiskState

store = CachedDiskState.open(state_dir, model, cache_bytes=512 * 1024**2)
# Or create(..., cache_bytes=...)
```

A `read_into` miss loads the full triplet only if it fits. Entries are evicted
least-recently-used first; an entry larger than the total budget falls back to
the original disk read. `update` on an uncached parameter uses `DiskState.update`
and does not load moments just to warm the cache. On a cache hit it applies the
same chunked FP32 AdamW arithmetic, writes the updated triplet to its canonical
file, flushes and fsyncs the file before returning. There is no delayed
writeback or alternate source of truth. A write error drops the cache entry;
`recover_stage` clears the entire cache before the existing journal restoration.

`cache_stats` reports hits, misses, evictions, current/budget bytes,
oversize-fallback count, and `estimated_bytes_avoided`. The byte counter is an
exact logical estimate: a cached weight read accounts for `4*N` bytes, while a
cached optimizer update accounts for `12*N` bytes. It does not claim measured
physical I/O avoided by the operating system or storage device.

The cache is protected by a lock shared by prefetch reads and optimizer updates.
Returned `read_into` values are copies into the caller's destination; callers do
not receive mutable views of cached memory. Cache contents are process-local
and are never serialized. Reopening a clean store starts with an empty cache.
