"""Bounded read-through cache for canonical FP32 disk optimizer state."""
from collections import OrderedDict
import math
import os
from pathlib import Path
import threading

import numpy as np
import torch

from .disk import DiskState, profile_fsync


class CachedDiskState(DiskState):
    """DiskState with a bounded in-RAM LRU of complete ``[weight, m, v]`` entries.

    Canonical files remain authoritative. Cached updates are written and fsynced
    to the canonical file before returning; this class never delays writeback.
    """

    def __init__(self, root, model, manifest, cache_bytes=0):
        super().__init__(root, model, manifest)
        self._cache_lock = threading.RLock()
        self._cache = OrderedDict()
        self._cache_budget = 0
        self._cache_bytes = 0
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "estimated_bytes_avoided": 0,
            "oversize_fallbacks": 0,
        }
        self.configure_cache(cache_bytes)

    @classmethod
    def create(cls, root, model, source=None, seed=42, chunk_elements=1048576,
               progress=None, cache_bytes=0):
        disk = DiskState.create(root, model, source=source, seed=seed,
                                chunk_elements=chunk_elements, progress=progress)
        return cls(disk.root, model, disk.manifest, cache_bytes=cache_bytes)

    @classmethod
    def open(cls, root, model, cache_bytes=0):
        disk = DiskState.open(root, model)
        return cls(disk.root, model, disk.manifest, cache_bytes=cache_bytes)

    def configure_cache(self, cache_bytes):
        if not isinstance(cache_bytes, int) or isinstance(cache_bytes, bool) or cache_bytes < 0:
            raise ValueError("cache_bytes must be a non-negative integer")
        with self._cache_lock:
            self._cache.clear()
            self._cache_bytes = 0
            self._cache_budget = cache_bytes

    @property
    def cache_stats(self):
        with self._cache_lock:
            return {**self._cache_stats, "cache_bytes": self._cache_bytes,
                    "cache_budget_bytes": self._cache_budget}

    def _entry_bytes(self, name):
        return self.entries[name]["size"] * 3 * np.dtype(np.float32).itemsize

    def _load_cache_entry(self, name):
        size = self.entries[name]["size"]
        footprint = self._entry_bytes(name)
        self._cache_stats["misses"] += 1
        if self._cache_budget <= 0 or footprint > self._cache_budget:
            if self._cache_budget > 0:
                self._cache_stats["oversize_fallbacks"] += 1
            return None
        while self._cache and self._cache_bytes + footprint > self._cache_budget:
            _, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= evicted.nbytes
            self._cache_stats["evictions"] += 1
        path = self.root / self.entries[name]["file"]
        mapped = np.memmap(path, dtype=np.float32, mode="r", shape=(3, size))
        values = np.array(mapped, dtype=np.float32, copy=True, order="C")
        del mapped
        self._cache[name] = values
        self._cache_bytes += values.nbytes
        return values

    def _cached(self, name):
        values = self._cache.get(name)
        if values is None:
            return self._load_cache_entry(name)
        self._cache.move_to_end(name)
        self._cache_stats["hits"] += 1
        return values

    def read_into(self, name, destination):
        with self._cache_lock:
            was_cached = name in self._cache
            values = self._cached(name)
            if values is None:
                return super().read_into(name, destination)
            np.copyto(destination.numpy().reshape(-1), values[0], casting="unsafe")
            if was_cached:
                self._cache_stats["estimated_bytes_avoided"] += values[0].nbytes

    def update(self, name, gradient, step, lr, clip, chunk_elements=1048576):
        with self._cache_lock:
            values = self._cache.get(name)
            if values is None:
                # Do not fetch dormant moments merely to populate the cache.
                # DiskState remains the bounded, authoritative fallback.
                self._cache_stats["misses"] += 1
                return super().update(name, gradient, step, lr, clip, chunk_elements)
            self._cache.move_to_end(name)
            self._cache_stats["hits"] += 1
            self._cache_stats["estimated_bytes_avoided"] += values.nbytes
            entry = self.entries[name]
            path = self.root / entry["file"]
            try:
                mapped = np.memmap(path, dtype=np.float32, mode="r+", shape=(3, entry["size"]))
                flat_gradient = gradient.flatten()
                for start in range(0, entry["size"], chunk_elements):
                    end = min(start + chunk_elements, entry["size"])
                    weight, first, second = [torch.from_numpy(values[i, start:end].copy()) for i in range(3)]
                    g = flat_gradient[start:end] * clip
                    first.lerp_(g, .1)
                    second.mul_(.999).addcmul_(g, g, value=.001)
                    denominator = second.sqrt().div_(math.sqrt(1 - .999**step)).add_(1e-8)
                    weight.mul_(1-lr*.01).addcdiv_(first, denominator, value=-lr/(1-.9**step))
                    updated = np.stack((weight.numpy(), first.numpy(), second.numpy()))
                    values[:, start:end] = updated
                    mapped[:, start:end] = updated
                    self.fault('partial_state_write')
                mapped.flush()
                del mapped
                descriptor = os.open(path, os.O_RDWR)
                try:
                    profile_fsync(descriptor)
                finally:
                    os.close(descriptor)
            except BaseException:
                self._drop(name)
                raise

    def _drop(self, name):
        values = self._cache.pop(name, None)
        if values is not None:
            self._cache_bytes -= values.nbytes

    def recover_stage(self):
        with self._cache_lock:
            transaction = self.manifest.get("transaction")
            rollback = (transaction is not None
                        and transaction.get("journal") is not None
                        and (self.root / "journal").exists())
            if rollback:
                self._cache.clear()
                self._cache_bytes = 0
            return super().recover_stage()
