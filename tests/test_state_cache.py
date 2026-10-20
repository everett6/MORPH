import numpy as np
import pytest
import torch

from streamtrain.disk import DiskState, durable_json
from streamtrain.state_cache import CachedDiskState


class Config:
    def to_dict(self):
        return {"name": "tiny"}


class Tiny(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))
        self.config = Config()


def create_pair(tmp_path, values=(1.0, -2.0, 0.5), cache_bytes=1024):
    model_a, model_b = Tiny(values), Tiny(values)
    a = DiskState.create(tmp_path / "plain", model_a, source=model_a)
    b = CachedDiskState.create(tmp_path / "cached", model_b, source=model_b, cache_bytes=cache_bytes)
    return model_a, model_b, a, b


def update_one(store, grad, step, clip):
    store.begin_step()
    store.update("weight", torch.tensor(grad, dtype=torch.float32), step, 0.003, clip, chunk_elements=2)
    store.finish_step()


def test_cached_updates_match_canonical_adam_across_gaps_clipping_and_eviction(tmp_path):
    _, _, plain, cached = create_pair(tmp_path, cache_bytes=36)
    # Cache only one 3-element parameter; repeated use should produce hits.
    for step, grad, clip in [(1, [2.0, -1.0, 0.5], 0.25), (2, None, 1.0), (3, [-0.3, 1.2, 5.0], 0.1)]:
        if grad is not None:
            update_one(plain, grad, step, clip)
            update_one(cached, grad, step, clip)
        else:
            assert torch.equal(plain.weight("weight"), cached.weight("weight"))
    assert cached.cache_stats["hits"] >= 1
    assert cached.cache_stats["cache_bytes"] <= 36
    torch.testing.assert_close(plain.weight("weight"), cached.weight("weight"), rtol=0, atol=0)
    assert (plain.root / plain.entries["weight"]["file"]).read_bytes() == (cached.root / cached.entries["weight"]["file"]).read_bytes()


def test_readthrough_caches_all_three_arrays_and_respects_budget(tmp_path):
    model = Tiny([1.0, 2.0, 3.0])
    store = CachedDiskState.create(tmp_path / "state", model, source=model, cache_bytes=36)
    target = torch.empty(3)
    store.read_into("weight", target)
    assert torch.equal(target, model.weight.detach())
    assert store.cache_stats["cache_bytes"] == 36
    assert store.cache_stats["misses"] == 1
    store.read_into("weight", target)
    assert store.cache_stats["hits"] == 1
    assert store.cache_stats["estimated_bytes_avoided"] == 12


def test_oversized_entry_falls_back_without_exceeding_budget(tmp_path):
    model = Tiny([1.0, 2.0, 3.0])
    store = CachedDiskState.create(tmp_path / "state", model, source=model, cache_bytes=35)
    target = torch.empty(3)
    store.read_into("weight", target)
    assert store.cache_stats["cache_bytes"] == 0
    assert store.cache_stats["oversize_fallbacks"] == 1
    assert torch.equal(target, model.weight.detach())


def test_failed_partial_cached_write_invalidates_then_recovery_restores_disk_truth(tmp_path):
    _, _, plain, cached = create_pair(tmp_path, cache_bytes=36)
    cached.weight("weight")  # populate cache
    cached.manifest["transaction"] = {"journal": None, "cursor": "cursor-0.pt"}
    durable_json(cached.root / "manifest.json", cached.manifest)
    cached.prepare_stage(0, ["weight"])
    cached.fault = lambda point: (_ for _ in ()).throw(OSError("partial")) if point == "partial_state_write" else None
    with pytest.raises(OSError, match="partial"):
        cached.update("weight", torch.tensor([1.0, 2.0, 3.0]), 1, 0.003, 1.0, chunk_elements=1)
    assert cached.cache_stats["cache_bytes"] == 0
    cached.fault = lambda point: None
    cached.recover_stage()
    restored = torch.empty(3)
    cached.read_into("weight", restored)
    torch.testing.assert_close(restored, plain.weight("weight"), rtol=0, atol=0)
    assert cached.cache_stats["misses"] >= 2


def test_lru_eviction_stays_within_configured_byte_budget(tmp_path):
    class Two(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
            self.b = torch.nn.Parameter(torch.arange(4, dtype=torch.float32) + 10)
            self.config = Config()

    model = Two()
    store = CachedDiskState.create(tmp_path / "state", model, source=model, cache_bytes=48)
    out = torch.empty(4)
    store.read_into("a", out)
    store.read_into("b", out)
    assert store.cache_stats["evictions"] == 1
    assert store.cache_stats["cache_bytes"] == 48
    store.read_into("a", out)
    assert store.cache_stats["evictions"] == 2
    assert store.cache_stats["cache_bytes"] <= 48


def test_clean_transaction_recovery_keeps_warm_cache_for_optimizer_update(tmp_path):
    model = Tiny([1.0, 2.0, 3.0])
    store = CachedDiskState.create(tmp_path / "state", model, source=model, cache_bytes=36)
    store.weight("weight")
    store.manifest["transaction"] = {"journal": None, "cursor": "cursor-0.pt"}
    durable_json(store.root / "manifest.json", store.manifest)

    store.recover_stage()
    store.update("weight", torch.tensor([0.5, -1.0, 2.0]), 1, 0.003, 1.0)

    assert store.cache_stats["cache_bytes"] == 36
    assert store.cache_stats["hits"] == 1


def test_actual_journal_rollback_invalidates_cached_entries(tmp_path):
    class Two(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
            self.b = torch.nn.Parameter(torch.arange(4, dtype=torch.float32) + 10)
            self.config = Config()

    model = Two()
    store = CachedDiskState.create(tmp_path / "state", model, source=model, cache_bytes=96)
    target = torch.empty(4)
    store.read_into("a", target)
    store.read_into("b", target)
    store.manifest["transaction"] = {"journal": None, "cursor": "cursor-0.pt"}
    durable_json(store.root / "manifest.json", store.manifest)
    store.prepare_stage(0, ["a"])

    store.recover_stage()

    assert store.cache_stats["cache_bytes"] == 0
