from dataclasses import replace
import pytest
from benchmarks.audit_qualification import guarded_count,difference
from morph.delta_config import DeltaConfig


def test_qualification_guard_rejects_five_billion_before_allocation():
    cfg=DeltaConfig.from_file('configs/morph_delta_5b.json')
    with pytest.raises(MemoryError,match='before allocation'):guarded_count(cfg)


def test_qualification_counter_deltas():
    assert difference({'read_bytes':120,'write_bytes':80},{'read_bytes':100,'write_bytes':30})=={'read_bytes':20,'write_bytes':50}
