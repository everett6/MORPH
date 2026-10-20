import json

import pytest
import torch

from morph.delta_config import DeltaConfig
from morph.model.delta import DeltaModel
from streamtrain.disk import DiskState
from streamtrain.single_pass import SinglePassDiskDLS


def pending_transaction(tmp_path):
    cfg = DeltaConfig(d_model=8, num_layers=1, d_ff=16, d_state=2, head_dim=4,
                      d_graph=8, graph_nodes=3, memory_dim=4, expert_count=3,
                      expert_depth=1, expert_ffn=16, latent_dim=4, proof_enabled=False,
                      fast_memory_enabled=False, step_expert_count=1)
    torch.manual_seed(8)
    source = DeltaModel(cfg)
    with torch.device("meta"):
        model = DeltaModel(cfg)
    store = DiskState.create(tmp_path / "state", model, source=source)
    trainer = SinglePassDiskDLS(model, store, stash_budget_bytes=1_000_000)
    batches = [torch.tensor([[1, 2, 3, 4]])]

    def stop_after_stash(point):
        if point == "stored_transaction_durable":
            raise OSError("stop after durable transaction")

    store.fault = stop_after_stash
    with pytest.raises(OSError, match="durable transaction"):
        trainer.step(batches)
    trainer.close()
    return model, store


def mutate_metadata(store, mutation):
    path = store.root / "gradient-stash" / "metadata.json"
    metadata = json.loads(path.read_text())
    mutation(metadata)
    path.write_text(json.dumps(metadata))


def test_recovery_rejects_gradient_assigned_to_invalid_stage_before_mutation(tmp_path):
    model, store = pending_transaction(tmp_path)
    before = {name: (store.root / entry["file"]).read_bytes() for name, entry in store.entries.items()}

    def corrupt(metadata):
        names = [name for stage_names in metadata["stages"].values() for name in stage_names]
        metadata["stages"] = {"999": names}

    mutate_metadata(store, corrupt)
    resumed = SinglePassDiskDLS(model, store)
    try:
        with pytest.raises(ValueError, match="stage"):
            resumed.resume_step()
        assert {name: (store.root / entry["file"]).read_bytes() for name, entry in store.entries.items()} == before
        assert store.manifest["dirty"]
        assert store.manifest["transaction"]
    finally:
        resumed.close()


def test_recovery_rejects_gradient_assigned_to_wrong_parameter_stage(tmp_path):
    model, store = pending_transaction(tmp_path)
    before = {name: (store.root / entry["file"]).read_bytes() for name, entry in store.entries.items()}
    trainer = SinglePassDiskDLS(model, store)
    owners = {}
    for index, spec in enumerate(trainer.all_specs):
        for _, parameter in spec:
            owners[trainer.canonical[id(parameter)]] = index
    trainer.close()

    def corrupt(metadata):
        source_stage, source_name = next(
            (int(index), name)
            for index, names in metadata["stages"].items()
            for name in names
            if any(other != int(index) for other in owners.values())
        )
        target_stage = next(index for index in range(metadata["stage_count"]) if index != owners[source_name])
        metadata["stages"][str(source_stage)].remove(source_name)
        if not metadata["stages"][str(source_stage)]:
            del metadata["stages"][str(source_stage)]
        metadata["stages"].setdefault(str(target_stage), []).append(source_name)

    mutate_metadata(store, corrupt)
    resumed = SinglePassDiskDLS(model, store)
    try:
        with pytest.raises(ValueError, match="stage"):
            resumed.resume_step()
        assert {name: (store.root / entry["file"]).read_bytes() for name, entry in store.entries.items()} == before
        assert store.manifest["dirty"]
    finally:
        resumed.close()
