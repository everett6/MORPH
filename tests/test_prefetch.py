"""Prefetch must overlap work without overwriting a slot still preparing."""
import threading
from types import SimpleNamespace
import pytest
from streamtrain.prefetch import SlotPrefetch


def test_prefetch_returns_before_load_and_reuse_waits_for_same_slot():
    entered = threading.Event(); finish = threading.Event(); second = threading.Event()
    slot = SimpleNamespace()
    queue = SlotPrefetch()
    def load():
        entered.set()
        assert finish.wait(5)
        slot.value = 7
    queue.submit(slot, load)
    assert entered.wait(5)
    assert not hasattr(slot, 'value')  # caller can compute while load blocks
    worker = threading.Thread(target=lambda: (queue.submit(slot, lambda: setattr(slot, 'value', 9)), second.set()))
    worker.start()
    assert not second.wait(.05)  # reusing pinned buffers must wait
    finish.set(); worker.join(5)
    queue.wait(slot)
    assert slot.value == 9
    queue.close()


def test_prefetch_failure_propagates_and_shutdown_joins():
    queue = SlotPrefetch(); slot = SimpleNamespace()
    def fail(): raise OSError('read failed')
    queue.submit(slot, fail)
    with pytest.raises(OSError, match='read failed'): queue.wait(slot)
    queue.close()
    with pytest.raises(RuntimeError, match='closed'): queue.submit(slot, lambda: None)


def test_disk_prefetch_matches_synchronous_weights(tmp_path):
    import torch
    from morph import MorphConfig, MorphModel
    from streamtrain.disk import DiskDLS, DiskState
    config = MorphConfig(vocab_size=8,d_model=8,num_layers=1,d_ff=16,d_state=2,head_dim=4)
    source = MorphModel(config)
    with torch.device('meta'): template = MorphModel(config)
    store = DiskState.create(tmp_path/'state', template, source=source)
    trainer = DiskDLS(template, store, cpu_prefetch=True)
    try:
        trainer.load(0, trainer.slots[0])
        parameters = trainer.parameters(0, trainer.slots[0], False)
        torch.testing.assert_close(parameters['module.weight'], source.embedding.weight)
    finally: trainer.close()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_prefetched_adam_step_matches_synchronous_transaction(device, tmp_path):
    import torch
    from morph import MorphConfig, MorphModel
    from streamtrain.disk import DiskDLS, DiskState
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    config = MorphConfig(vocab_size=8,d_model=8,num_layers=2,d_ff=16,d_state=2,head_dim=4)
    source = MorphModel(config)
    with torch.device('meta'): template = MorphModel(config)
    a = DiskState.create(tmp_path/'prefetched', template, source=source)
    b = DiskState.create(tmp_path/'sync', template, source=source)
    prefetched = DiskDLS(template,a,device=device,async_mode=device=='cuda',cpu_prefetch=True)
    synchronous = DiskDLS(template,b,device=device,async_mode=device=='cuda')
    try:
        batches = [torch.tensor([[1,2,3,4]])]
        result = prefetched.step(batches)
        reference = synchronous.step(batches)
        assert result == reference
        for name in a.entries: torch.testing.assert_close(a.weight(name), b.weight(name), rtol=0,atol=0)
    finally:
        prefetched.close(); synchronous.close()
