"""One bounded host preparation worker shared by two reusable transfer slots."""
from concurrent.futures import ThreadPoolExecutor


class SlotPrefetch:
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='disk-prefetch')
        self._pending = {}
        self._closed = False

    def wait(self, slot):
        future = self._pending.pop(id(slot), None)
        if future is not None:
            return future.result()

    def submit(self, slot, function):
        if self._closed:
            raise RuntimeError('prefetch queue is closed')
        # A slot's pinned and device buffers cannot be prepared concurrently.
        self.wait(slot)
        self._pending[id(slot)] = self._executor.submit(function)

    def close(self):
        if self._closed:
            return
        self._closed = True
        failure = None
        try:
            for future in self._pending.values():
                try:
                    future.result()
                except BaseException as error:
                    if failure is None:
                        failure = error
        finally:
            self._pending.clear()
            self._executor.shutdown(wait=True)
        if failure is not None:
            raise failure
