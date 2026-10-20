import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from data.hf_stream import HFByteStream


DATASET_SHA = "0123456789abcdef0123456789abcdef01234567"


class DatasetAPI:
    def __init__(self, rows, *, page_size=4, revision=DATASET_SHA):
        self.rows = rows
        self.page_size = page_size
        self.revision = revision
        self.requests = []
        self.failures_remaining = 0
        self.response_delay = 0

    def handler(self):
        api = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path.startswith("/api/datasets/"):
                    payload = {"sha": DATASET_SHA}
                elif parsed.path == "/rows":
                    query = parse_qs(parsed.query)
                    offset = int(query["offset"][0])
                    length = int(query["length"][0])
                    api.requests.append((offset, length))
                    if api.response_delay:
                        time.sleep(api.response_delay)
                    if api.failures_remaining:
                        api.failures_remaining -= 1
                        self.send_response(503)
                        self.end_headers()
                        return
                    values = api.rows[offset:offset + length]
                    payload = {
                        "features": [],
                        "rows": [{"row_idx": offset + i, "row": row} for i, row in enumerate(values)],
                        "num_rows_total": len(api.rows),
                        "num_rows_per_page": api.page_size,
                        "partial": False,
                    }
                    if hasattr(api, "truncated_fields") and values:
                        payload["rows"][0]["truncated_cells"] = api.truncated_fields
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                if parsed.path == "/rows":
                    self.send_header("x-revision", api.revision)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        return Handler


class RunningAPI:
    def __init__(self, rows, **kwargs):
        self.api = DatasetAPI(rows, **kwargs)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self.api.handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        base = f"http://127.0.0.1:{self.server.server_port}"
        self.hub_url = base
        self.rows_url = base
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def stream(self, **kwargs):
        text_field = kwargs.pop("text_field", "text")
        return HFByteStream(
            "sample/math",
            "default",
            "train",
            text_field,
            revision="main",
            hub_api_url=self.hub_url,
            datasets_api_url=self.rows_url,
            **kwargs,
        )


def test_byte_windows_cross_record_boundaries_without_losing_targets():
    with RunningAPI([{"text": "ab"}, {"text": "cdefghi"}]) as api:
        stream = api.stream(page_size=1, record_separator="|")
        first = stream.next_batch(2, 3)
        stream.commit_batch()
        second = stream.next_batch(1, 3)

    windows = first.tolist() + second.tolist()
    assert [bytes(row).decode() for row in windows] == ["ab|c", "cdef", "fghi"]
    targets = b"".join(bytes(row[1:]) for row in windows).decode()
    assert targets == "b|cdefghi"


def test_uncommitted_batch_replays_exactly_after_checkpoint_restore():
    with RunningAPI([{"text": "alpha"}, {"text": "beta"}, {"text": "gamma"}]) as api:
        stream = api.stream(page_size=2)
        delivered = stream.next_batch(2, 3)
        checkpoint = json.loads(json.dumps(stream.state_dict()))

        resumed = api.stream(page_size=2)
        resumed.load_state_dict(checkpoint)
        replay = resumed.next_batch(2, 3)
        assert replay.tolist() == delivered.tolist()

        resumed.commit_batch()
        next_batch = resumed.next_batch(1, 3)
        stream.commit_batch()
        assert stream.next_batch(1, 3).tolist() == next_batch.tolist()


def test_revision_mismatch_is_rejected_before_any_batch_is_returned():
    with RunningAPI([{"text": "abcd"}], revision="a-different-commit") as api:
        stream = api.stream(page_size=1)
        with pytest.raises(ValueError, match="revision"):
            stream.next_batch(1, 2)


def test_transient_http_error_retries_with_a_finite_attempt_count():
    with RunningAPI([{"text": "abcd"}]) as api:
        api.api.failures_remaining = 1
        stream = api.stream(page_size=1, retries=1, retry_backoff=0)
        assert stream.next_batch(1, 2).tolist() == [[97, 98, 99]]
        assert api.api.requests == [(0, 1), (0, 1)]


def test_retry_limit_is_enforced_instead_of_retrying_forever():
    with RunningAPI([{"text": "abcd"}]) as api:
        api.api.failures_remaining = 10
        stream = api.stream(page_size=1, retries=2, retry_backoff=0)
        with pytest.raises(RuntimeError, match="after 3 attempts"):
            stream.next_batch(1, 2)
        assert len(api.api.requests) == 3


def test_request_timeout_is_bounded_and_retried_only_to_the_limit():
    with RunningAPI([{"text": "abcd"}]) as api:
        api.api.response_delay = 0.1
        stream = api.stream(page_size=1, timeout=0.01, retries=1, retry_backoff=0)
        with pytest.raises(RuntimeError, match="after 2 attempts"):
            stream.next_batch(1, 2)
        assert len(api.api.requests) == 2


def test_exhaustion_is_explicit_and_never_repeats_the_dataset():
    with RunningAPI([{"text": "abc"}]) as api:
        stream = api.stream(page_size=1)
        assert stream.next_batch(1, 2).tolist() == [[97, 98, 99]]
        stream.commit_batch()
        with pytest.raises(StopIteration):
            stream.next_batch(1, 2)
        assert api.api.requests == [(0, 1)]


def test_short_final_batch_preserves_complete_windows_and_terminal_byte_carry():
    with RunningAPI([{"text": "abcd"}]) as api:
        stream = api.stream(page_size=1)
        final = stream.next_batch(2, 3)
        assert final.shape == (1, 4)
        assert final.tolist() == [[97, 98, 99, 100]]
        checkpoint = json.loads(json.dumps(stream.state_dict()))
        resumed = api.stream(page_size=1)
        resumed.load_state_dict(checkpoint)
        assert resumed.next_batch(2, 3).tolist() == [[97, 98, 99, 100]]
        resumed.commit_batch()
        with pytest.raises(StopIteration):
            resumed.next_batch(2, 3)
        assert api.api.requests == [(0, 1)]
        assert base64.b64decode(resumed.state_dict()["committed"]["buffer_b64"]) == b"d\n"


def test_oversized_response_page_is_refused_to_preserve_memory_bound():
    with RunningAPI([{"text": "x" * 3000}]) as api:
        stream = api.stream(page_size=1, max_response_bytes=512)
        with pytest.raises(ValueError, match="response exceeds"):
            stream.next_batch(1, 2)


def test_truncated_configured_text_field_is_rejected():
    with RunningAPI([{"text": "abcd"}]) as api:
        api.api.truncated_fields = ["text"]
        stream = api.stream(page_size=1)
        with pytest.raises(ValueError, match="truncated"):
            stream.next_batch(1, 2)


def test_state_from_a_different_dataset_fingerprint_is_rejected():
    with RunningAPI([{"text": "abcd"}]) as api:
        original = api.stream(page_size=1)
        original.next_batch(1, 2)
        state = original.state_dict()
        other = api.stream(page_size=1, text_field="other")
        with pytest.raises(ValueError, match="fingerprint"):
            other.load_state_dict(state)
