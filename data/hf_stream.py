"""Bounded, resumable UTF-8 byte streaming from the Hugging Face datasets API."""

import base64
import copy
import hashlib
import json
import math
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import torch

from .tokenizer import ByteTokenizer


_FORMAT_VERSION = 1
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class HFByteStream:
    """Stream dataset text as byte-token next-token batches.

    A returned batch remains pending until :meth:`commit_batch` is called.
    Save :meth:`state_dict` alongside the training checkpoint: if a process
    stops before committing, restoring the state replays that exact batch.
    The datasets-server revision response header is checked on every row page
    so a moving branch cannot silently change the source during a run.
    """

    def __init__(
        self,
        repo,
        config,
        split,
        text_field,
        revision,
        *,
        start_cursor=0,
        page_size=1,
        max_buffer_bytes=4 * 1024 * 1024,
        max_response_bytes=8 * 1024 * 1024,
        max_pending_bytes=16 * 1024 * 1024,
        retries=3,
        timeout=20.0,
        retry_backoff=0.25,
        record_separator="\n",
        hub_api_url="https://huggingface.co",
        datasets_api_url="https://datasets-server.huggingface.co",
    ):
        for name, value in (("repo", repo), ("config", config), ("split", split), ("text_field", text_field), ("revision", revision)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty explicit string")
        if not isinstance(start_cursor, int) or isinstance(start_cursor, bool) or start_cursor < 0:
            raise ValueError("start_cursor must be a non-negative row offset")
        if not isinstance(page_size, int) or isinstance(page_size, bool) or not 1 <= page_size <= 100:
            raise ValueError("page_size must be an integer from 1 through 100")
        for name, value in (("max_buffer_bytes", max_buffer_bytes), ("max_response_bytes", max_response_bytes), ("max_pending_bytes", max_pending_bytes)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(retries, int) or isinstance(retries, bool) or not 0 <= retries <= 10:
            raise ValueError("retries must be between 0 and 10")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a finite positive number")
        if not math.isfinite(retry_backoff) or retry_backoff < 0:
            raise ValueError("retry_backoff must be finite and non-negative")
        if not isinstance(record_separator, str):
            raise ValueError("record_separator must be a string")
        if not hub_api_url.startswith("https://") and not hub_api_url.startswith("http://127.0.0.1"):
            raise ValueError("hub_api_url must use HTTPS (localhost is allowed for tests)")
        if not datasets_api_url.startswith("https://") and not datasets_api_url.startswith("http://127.0.0.1"):
            raise ValueError("datasets_api_url must use HTTPS (localhost is allowed for tests)")

        self.repo, self.config, self.split, self.text_field = repo, config, split, text_field
        self.requested_revision = revision
        self.start_cursor = start_cursor
        self.page_size = page_size
        self.max_buffer_bytes = max_buffer_bytes
        self.max_response_bytes = max_response_bytes
        self.max_pending_bytes = max_pending_bytes
        self.retries, self.timeout, self.retry_backoff = retries, timeout, retry_backoff
        self.separator = record_separator.encode("utf-8")
        self.hub_api_url = hub_api_url.rstrip("/")
        self.datasets_api_url = datasets_api_url.rstrip("/")

        self.resolved_revision = self._resolve_revision()
        spec = {
            "schema": _FORMAT_VERSION,
            "repo": repo,
            "config": config,
            "split": split,
            "text_field": text_field,
            "revision": self.resolved_revision,
            "record_separator_utf8_hex": self.separator.hex(),
            "tokenizer": ByteTokenizer.fingerprint,
        }
        self.source_fingerprint = hashlib.sha256(
            json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._committed = self._empty_cursor(start_cursor)
        self._pending = None

    def _empty_cursor(self, offset):
        return {"next_offset": offset, "total_rows": None, "buffer": b"", "queued_rows": [], "eof": False}

    def _request_json(self, url, *, expected_revision=None):
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                request = Request(url, headers={"User-Agent": "MORPH-HFByteStream/1.0"})
                with urlopen(request, timeout=self.timeout) as response:
                    declared = response.headers.get("Content-Length")
                    if declared is not None and int(declared) > self.max_response_bytes:
                        raise ValueError(f"Hugging Face response exceeds {self.max_response_bytes} bytes")
                    body = response.read(self.max_response_bytes + 1)
                    if len(body) > self.max_response_bytes:
                        raise ValueError(f"Hugging Face response exceeds {self.max_response_bytes} bytes")
                    response_revision = response.headers.get("x-revision")
                    if expected_revision is not None and response_revision != expected_revision:
                        raise ValueError(
                            "datasets-server did not serve the pinned revision "
                            f"{expected_revision}; response x-revision was {response_revision!r}"
                        )
                try:
                    return json.loads(body), response_revision
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError("Hugging Face returned invalid JSON") from exc
            except ValueError:
                raise
            except HTTPError as exc:
                last_error = exc
                if exc.code not in (408, 425, 429) and not 500 <= exc.code <= 599:
                    raise RuntimeError(f"Hugging Face request failed with HTTP {exc.code}") from exc
            except (URLError, TimeoutError, OSError) as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(self.retry_backoff * (2 ** attempt))
        raise RuntimeError(f"Hugging Face request failed after {self.retries + 1} attempts") from last_error

    def _resolve_revision(self):
        repo_path = quote(self.repo, safe="/")
        revision_path = quote(self.requested_revision, safe="")
        url = f"{self.hub_api_url}/api/datasets/{repo_path}/revision/{revision_path}"
        payload, _ = self._request_json(url)
        sha = payload.get("sha") if isinstance(payload, dict) else None
        if not isinstance(sha, str) or not _SHA_RE.fullmatch(sha):
            raise ValueError("Hugging Face did not resolve the requested dataset revision to a commit SHA")
        return sha

    def _fetch_page(self, cursor):
        total = cursor["total_rows"]
        if total is not None and cursor["next_offset"] >= total:
            cursor["eof"] = True
            return
        params = urlencode({
            "dataset": self.repo,
            "config": self.config,
            "split": self.split,
            "offset": cursor["next_offset"],
            "length": self.page_size,
            "revision": self.resolved_revision,
        })
        url = f"{self.datasets_api_url}/rows?{params}"
        payload, _ = self._request_json(url, expected_revision=self.resolved_revision)
        if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
            raise ValueError("datasets-server rows response has an invalid schema")
        total_rows = payload.get("num_rows_total")
        if not isinstance(total_rows, int) or total_rows < 0:
            raise ValueError("datasets-server did not report a valid num_rows_total")
        if cursor["total_rows"] is not None and cursor["total_rows"] != total_rows:
            raise ValueError("dataset row count changed during this pinned stream")
        cursor["total_rows"] = total_rows
        if cursor["next_offset"] > total_rows:
            raise ValueError("start cursor exceeds the dataset row count")
        rows = payload["rows"]
        if not rows:
            if cursor["next_offset"] == total_rows:
                cursor["eof"] = True
                return
            raise RuntimeError("datasets-server returned an empty page before the pinned split ended")
        if len(rows) > self.page_size:
            raise ValueError("datasets-server returned more rows than requested")
        queued = []
        for index, item in enumerate(rows):
            if not isinstance(item, dict) or item.get("row_idx") != cursor["next_offset"] + index:
                raise ValueError("datasets-server row indexes are missing or discontinuous")
            truncated_cells = item.get("truncated_cells", [])
            if not isinstance(truncated_cells, list):
                raise ValueError("datasets-server returned an invalid truncated_cells marker")
            if self.text_field in truncated_cells:
                raise ValueError(
                    f"datasets-server truncated configured text field {self.text_field!r} "
                    f"in row {item['row_idx']}"
                )
            row = item.get("row")
            if not isinstance(row, dict) or self.text_field not in row:
                raise ValueError(f"row is missing configured text field {self.text_field!r}")
            value = row[self.text_field]
            if not isinstance(value, str):
                raise ValueError(f"row field {self.text_field!r} must contain text")
            queued.append(value)
        cursor["next_offset"] += len(rows)
        cursor["queued_rows"].extend(queued)

    def _append_next_record(self, cursor):
        if not cursor["queued_rows"]:
            if cursor["eof"]:
                return False
            self._fetch_page(cursor)
            if not cursor["queued_rows"]:
                return False
        text = cursor["queued_rows"].pop(0)
        encoded = text.encode("utf-8") + self.separator
        if len(cursor["buffer"]) + len(encoded) > self.max_buffer_bytes:
            raise ValueError(
                f"record exceeds the {self.max_buffer_bytes}-byte stream buffer; "
                "increase max_buffer_bytes or use a dataset with shorter records"
            )
        cursor["buffer"] += encoded
        return True

    def _window(self, cursor, length):
        if length + 1 > self.max_buffer_bytes:
            raise ValueError("one byte window exceeds max_buffer_bytes")
        while len(cursor["buffer"]) < length + 1:
            if not self._append_next_record(cursor):
                return None
        result = cursor["buffer"][:length + 1]
        cursor["buffer"] = cursor["buffer"][length:]
        return result

    @staticmethod
    def _encode_bytes(value):
        return base64.b64encode(value).decode("ascii")

    @staticmethod
    def _decode_bytes(value, label):
        if not isinstance(value, str):
            raise ValueError(f"invalid {label} in HF stream state")
        try:
            return base64.b64decode(value, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise ValueError(f"invalid {label} in HF stream state") from exc

    def _cursor_to_json(self, cursor):
        return {
            "next_offset": cursor["next_offset"],
            "total_rows": cursor["total_rows"],
            "buffer_b64": self._encode_bytes(cursor["buffer"]),
            "queued_rows": list(cursor["queued_rows"]),
            "eof": cursor["eof"],
        }

    def _cursor_from_json(self, value):
        if not isinstance(value, dict):
            raise ValueError("invalid cursor in HF stream state")
        next_offset, total_rows, queued_rows, eof = (
            value.get("next_offset"), value.get("total_rows"), value.get("queued_rows"), value.get("eof")
        )
        if not isinstance(next_offset, int) or isinstance(next_offset, bool) or next_offset < self.start_cursor:
            raise ValueError("invalid row offset in HF stream state")
        if total_rows is not None and (not isinstance(total_rows, int) or total_rows < next_offset):
            raise ValueError("invalid total row count in HF stream state")
        if not isinstance(queued_rows, list) or not all(isinstance(row, str) for row in queued_rows):
            raise ValueError("invalid buffered rows in HF stream state")
        if not isinstance(eof, bool):
            raise ValueError("invalid end-of-stream flag in HF stream state")
        buffer = self._decode_bytes(value.get("buffer_b64"), "byte carry")
        queued_bytes = sum(len(row.encode("utf-8")) + len(self.separator) for row in queued_rows)
        if len(buffer) > self.max_buffer_bytes or queued_bytes > self.max_response_bytes:
            raise ValueError("HF stream state exceeds configured memory bounds")
        return {"next_offset": next_offset, "total_rows": total_rows, "buffer": buffer, "queued_rows": list(queued_rows), "eof": eof}

    def state_dict(self):
        """Return JSON-serializable cursor state, including an uncommitted batch."""
        state = {
            "format_version": _FORMAT_VERSION,
            "source_fingerprint": self.source_fingerprint,
            "start_cursor": self.start_cursor,
            "committed": self._cursor_to_json(self._committed),
            "pending": None,
        }
        if self._pending is not None:
            state["pending"] = {
                "batch_size": self._pending["batch_size"],
                "requested_batch_size": self._pending["requested_batch_size"],
                "length": self._pending["length"],
                "batch_b64": self._encode_bytes(self._pending["batch_bytes"]),
                "after": self._cursor_to_json(self._pending["after"]),
            }
        return state

    def load_state_dict(self, state):
        """Restore a state produced for this exact dataset fingerprint."""
        if not isinstance(state, dict) or state.get("format_version") != _FORMAT_VERSION:
            raise ValueError("unsupported HF stream state format")
        if state.get("source_fingerprint") != self.source_fingerprint:
            raise ValueError("HF stream source fingerprint does not match the configured dataset")
        if state.get("start_cursor") != self.start_cursor:
            raise ValueError("HF stream start cursor does not match saved state")
        committed = self._cursor_from_json(state.get("committed"))
        pending_value = state.get("pending")
        pending = None
        if pending_value is not None:
            batch_size = pending_value.get("batch_size")
            requested_batch_size = pending_value.get("requested_batch_size", batch_size)
            length = pending_value.get("length")
            if any(not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in (batch_size, requested_batch_size, length)):
                raise ValueError("invalid pending batch shape in HF stream state")
            if batch_size > requested_batch_size:
                raise ValueError("invalid pending batch size in HF stream state")
            batch_bytes = self._decode_bytes(pending_value.get("batch_b64"), "pending batch")
            expected_size = batch_size * (length + 1)
            if expected_size * 8 > self.max_pending_bytes or len(batch_bytes) != expected_size:
                raise ValueError("pending batch exceeds configured memory bound or has an invalid size")
            pending = {
                "batch_size": batch_size,
                "requested_batch_size": requested_batch_size,
                "length": length,
                "batch_bytes": batch_bytes,
                "after": self._cursor_from_json(pending_value.get("after")),
            }
        self._committed, self._pending = committed, pending

    def next_batch(self, batch_size, length):
        """Return CPU int64 byte-token rows of shape ``[B, length + 1]``.

        The one-token overlap between adjacent windows preserves every
        next-token target in the concatenated UTF-8 record stream.
        """
        for name, value in (("batch_size", batch_size), ("length", length)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if batch_size * (length + 1) * 8 > self.max_pending_bytes:
            raise ValueError("requested batch exceeds max_pending_bytes")
        if self._pending is not None:
            if (self._pending["requested_batch_size"], self._pending["length"]) != (batch_size, length):
                raise ValueError("an uncommitted batch exists with a different shape; commit or restore it first")
            return torch.frombuffer(bytearray(self._pending["batch_bytes"]), dtype=torch.uint8).to(torch.long).reshape(
                self._pending["batch_size"], length + 1
            )

        after = copy.deepcopy(self._committed)
        windows = []
        for _ in range(batch_size):
            window = self._window(after, length)
            if window is None:
                break
            windows.append(window)
        if not windows:
            # Commit terminal discovery so repeated calls do not refetch the
            # last page. Any incomplete tail remains visible in cursor state.
            self._committed = after
            raise StopIteration("dataset has no complete next-token window remaining")
        batch_bytes = b"".join(windows)
        actual_batch_size = len(windows)
        self._pending = {
            "batch_size": actual_batch_size,
            "requested_batch_size": batch_size,
            "length": length,
            "batch_bytes": batch_bytes,
            "after": after,
        }
        return torch.frombuffer(bytearray(batch_bytes), dtype=torch.uint8).to(torch.long).reshape(
            actual_batch_size, length + 1
        )

    def commit_batch(self):
        """Advance the durable cursor after the caller completes an optimizer step."""
        if self._pending is None:
            raise RuntimeError("there is no pending HF batch to commit")
        self._committed = self._pending["after"]
        self._pending = None
