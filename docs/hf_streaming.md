# Bounded Hugging Face text streaming

`data.hf_stream.HFByteStream` fetches small pages from the Hugging Face
datasets-server `/rows` API with Python's standard library. It does not install
`datasets`/`pyarrow`, download Parquet shards, or write a source-data cache. The
only training data retained is a bounded response page, a byte carry, and at
most one pending token batch. Its output is a CPU `torch.int64` tensor with
shape `[batch_size, sequence_length + 1]`, ready for the existing causal LM
loss (`logits[:, :-1]` vs. `tokens[:, 1:]`).

## Verified source

The default research source has been checked against the live Hugging Face API:

| Setting | Value |
|---|---|
| Dataset repo | `open-web-math/open-web-math` |
| Config | `default` |
| Split | `train` |
| Text field | `text` |
| Branch resolved during this check | `fde8ef8de2300f5e778f56261843dab89f230815` |
| Dataset size reported by API | 6,315,233 rows |

The first page was streamed in memory to verify the field and response format.
The source fingerprint is computed from the repo, config, split, text field,
resolved commit SHA, separator, and tokenizer schema. Each rows response must
return the same SHA in its `x-revision` header. If the server ignores the
requested pin or serves another revision, the stream fails before returning a
batch. Pass a commit SHA directly for reproducible reruns; a branch name is
resolved to its current commit when a new stream is constructed.

## Use

```python
from data.hf_stream import HFByteStream

source = HFByteStream(
    repo="open-web-math/open-web-math",
    config="default",
    split="train",
    text_field="text",
    revision="fde8ef8de2300f5e778f56261843dab89f230815",
    start_cursor=0,
    page_size=1,
    max_buffer_bytes=4 * 1024 * 1024,
    max_response_bytes=8 * 1024 * 1024,
    max_pending_bytes=16 * 1024 * 1024,
    retries=3,
    timeout=20.0,
)

batch = source.next_batch(batch_size=2, length=1024)
# Compute loss and complete optimizer.step() with `batch`.
source.commit_batch()
```

Each text field is UTF-8 encoded, and the configured record separator (newline
by default) is appended after each row. Adjacent windows overlap by one byte:
the first row contains input bytes 0…L and the next begins at byte L. Therefore
the next-token targets cover every byte in the concatenated record stream
without dropping or repeating a target at window or row boundaries. A final
batch may contain fewer rows than requested; this occurs only when the pinned
split ends. Any terminal carry too short to form a full target window remains
in the saved cursor rather than causing a refetch or silent wraparound.

## Checkpoint and resume

`state_dict()` returns JSON-safe data containing the source fingerprint, row
cursor, bounded byte carry, any queued rows already fetched in the current
page, and an uncommitted batch if one exists. A pending batch is replayed
exactly after restore. Commit only after the optimizer step succeeds:

```python
batch = source.next_batch(batch_size, sequence_length)
loss = train_step(batch)
source.commit_batch()
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "stream": source.state_dict(),
}
```

Save model, optimizer, and stream state together as one consistent training
checkpoint. If the process stops before committing, a checkpoint containing
the pending batch replays it; if it stops after commit, the checkpoint cursor
starts after that batch. Do not save model/optimizer state from after the update
with stream state from before it.

```python
source = HFByteStream(
    repo="open-web-math/open-web-math", config="default", split="train",
    text_field="text", revision="fde8ef8de2300f5e778f56261843dab89f230815",
)
source.load_state_dict(checkpoint["stream"])
```

The constructor must use the same explicit dataset selection and starting row
cursor. A different source fingerprint, changed commit, malformed state,
oversized buffered state, or a server response without the requested pinned
revision is rejected.

## Memory and failure behavior

- `page_size` is limited to 1–100; the default is one row. `max_response_bytes`
  hard-limits each HTTP response before JSON decoding. `max_buffer_bytes`
  limits token carry plus the next record. `max_pending_bytes` limits the
  returned int64 batch. These bounds make RAM usage predictable; Python JSON
  object overhead sits on top of the response-byte limit.
- Requests use a finite timeout and a capped retry count with exponential
  backoff for transient network errors and HTTP 408/425/429/5xx. Oversized
  responses, missing text fields, discontinuous row indexes, changing split
  sizes, revision mismatch, and invalid data fail with an explicit error.
- `StopIteration` means there is no complete next-token window left. The
  dataset never loops back to its beginning. Any incomplete terminal bytes
  stay in cursor state for inspection; the caller can use that carry to decide
  whether to request a shorter final sequence length.
- `hub_api_url` and `datasets_api_url` can be overridden for local integration
  tests; production endpoints require HTTPS.
