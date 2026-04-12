# rumus-serve

High-throughput inference server for the **RUMUS** deep learning framework with continuous batching, KV-caching, and attention masking.

## Architecture

```
HTTP Client ──► axum (tokio async) ──► mpsc channel ──► GPU Worker Thread
                    ▲                                         │
                    │                                    ┌────▼────┐
                    │                                    │ Collect  │
                    │                                    │ (5ms     │
                    │                                    │ deadline)│
                    │                                    └────┬────┘
                    │                                         │
                    │                                    ┌────▼────┐
                    │                                    │ Pad +   │
                    │                                    │ Mask    │
                    │                                    └────┬────┘
                    │                                         │
                    │                                    ┌────▼────┐
                    │                                    │ Prefill │
                    │                                    │ (full   │
                    │                                    │  seq)   │
                    │                                    └────┬────┘
                    │                                         │
                    │                                    ┌────▼────┐
                    │                                    │ Decode  │
                    │                                    │ Loop    │
                    │                                    │ (KV$)   │
                    │                                    └────┬────┘
                    │                                         │
                    └──── oneshot channel ◄────────────────────┘
```

## Key Features

- **Async↔Sync Bridge**: HTTP handlers run on `tokio`, GPU work runs on a dedicated `std::thread`. Connected via bounded `mpsc` (backpressure) + per-request `oneshot` (zero-lock response routing).
- **Deadline-Timer Batching**: Blocks for the first request, then drains the channel for up to `batch_timeout_ms` (default 5ms) to accumulate a batch. Balances latency vs throughput.
- **Attention Masking**: `pad_batch` generates both `input_ids: [B, max_seq]` and `attention_mask: [B, max_seq]` (1.0 for real tokens, 0.0 for padding). The model skips padded positions in softmax.
- **KV-Cache**: Two-phase generation — Prefill (full sequence, populates cache) → Decode (only `[B, 1]` new token + cache, O(1) per step).
- **Scatter-Return**: Results routed back to individual HTTP clients via `oneshot` senders — no shared map, no locking, automatic cleanup on client disconnect.

## Quick Start

```bash
# Build
cargo build -p rumus-serve --release

# Run (uses a mock transformer model)
./target/release/rumus-serve \
    --port 8080 \
    --max-batch 8 \
    --batch-timeout-ms 5 \
    --vocab-size 32000 \
    --hidden-dim 256 \
    --num-heads 4 \
    --num-layers 2
```

## API

### `POST /v1/generate`

**Request:**

```json
{
    "token_ids": [101, 2023, 2003, 1037, 3231],
    "max_new_tokens": 50
}
```

**Response:**

```json
{
    "generated_ids": [101, 2023, 2003, 1037, 3231, 2008, ...],
    "latency_ms": 42.3
}
```

Token IDs are pre-tokenized. The server does not perform tokenization — use a client-side tokenizer.

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8080 | HTTP listen port |
| `--max-batch` | 8 | Maximum requests per batch |
| `--batch-timeout-ms` | 5 | Deadline timer for batch collection (ms) |
| `--vocab-size` | 32000 | Model vocabulary size |
| `--hidden-dim` | 256 | Transformer hidden dimension |
| `--num-heads` | 4 | Number of attention heads |
| `--num-layers` | 2 | Number of transformer layers |

## Dependencies

- `rumus` (v0.3.0) — core framework with GPU backend
- `axum` — async HTTP framework
- `tokio` — async runtime
- `serde` / `serde_json` — JSON serialization
- `tower-http` — CORS middleware

## License

Licensed under either of

- [Apache License, Version 2.0](../LICENSE-APACHE)
- [MIT License](../LICENSE-MIT)

at your option.
