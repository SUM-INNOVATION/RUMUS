# rumus-distributed

3D parallelism engine for the **RUMUS** deep learning framework — Tensor Parallelism (TP), Pipeline Parallelism (PP), and async collective operations.

## Architecture

### Async Communication

```text
Compute Thread                    Comm Thread (owns Arc<Device> + Arc<Queue>)
──────────────                    ──────────────────────────────────────────
1. Dispatch compute kernel        (idle)
2. Encode copy → staging          (idle)
3. queue.submit()                 (idle)
4. Send CommRequest ──────────►   5. staging.map_async(Read)
5. Start NEXT micro-batch         6. device.poll(Wait)  ← blocks HERE only
   (non-blocking!)                7. Read mapped range → Vec<f32>
                                  8. CollectiveBarrier: push + wait
                                  9. Read reduced result
                                 10. queue.write_buffer(result → dst_buf)
N. handle.wait() ◄───────────── 11. Send completion signal
```

The compute thread never calls `poll(Wait)` — all blocking GPU-to-CPU transfers happen on the dedicated comm thread.

### Tensor Parallelism

**ColumnParallelLinear:** Weight `[K, N/T]` sharded along columns.
- Forward: `Y_t = X @ W_t` (no collective)
- Backward: `grad_X = AllReduce(Σ_t grad_X_t)` via `CollectiveBarrier`

**RowParallelLinear:** Weight `[K/T, N]` sharded along rows.
- Forward: `Y = AllReduce(Σ_t X_t @ W_t)` (partial sums reduced)
- Backward: `grad_X_t = grad_Y @ W_t^T` (no collective)

Only 2 AllReduces per Transformer block (1 forward + 1 backward).

### Pipeline Parallelism (1F1B Schedule)

```text
GPU 0: F0  F1  F2  F3  B3  B2  B1  B0
GPU 1:     F0  F1  F2  B2  B1  B0  ...
GPU 2:         F0  F1  B1  B0  ...
GPU 3:             F0  B0  ...
```

**Per-micro-batch isolated tapes:**
1. `context::install_tape(Tape::new())` — fresh tape per micro-batch
2. Forward pass records to the isolated tape
3. `context::take_tape()` — saves the tape for later backward

**Cross-stage gradient injection:**
1. Incoming tensor: `set_requires_grad(true)` → assigns known `GradId`
2. Forward with local layers
3. `backward_with_grad(output, grad_from_next_stage)` — injected seed
4. `grads.remove(saved_grad_id)` — extracts grad_input deterministically
5. Send grad_input to previous stage via channel

## Quick Start

```rust
use rumus_distributed::{PipelineExecutor, PipelineStage, CollectiveBarrier};

let stages = vec![
    PipelineStage { device_index: 0, forward_fn: Box::new(|x| layer0.forward(x)) },
    PipelineStage { device_index: 1, forward_fn: Box::new(|x| layer1.forward(x)) },
];

let executor = PipelineExecutor::new(stages, 4); // 4 micro-batches
let grad_stores = executor.run(&input_batch, &|output| loss_fn(output));
```

## API

### Collectives

| Struct | Description |
|--------|-------------|
| `CollectiveBarrier` | `Mutex` + `Condvar` AllReduce: last arrival sums + averages, `notify_all` |
| `CommThread` | Background thread with `Arc<Device/Queue>` for non-blocking GPU staging |
| `AllReduceHandle` | Oneshot completion handle — `.wait()` only when result needed |

### Tensor Parallelism

| Struct | Weight Split | Forward Collective | Backward Collective |
|--------|--------------|--------------------|---------------------|
| `ColumnParallelLinear` | Along N (columns) | None | AllReduce on grad_X |
| `RowParallelLinear` | Along K (rows) | AllReduce on output | None |

### Pipeline Parallelism

| Struct | Description |
|--------|-------------|
| `PipelineStage` | Per-device stage with user's `forward_fn` |
| `PipelineExecutor` | 1F1B scheduler: micro-batch tapes, gradient injection, `merge_from` |

## Core Engine Extensions

M24 adds the following to the `rumus` core crate:

| Addition | Location | Purpose |
|----------|----------|---------|
| `Arc<Device>` + `Arc<Queue>` | `GpuContext` | Enables comm thread device sharing |
| `install_tape(tape)` | `autograd::context` | Per-micro-batch tape isolation |
| `backward_with_grad(tensor, grad)` | `autograd::backward` | Cross-stage gradient injection |
| `merge_from(&mut other)` | `GradientStore` | Accumulate parameter gradients across micro-batches |

## Dependencies

- `rumus` (v0.3.0) — core framework with `gpu` + `multi_gpu` features
- `wgpu` — WebGPU device/buffer types
- `bytemuck` — safe byte casts

## License

Licensed under either of

- [Apache License, Version 2.0](../LICENSE-APACHE)
- [MIT License](../LICENSE-MIT)

at your option.
