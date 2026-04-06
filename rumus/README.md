# rumus

Core crate for the **RUMUS** native-Rust deep learning framework.

## What's Inside

| Module | Description |
|--------|-------------|
| `tensor` | `StorageHandle` (CPU `Vec` or GPU `wgpu::Buffer` via `parking_lot::RwLock<StorageData>`), `Layout`, `AutogradState`, `DType` (`F32`/`F16`/`Q8`), N-dimensional broadcasting, `to_dtype()` cast, `quantize()`/`dequantize()`, and all tensor operations (`add`, `mul`, `matmul`, `relu`, `sigmoid`, `tanh`, `gelu`, `leaky_relu`, `dropout`, `im2col`, `flatten`, `max_pool2d`, `batch_norm_2d`, `adaptive_avg_pool2d`, `bmm`, `softmax`, `layer_norm`, `embedding_forward`, `cross_entropy_loss`, `broadcast_add/sub/mul`, etc.) |
| `autograd` | Thread-local `Tape`, `GradientStore`, Kahn's algorithm backward engine, `no_grad()` RAII guard, `VersionSnapshot` with `Weak` references, 31 concrete `BackwardOp` variants (incl. `Cast`) |
| `backend` | `Backend` trait (CPU) + feature-gated `gpu` module: `GpuContext` singleton (`supports_f16`), `BufferPool`, `PipelineCache` (35+ F32 pipelines + 30 F16 + cast + Q8 quantize/dequantize/matmul pipelines), WGSL metaprogramming via `alias scalar` |
| `nn` | `Parameter`, `Module` trait, `#[derive(Module)]` (re-exported from `rumus-macros`), `Linear`, `Conv2d`, `ConvTranspose2d`, `MaxPool2d`, `AdaptiveAvgPool2d`, `Flatten`, `Dropout`, `BatchNorm2d`, `LayerNorm`, `Embedding`, `MultiheadAttention`, `TransformerBlock`, activations (`relu`, `sigmoid`, `tanh`, `gelu`, `leaky_relu`), `mse_loss`, `cross_entropy_loss`, safetensors IO |
| `optim` | `Optimizer` trait (`step` + `set_lr`/`get_lr`), `SGD`, `Adam`, `AdamW` — all with CPU + GPU dual-path dispatch. `LRScheduler` trait with `StepLR` and `CosineAnnealingLR`. `clip_grad_norm_` with 3-pass non-stalling GPU strategy |
| `data` | `Dataset` trait, `DataItem`, `DataLoader` with multithreaded prefetching (`std::thread` + bounded `mpsc`), Fisher-Yates shuffle, deadlock-free `Drop` teardown |
| `train` | `Trainer<O: Optimizer>` — closure-based `train_step()` orchestrator |

## Features

- **`default`** — CPU-only build. No external GPU dependencies.
- **`gpu`** — Enables WGPU compute backend (`wgpu` + `pollster`). All tensor ops auto-dispatch to WGSL shaders when data is GPU-resident.

## Quick Start

```rust
use rumus::nn::{self, Linear, Module, cross_entropy_loss};
use rumus::optim::{AdamW, Optimizer};
use rumus::train::Trainer;
use rumus::tensor::Tensor;

#[derive(Module)]
struct MLP {
    fc1: Linear,
    fc2: Linear,
}

impl MLP {
    fn new() -> Self {
        Self {
            fc1: Linear::new(4, 8, true),
            fc2: Linear::new(8, 3, true),
        }
    }
    fn forward(&self, x: &Tensor) -> Tensor {
        let h = nn::relu(&self.fc1.forward(x));
        self.fc2.forward(&h)
    }
}

let model = MLP::new();
let mut trainer = Trainer::new(AdamW::new(model.parameters(), 0.01));

// One training step:
let loss = trainer.train_step(|| {
    let logits = model.forward(&inputs);
    cross_entropy_loss(&logits, &targets)
}).unwrap();
```

## Dependencies

- `rumus-macros` — `#[derive(Module)]` proc macro
- `safetensors` — model persistence
- `bytemuck` — safe f32/u8 casts
- `parking_lot` — mapped `RwLock` guards for `StorageData`
- `wgpu` + `pollster` (optional, behind `gpu` feature)

## License

Licensed under either of

- [Apache License, Version 2.0](../LICENSE-APACHE)
- [MIT License](../LICENSE-MIT)

at your option.
