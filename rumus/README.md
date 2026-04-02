# rumus

Core crate for the **RUMUS** native-Rust deep learning framework.

## What's Inside

| Module | Description |
|--------|-------------|
| `tensor` | `StorageHandle` (CPU `Vec` or GPU `wgpu::Buffer` via `parking_lot::RwLock<StorageData>`), `Layout`, `AutogradState`, and all tensor operations (`add`, `mul`, `matmul`, `relu`, `dropout`, `im2col`, `flatten`, `max_pool2d`, `cross_entropy_loss`, etc.) |
| `autograd` | Thread-local `Tape`, `GradientStore`, Kahn's algorithm backward engine, `no_grad()` RAII guard, `VersionSnapshot` with `Weak` references |
| `backend` | `Backend` trait (CPU) + feature-gated `gpu` module: `GpuContext` singleton, `BufferPool`, `PipelineCache` (25 WGSL compute pipelines), and all GPU dispatch functions |
| `nn` | `Parameter`, `Module` trait, `#[derive(Module)]` (re-exported from `rumus-macros`), `Linear`, `Conv2d`, `MaxPool2d`, `Flatten`, `Dropout`, `mse_loss`, `cross_entropy_loss`, safetensors IO |
| `optim` | `Optimizer` trait, `SGD`, `Adam`, `AdamW` — all with CPU + GPU dual-path dispatch |
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

MIT
