# RUMUS

**RUMUS** — *formula* in Indonesian — is a native-Rust deep learning framework.
The name reflects the project's goal: providing a foundational, mathematically
pure, and strict *formula* for high-performance deep learning in Rust.

## Goals

- **PyTorch-like ergonomics** — eager execution, familiar tensor API, `#[derive(Module)]` macro for model definitions.
- **Strict, explicit memory safety** — the borrow checker enforces invariants that other frameworks paper over with runtime checks or `unsafe`.
- **Zero-cost abstractions** — view operations (reshape, transpose) are metadata-only; inference mode carries zero autograd allocation overhead.
- **Borrow-safe optimizers** — optimizers drain gradients through `&mut` references, preventing overlapping borrows and supporting multi-optimizer setups (e.g., GANs) without `RefCell` hacks.
- **Hardware acceleration** — WGPU compute backend with per-resource fences instead of global pipeline stalls.

## Current Status

| Milestone | Status |
|-----------|--------|
| **M1 — Core Memory Model & CPU Compute** | Complete |
| **M2 — Autograd Engine** | Complete |
| **M3 — Core API, Optimizers & MVP Macros** | Complete |
| **M4 — WGPU Acceleration & Memory Pools** | Complete |
| M5A — Ergonomics & Polish | Planned |
| M5B — Advanced Vision & Checkpointing | Planned |

**Milestone 1** delivers the foundational tensor data model (`StorageHandle`,
`Layout`, `AutogradState`), the `Backend` trait, a pure-Rust CPU backend with
naive `ikj` matrix multiplication, zero-copy view operations, and
strided-to-contiguous materialisation.

**Milestone 2** delivers the full autograd engine: `GradientStore` (dumb
accumulate-only gradient buffer), `Tape` (append-only Wengert list), concrete
`BackwardOp` enum (no closures — `AddBackward`, `MulBackward`,
`MatmulBackward`), `VersionSnapshot` with weak-reference semantics to avoid
keeping intermediate tensor memory alive, structured `AutogradError` types,
thread-local autograd context with `no_grad()` RAII guard, `Arc<TensorMeta>`
with `AtomicUsize` edge counting for shared metadata across clones, forward-pass
tape recording integrated into `add`/`mul`/`matmul`, and Kahn's algorithm
backward traversal with correct dead-branch decrementing.

**Milestone 3** delivers the complete user-facing API. `StorageInner::data`
migrated to `RwLock<Vec<f32>>` for safe concurrent reads and exclusive optimizer
writes (zero `unsafe`). `Parameter` with global `AtomicUsize` `ParamId` allocator
and auto `requires_grad`. `trait Module` (state-only: `parameters`, `train`/`eval`,
`state_dict`/`load_state_dict` — `forward` is deliberately not in the trait).
`trait Optimizer` with drain-on-apply `&mut GradientStore` pattern. `SGD` (with
momentum) and `Adam` (with `ParamId`-keyed moment buffers and bias correction).
`Linear` layer with `[in, out]` weight layout (no-transpose forward), Kaiming
Uniform init via zero-dep LCG, and `add_bias` with `sum_rows` backward. `relu`
and `mse_loss` as differentiable ops with fused backward. Cargo workspace split
(`rumus` + `rumus-macros`), `#[derive(Module)]` proc macro generating
`parameters`, `train`, `eval`, `state_dict`, `load_state_dict` by iterating
all named struct fields. Safetensors serialization via `bytemuck::cast_slice`
(zero `unsafe`) for save and `f32::from_le_bytes` for load. E2E integration
test: XOR MLP trains with Adam, converges in <200 epochs, save/load roundtrip
produces exact output match.

**Milestone 4** delivers full GPU acceleration. `StorageInner` refactored to
`parking_lot::RwLock<StorageData>` enum (`Cpu`, `Gpu`, `Both`, `Transferring`)
with mapped guards, `GpuContext` singleton, `BufferPool`, and
lock-drop-transfer-reacquire device transfers. 5 WGSL shader modules (12 entry
points) with all uniform structs 16-byte padded per WebGPU spec and compile-time
`size_of` assertions. `PipelineCache` with named struct fields for 12 pipelines
+ 6 bind group layouts. GPU dispatch integrated into all tensor ops via
`is_gpu()` check. Fused WGSL optimizer kernels (`adam_step`, `sgd_step`) —
moments + weights updated in a single dispatch per parameter, zero D2H for
weights. Backward pass stays on-device: seed gradient pushed to GPU causes the
entire backward cascade to dispatch WGSL kernels. `Tensor::to_gpu()` and
`ModuleToGpu` blanket trait for device transfer. E2E GPU test: XOR MLP trains
entirely on the GPU and converges. All WGPU code feature-gated behind
`--features gpu`.

## Architecture

### Immutable Tenets

1. **The Tensor is Not a Junk Drawer** — strictly partitioned into `StorageHandle` (memory/versioning), `Layout` (shape/strides/views), and `AutogradState` (graph tracking).
2. **Op-Driven Autograd** — `GradientStore` is dumb (allocate + `+=` only). Individual ops handle un-broadcasting and layout unwinding in their backward pass.
3. **Strict Graph Edge Counting** — `total_grads` (`AtomicUsize` on shared `Arc<TensorMeta>`) tracks incoming gradient edges, not node usage. `y = x + x` increments the same counter twice.
4. **Borrow-Safe Optimizers** — optimizer takes `&mut grads` and drains only its registered `ParamId`s. No overlapping borrows, immediate memory release.
5. **Zero-Allocation Inference Mode** — `AutogradState::None` guarantees zero metadata allocations, zero version tracking, zero tape writes. `no_grad()` RAII guard suppresses recording.
6. **Isolated Checkpointing** — `RecomputePlan` holds strong `Arc` refs to boundary inputs, `Weak` refs to internals. Failed upgrade triggers recomputation, not panic.
7. **Per-Resource Hardware Fences** — `last_writer_fence` per `StorageHandle` via `AtomicUsize` sentinel. No global pipeline stalls.
8. **Errors over Panics** — user-triggerable faults return `AutogradError`. Panics reserved for internal invariants only.

### Core Data Model

| Struct | Role |
|--------|------|
| `StorageHandle` | `Arc<StorageInner>` wrapping `parking_lot::RwLock<StorageData>` + atomic version counter + fence |
| `StorageData` | Enum: `Cpu(Vec<f32>)`, `Gpu(wgpu::Buffer)`, `Both { cpu, gpu, dirty }`, `Transferring` |
| `GpuContext` | `OnceLock<Option<...>>` singleton holding `Device`, `Queue`, `PipelineCache`, `BufferPool` |
| `PipelineCache` | Named struct fields for 12 compute pipelines + 6 bind group layouts (no HashMap) |
| `BufferPool` | Power-of-2 bucketed GPU buffer cache (`Mutex<HashMap<PoolKey, Vec<Buffer>>>`) |
| `WeakStorageHandle` | Non-owning ref for `VersionSnapshot` (dead tensor = provably unmutated) |
| `Layout` | Shape, strides, offset — views share storage with different layouts |
| `AutogradState` | `None` (inference) or `Tracked(Arc<TensorMeta>)` |
| `TensorMeta` | `requires_grad`, `grad_id`, `creator`, `total_grads` (atomic), `is_leaf` |
| `Tensor` | Composes `StorageHandle` + `Layout` + `AutogradState` |
| `Parameter` | `Tensor` + globally unique `ParamId` (auto `requires_grad`), implements `Module` |
| `Backend` trait | Stateless associated fns (no `&self`) — `CpuBackend` is zero-sized |
| `Tape` | Append-only Wengert list of `TapeEntry` nodes |
| `GradientStore` | `HashMap<GradId, Tensor>` — accumulate-only, shape-checked |
| `BackwardOp` | Concrete enum (7 variants: `Add`, `Sub`, `Mul`, `Matmul`, `Relu`, `MseLoss`, `AddBias`) — no closures, `Send + Sync` |
| `VersionSnapshot` | `WeakStorageHandle` + recorded version — upgrade-or-dead check |
| `Module` trait | State-only: `parameters`, `train`/`eval`, `state_dict`/`load_state_dict` — `forward` is inherent |
| `#[derive(Module)]` | Proc macro generating all `Module` methods by iterating struct fields |
| `Optimizer` trait | `step(&mut self, &mut GradientStore)` — drain pattern |
| `SGD` | CPU: block-scoped `RwLock` guards. GPU: fused `sgd_step` WGSL kernel |
| `Adam` | CPU: block-scoped locks. GPU: fused `adam_step` WGSL kernel (m + v + param in one dispatch) |
| `Tensor::to_gpu()` | Triggers H2D transfer; `ModuleToGpu` blanket trait pushes all params |
| `Linear` | `[in, out]` weight layout, Kaiming init, `add_bias` for 1D bias broadcasting |
| `save/load_safetensors` | Dot-path state dict serialization via `bytemuck` + `safetensors` (zero `unsafe`) |

### Backward Engine

Kahn's algorithm in reverse tape order:
- **Pending map** built by counting input appearances across all tape entries.
- **Strict zero-ready gate** — `pending != 0` means unreachable, skip.
- **Dead-branch decrementing** — ready but no gradient → decrement parents before skip (prevents upstream starvation in branching graphs).
- **Version checks** via `VersionSnapshot::check()` — `Weak` upgrade failure = dead = `Ok(())`.

## Building

The project is a Cargo workspace with two crates:

```
RUMUS/
├── rumus/          # core framework
└── rumus-macros/   # #[derive(Module)] proc macro
```

```bash
cargo build                   # CPU-only build
cargo build --features gpu    # with WGPU GPU backend
cargo test                    # runs all tests (CPU)
cargo test --features gpu     # runs CPU + GPU tests
```

External dependencies: `syn`/`quote`/`proc-macro2` (macro crate),
`safetensors` + `bytemuck` + `parking_lot` (core crate).
GPU-only (behind `--features gpu`): `wgpu` + `pollster`.

## License

TBD
