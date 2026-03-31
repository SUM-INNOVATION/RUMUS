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
| M3 — Core API, Optimizers & MVP Macros | Planned |
| M4 — WGPU Acceleration & Memory Pools | Planned |
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
| `StorageHandle` | `Arc<StorageInner>` wrapping raw memory + atomic version counter + fence |
| `WeakStorageHandle` | Non-owning ref for `VersionSnapshot` (dead tensor = provably unmutated) |
| `Layout` | Shape, strides, offset — views share storage with different layouts |
| `AutogradState` | `None` (inference) or `Tracked(Arc<TensorMeta>)` |
| `TensorMeta` | `requires_grad`, `grad_id`, `creator`, `total_grads` (atomic), `is_leaf` |
| `Tensor` | Composes `StorageHandle` + `Layout` + `AutogradState` |
| `Parameter` | `Tensor` + globally unique `ParamId` |
| `Backend` trait | Stateless associated fns (no `&self`) — `CpuBackend` is zero-sized |
| `Tape` | Append-only Wengert list of `TapeEntry` nodes |
| `GradientStore` | `HashMap<GradId, Tensor>` — accumulate-only, shape-checked |
| `BackwardOp` | Concrete enum (`Add`, `Mul`, `Matmul`) — no closures, `Send + Sync` |
| `VersionSnapshot` | `WeakStorageHandle` + recorded version — upgrade-or-dead check |

### Backward Engine

Kahn's algorithm in reverse tape order:
- **Pending map** built by counting input appearances across all tape entries.
- **Strict zero-ready gate** — `pending != 0` means unreachable, skip.
- **Dead-branch decrementing** — ready but no gradient → decrement parents before skip (prevents upstream starvation in branching graphs).
- **Version checks** via `VersionSnapshot::check()` — `Weak` upgrade failure = dead = `Ok(())`.

## Building

```bash
cargo build
cargo test
```

No external dependencies are required for the CPU backend.

## License

TBD
