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
| **M2 — Autograd Engine** | In Progress |
| M3 — Core API, Optimizers & MVP Macros | Planned |
| M4 — WGPU Acceleration & Memory Pools | Planned |
| M5A — Ergonomics & Polish | Planned |
| M5B — Advanced Vision & Checkpointing | Planned |

**Milestone 1** delivers the foundational tensor data model (`StorageHandle`,
`Layout`, `AutogradState`), the `Backend` trait, a pure-Rust CPU backend with
naive `ikj` matrix multiplication, zero-copy view operations, and
strided-to-contiguous materialisation.

**Milestone 2** (in progress) adds the autograd engine infrastructure:
`GradientStore` (dumb accumulate-only gradient buffer), `Tape` (append-only
Wengert list), concrete `BackwardOp` enum (no closures — `AddBackward`,
`MulBackward`, `MatmulBackward`), `VersionSnapshot` with weak-reference
semantics to avoid keeping intermediate tensor memory alive, and structured
`AutogradError` types. Kahn's algorithm backward traversal is next.

## Building

```bash
cargo build
cargo test
```

No external dependencies are required for the CPU backend.

## License

TBD
