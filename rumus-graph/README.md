# rumus-graph

Sparse graph engine for the **RUMUS** deep learning framework — WebGPU-native fused Sparse-Dense Matrix Multiplication (SpMM) for Graph Neural Networks.

## Why

Standard GNN message passing materializes an intermediate `[Edges, HiddenDim]` tensor that OOMs on massive graphs. `rumus-graph` fuses the gather → transform → scatter pipeline into a single WGSL kernel that accumulates directly in thread-private registers — zero intermediate VRAM allocation.

## Architecture

### CSR Memory Layout

Graphs are stored in **Compressed Sparse Row (CSR)** format on the GPU:

```text
row_ptr:     [u32; N + 1]   — row_ptr[i] = start of row i in col_indices
col_indices: [u32; E]       — neighbor node IDs
values:      [f32; E]       — edge weights (optional; 1.0 if absent)
```

Both forward (`A`: src → dst) and transposed (`A^T`: dst → src) CSR representations are pre-computed at graph construction time. The transpose is needed for the backward pass because WGSL lacks f32 atomic operations.

### Fused SpMM Kernel

```text
output[i, :] = Σ_{j ∈ neighbors(i)} A[i,j] × features[j, :]
```

- **1 thread = 1 node** — each thread owns its output row
- **Edge-outer / dim-inner loop** — reads `col_indices[e]` once per edge, sweeps `D` dimensions with stride-1 access on features (cache-friendly)
- **Private register accumulation** — zero shared memory, zero intermediate buffers
- **workgroup_size(256)** — dispatches `ceil(N / 256)` workgroups

### Autograd

Fully differentiable through the **M19 Custom Ops Plugin API**:

- **Forward:** `SpMMOp` implements `CustomOp` → `custom_forward()` dispatches the SpMM kernel
- **Backward:** `SpMMBackward` implements `CustomBackward` → runs SpMM on the transposed graph `A^T` for gradient routing (`grad_features = SpMM(A^T, grad_output)`)
- No f32 atomics needed — the transpose CSR is the mathematically correct and WebGPU-compatible approach

## Quick Start

```rust
use rumus_graph::Graph;
use rumus::tensor::Tensor;

// Build a graph from edge lists.
let src = vec![0, 0, 1, 2, 3];
let dst = vec![1, 2, 2, 3, 0];
let graph = Graph::new(&src, &dst, None, 4);  // 4 nodes, 5 edges

// Node features: [4 nodes, 16-dim]
let features = Tensor::new(vec![0.0f32; 4 * 16], vec![4, 16]);
features.to_gpu();

// Differentiable message passing.
let output = graph.spmm(&features);  // [4, 16]
```

## API

### `Graph::new(src, dst, weights, num_nodes)`

Builds a graph from edge lists. Constructs both forward and transposed CSR on the CPU, uploads to GPU. One-time `O(E log E)` cost.

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `&[u32]` | Source node IDs for each edge |
| `dst` | `&[u32]` | Destination node IDs for each edge |
| `weights` | `Option<&[f32]>` | Edge weights (`None` = unweighted, all 1.0) |
| `num_nodes` | `usize` | Total number of nodes in the graph |

### `Graph::spmm(features) -> Tensor`

Differentiable sparse message passing. Input `features: [N, D]`, output `[N, D]`.

### `SparseTensor`

Low-level CSR representation with raw `wgpu::Buffer` ownership. Buffers are returned to the `BufferPool` on `Drop`.

## WGSL Kernel Bindings

| Binding | Type | Content |
|---------|------|---------|
| 0 | `array<u32>` (read) | `row_ptr` — CSR row pointers |
| 1 | `array<u32>` (read) | `col_indices` — neighbor IDs |
| 2 | `array<scalar>` (read) | `values` — edge weights |
| 3 | `array<scalar>` (read) | `features` — dense node features `[N, D]` |
| 4 | `array<scalar>` (rw) | `output` — result `[N, D]` |
| 5 | uniform | `SpMMParams` — num_nodes, num_edges, hidden_dim, has_values |

## Dependencies

- `rumus` (v0.3.0) — core framework with GPU backend
- `wgpu` — WebGPU buffer types
- `bytemuck` — safe u32/f32 byte reinterpretation

## License

Licensed under either of

- [Apache License, Version 2.0](../LICENSE-APACHE)
- [MIT License](../LICENSE-MIT)

at your option.
