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
| **M5 — Conv2D & Advanced Layers** | Complete |
| **M6 — Inference Mode & Dropout** | Complete |
| **M7a — Loss Functions & GPU Optimizers** | Complete |
| **M7b — The Training Loop** | Complete |
| **M8a — Activations & Broadcasting** | Complete |
| **M8b — LayerNorm & Embeddings** | Complete |
| **M8c — BMM, Softmax & SDPA** | Complete |
| **M8d — Transformer Block** | Complete |
| **M9 — The Vision Stack** | Complete |
| **M10 — Training Ergonomics** | Complete |
| **M11 — FP16 Mixed Precision** | Complete |
| **M12 — INT8 Quantization Engine** | Complete |
| **M13 — ONNX Exporter** | Complete |
| **M14 — Data Engine (Memory-Mapped Records)** | Complete |
| **M15 — JIT Compiler (Kernel Fusion)** | Complete |
| **M16 — Distributed Strategy (Multi-GPU)** | Complete |
| **M17 — FlashAttention** | Complete |
| **M18 — FSDP (Fully Sharded Data Parallelism)** | Complete |
| **M19 — Custom Ops Extension API** | Complete |
| **M20 — Inference Server (Continuous Batching)** | Complete |
| **M21 — Sparse Graph Engine (GNNs)** | Complete |
| **M22 — Spatial Engine (Direct Conv & Pool)** | Complete |
| **M23 — Ultra-Low Precision (INT4 AWQ/GPTQ)** | Complete |
| **M24 — 3D Parallelism (TP + PP)** | Complete |

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
`size_of` assertions. `PipelineCache` with named struct fields (compile-time
guarantees, no HashMap). GPU dispatch integrated into all tensor ops via
`is_gpu()` check. Fused WGSL optimizer kernels (`adam_step`, `sgd_step`) —
moments + weights updated in a single dispatch per parameter, zero D2H for
weights. Backward pass stays on-device: seed gradient pushed to GPU causes the
entire backward cascade to dispatch WGSL kernels. `Tensor::to_gpu()` and
`ModuleToGpu` blanket trait for device transfer. E2E GPU test: XOR MLP trains
entirely on the GPU and converges. All WGPU code feature-gated behind
`--features gpu`.

**Milestone 5** delivers full CNN support. Chunk 1: Conv2D via im2col — tracked
`im2col`, `col2im`, `slice_batch`, `stack`, `add_channel_bias`, `reshape_tracked`
ops with dedicated backward implementations. Chunk 2: `MaxPool2d` with f32
argmax indexing and `stride >= kernel_size` assertion (non-atomic backward).
Zero-copy tracked `flatten` (forward: clone storage + new layout; backward:
reshape). `MaxPool2d`, `Flatten` modules (no learnable params). WGSL kernels for
im2col, col2im, channel-bias, max_pool2d forward/backward. E2E CNN test: Mini-LeNet
(Conv2d→ReLU→MaxPool2d→Flatten→Linear→ReLU→Linear) trains on synthetic spatial
patterns and converges. `BackwardOp` enum now has 14 variants.

**M6 — Inference Mode & Dropout (Complete):** Achieved zero-copy GPU memory
locality, implemented WGSL PCG PRNG, bypassed tape recording for inference, and
deployed a Fused Stride-Aware Kernel for Dropout to prevent VRAM bloat.

**M7a — Loss Functions & GPU Optimizers (Complete):** GPU-fused Cross-Entropy
Loss with Log-Sum-Exp stability (workgroup shared memory reduction for max +
sum_exp, gradient pre-computed during forward — backward is a zero-copy
broadcast-scale via dedicated WGSL kernel). AdamW optimizer with decoupled
weight decay, GPU-native moment buffer initialization via `clear_buffer`
(zero host allocation), and fused WGSL update kernel. `broadcast_scale_kernel`
for on-device scalar-tensor multiplication (no D2H for the scalar).

**M7b — The Training Loop (Complete):** `Trainer<O: Optimizer>` struct with
closure-based `train_step()` — executes forward, loss read (4-byte D2H),
backward, and optimizer update in one call. No `zero_grad()` needed (drain
pattern clears automatically). `impl Drop for StorageInner` returns GPU buffers
to the `BufferPool` when the last `Arc` reference drops — completes the
allocation-free resource lifecycle. `BufferPool::cached_count()` introspection
verifies pool recycling (`cached_count > 0` after first iteration). Fixed
WebGPU uniform alignment (`vec4<u32>` packing). E2E tests: 3-class
classification with AdamW + cross-entropy on CPU and GPU, Trainer API
ergonomics, plus BufferPool recycling assertion. All 7 tests pass.

**M8a — Activations & Broadcasting (Complete):** Four new activation functions
(`sigmoid`, `tanh`, `gelu`, `leaky_relu`) with CPU + GPU forward/backward.
N-dimensional broadcasting (`broadcast_add`, `broadcast_sub`, `broadcast_mul`)
with stride-0 indexing, `BroadcastBinaryParams` WGSL kernel, and `reduce_sum`
GPU kernel for backward reduction along broadcast dimensions.
`fused_scale_kernel` for stride-aware negation (no `.contiguous()` needed).
`BackwardOp` enum expanded to 22 variants.

**M8b — LayerNorm & Embeddings (Complete):** Layer normalization with 3-phase
WGSL kernels (forward: mean+invstd per instance; backward: per-instance
grad_input + grad_weight product + reduce_sum for parameter gradients).
Embedding lookup (indices as f32, CPU forward/backward — no f32 atomics in
WGSL). `LayerNorm` and `Embedding` nn modules. Full GPU dispatch for LayerNorm
forward and backward.

**M8c — BMM, Softmax & SDPA (Complete):** Batched matrix multiplication via
Z-axis dispatch (`dispatch_workgroups((n+15)/16, (m+15)/16, batch)`). Row-wise
softmax with Log-Sum-Exp stability (forward + backward WGSL kernels). Scaled
Dot-Product Attention as a composition of tracked ops (`bmm`, `softmax`,
`broadcast_mul`). `BackwardOp` enum expanded to 25 variants.

**M8d — Transformer Block (Complete):** `MultiheadAttention` with tracked head
splitting (`reshape_tracked`, `transpose_tracked`, `contiguous_tracked`),
per-head SDPA, and output projection. `TransformerBlock` composing MHA +
LayerNorm + 2-layer FFN with residual connections. Causal masking via
`broadcast_add` with upper-triangular `-1e9` matrix. E2E test: TinyGPT (2-layer
transformer with positional embeddings) trains on CPU and generates tokens.
`BackwardOp` enum now has 28 variants.

**M9 — The Vision Stack (Complete):** `BatchNorm2d` with per-channel
normalization, `RefCell`-based running statistics (no unsafe), train/eval
toggle, and momentum-based EMA updates. GPU forward via WGSL `batch_norm.wgsl`
kernel; backward via `batch_norm_bw.wgsl` for grad_input + CPU reduction for
grad_weight/grad_bias. `ConvTranspose2d` (transposed convolution) implemented
as a composition of tracked ops: `weight^T @ x_flat → col2im → channel_bias`.
`AdaptiveAvgPool2d` with dynamic bin boundaries (floor-start, ceil-end) via
WGSL `adaptive_pool.wgsl` kernel (forward + backward). All three modules
implement `Module` trait with state_dict support. `BackwardOp` enum now has 30
variants.

**M10 — Training Ergonomics (Complete):** `data::Dataset` trait and
`data::DataLoader` with multithreaded prefetching via bounded `mpsc` channels.
Worker threads fetch and collate batches in the background; channel capacity
`prefetch_factor * num_workers` prevents OOM. Deadlock-free teardown: `Drop`
drops `batch_rx` (via `Option::take`) before joining workers, ensuring blocked
`send()` calls wake with `Err`. Feeder thread explicitly joined. Fisher-Yates
shuffle with zero-dep LCG. Learning rate schedulers: `LRScheduler` trait with
`StepLR` (step decay) and `CosineAnnealingLR` (smooth cosine annealing with
`eta_min`). `Optimizer` trait extended with `set_lr()`/`get_lr()` on SGD, Adam,
AdamW. Gradient clipping: `clip_grad_norm_` with 3-pass non-stalling GPU
strategy — Pass 1: dispatch `reduce_sum_sq` WGSL kernel for all GPU gradients;
Pass 2: read back per-parameter squared norms (single sync point), compute
global L2 norm; Pass 3: scale via `broadcast_scale` if norm exceeds `max_norm`.
`GradientStore::replace()` swaps gradient tensors without WebGPU buffer aliasing.
Zero new external dependencies.

**M11 — FP16 Mixed Precision (Complete):** `DType` enum (`F32`, `F16`) with
`gpu_buf_size(numel)` for 4-byte-aligned buffer allocation. `dtype` tracked on
`StorageInner` (immutable after construction), `Tensor::dtype()` derived from
storage. `GpuContext` requests `wgpu::Features::SHADER_F16` at init, stores
`supports_f16` flag. WGSL shader metaprogramming via `preprocess_shader()`:
prepends `alias scalar = f32;` or `enable f16; alias scalar = f16;` — all 23
data-path shaders refactored to use `scalar` type alias for data buffers,
accumulators, and workgroup shared memory, with uniform reads explicitly cast
via `scalar(params.field)`. `F16Pipelines` struct with 30 compute pipelines
compiled conditionally when hardware supports `shader-f16`. `cast.wgsl` with
`cast_f32_to_f16_kernel` / `cast_f16_to_f32_kernel` using concrete types.
`Tensor::to_dtype(target)` tracked op with `CastBackward` (gradient = reverse
cast). `data()` on F16 tensors dynamically casts to F32 on GPU before download
(no panics on inspect). All optimizers auto-cast F16 gradients to F32 before
master weight update. All hardcoded `* 4` buffer allocations replaced with
`DType::gpu_buf_size()`. `BackwardOp` enum now has 31 variants. Zero new
external dependencies.

**M12 — INT8 Quantization Engine (Complete):** `DType::Q8 { block_size }`
variant for symmetric block-quantized INT8. Fused GPU buffer layout: 4-byte
header (f16 scale in lower 16 bits + 2B padding for u32 alignment) followed by
`block_size` i8 values per block. `gpu_buf_size()` computes
`num_blocks * (4 + block_size)` with 4-byte alignment. `quantize.wgsl`:
per-block workgroup reduction for `abs_max`, `pack2x16float` for f16 scale,
4-i8-per-u32 packing. `dequantize.wgsl`: per-element `unpack2x16float` +
sign-extended i8 extraction. `matmul_q8.wgsl`: mixed-precision kernel where
`scalar` activations x Q8 weights -> `scalar` output, dequantizing i8 weights
in-register via vectorized `unpack_i8x4()` — no intermediate VRAM expansion.
`Tensor::quantize(block_size)` transposes-then-quantizes to produce column-major
block order matching the matmul kernel's access pattern. `Tensor::matmul()`
auto-dispatches `matmul_q8` when rhs is Q8. All quantization ops are untracked
(PTQ inference-only, `AutogradState::None`). `data()` on Q8 tensors
dequantizes on GPU before download. Zero new external dependencies.

**M13 — ONNX Exporter (Complete):** Thread-local `onnx::Tracer` context
(same pattern as autograd) intercepts tensor ops during a tracing forward pass.
Primitive ops (`Add`, `Mul`, `MatMul`, `Relu`) record `TracedNode` entries;
composite modules (`Linear` → `Gemm`, `Conv2d` → `Conv`) use
`enter_fusion()`/`leave_fusion()` to suppress primitives and emit single fused
ONNX nodes. `TracedGraph` captures nodes, inputs, outputs, and initializers.
`export_onnx()` serializes to `.onnx` via `prost`-generated Protobuf types
from a vendored `onnx.proto` schema. F16 weights preserved natively via
`download_raw_bytes()` (no F32 cast). Q8 weights dequantized to F32 for ONNX
compatibility. Feature-gated behind `--features onnx`; `prost-build` compiles
proto at build time (env var `CARGO_FEATURE_ONNX` gating in `build.rs`).
Configurable opset version (default 17). `StorageHandle::ptr_id()` for value
identity tracking across ops.

**M14 — Data Engine (Complete):** `.rrec` (RUMUS Record) binary format for
high-throughput data loading. 64-byte header (magic `RREC`, version, num_records,
index_offset) + sequential data blocks (per-tensor meta: ndim/shape/dtype_tag +
raw bytes + 4-byte alignment padding) + trailing index table (offset/length u64
pairs for O(1) random access). `RecordWriter`: append-only sequential writer
with `create`/`append`/`finish` — streams records in a single forward pass,
patches header on finish. F32 via `bytemuck::cast_slice`, F16 via
`download_raw_bytes()`. `RecordDataset`: `memmap2::Mmap`-backed reader
implementing `Dataset` trait — O(1) index lookup, `parse_record` with
cursor-based byte parsing, `bytemuck::cast_slice` for F32, inline `f16_to_f32`
IEEE 754 bit conversion for F16. Single `memcpy` per record (mmap avoids
syscall overhead, `to_vec()` is a transient CPU staging buffer). `Send + Sync`
safe for concurrent `DataLoader` workers. New dependency: `memmap2 = "0.9"`.

**M15 — JIT Compiler (Complete):** XLA-style kernel fusion for element-wise ops.
Thread-local `JitTracer` captures `FusedOp` IR (Add, Sub, Mul, Relu, Sigmoid,
Tanh, Gelu, LeakyRelu, Scale, Neg) into a `FusionBlock`. `jit::compile(|| ...)`
scope activates tracing — element-wise ops record IR + return `Deferred` storage
handles instead of dispatching individual GPU kernels. Autograd tape recording
proceeds normally (backward ops see the same `StorageHandle` references).
`codegen::generate_wgsl()` emits straight-line `@compute` kernel: each thread
processes one element, each `FusedOp` maps to one `let vN = ...;` WGSL line.
`JitCache` hashes `FusionKey` (op tags + numel + dtype + binding counts) for O(1)
pipeline reuse — `create_shader_module` + `create_compute_pipeline` compiled once
per unique op sequence, cached forever. Dynamic `BindGroupLayout` built per
fusion block (N read + M write + 1 uniform). `flush_block()` materializes
deferred storages with real GPU buffers after the single fused dispatch.
`StorageData::Deferred { var_id }` variant panics with clear messages on premature
access. Feature-gated: `jit = ["gpu"]`. Zero new external dependencies.

**M16 — Distributed Strategy (Complete):** `MultiGpuContext` enumerates all
discrete/integrated GPUs, initializes each with its own `Device`, `Queue`,
`BufferPool`, and `PipelineCache`. `device_index: usize` on `StorageInner`
tracks which GPU a tensor lives on; `device_ctx()` helper routes all
`ensure_gpu`/`ensure_cpu`/`download_raw_bytes` to the correct device.
`Tensor::to_device(idx)` transfers via CPU staging (WebGPU has no peer-to-peer
DMA). `Tensor::slice_range(dim, start, end)` and `tensor::cat(tensors, dim)`
with tracked backward ops (`SliceRangeBackward`, `CatBackward`).
`DataParallel<M>` wrapper: broadcasts master weights, scatters input batch via
`slice_range` + `to_device`, runs forward concurrently via `std::thread::scope`
(one thread per GPU), gathers output via `cat`. `AllReduceSync` implements
4-phase WebGPU async gradient averaging: (1) submit `copy_buffer_to_buffer` to
staging, (2) `map_async` all staging buffers, (3) `device.poll(Wait)` per
device, (4) read mapped views + element-wise average. Feature-gated:
`multi_gpu = ["gpu"]`. Zero new external dependencies.

**M17 — FlashAttention (Complete):** Memory-efficient scaled dot-product
attention via tiled online softmax. 1D workgroup (64 threads, one per query
row) cooperatively loads K/V blocks into `var<workgroup>` shared memory, each
thread independently computes its row's attention via online softmax recurrence
(`m_i`, `l_i`, `o_i` private state). No N×N attention matrix materialized.
Causal mask via early column-loop exit. `select_flash_block_sizes(d, dtype)`
dynamically fits B_r/B_c within 16KB shared memory budget. Transparent autograd
fallback: if inputs require grad, delegates to standard SDPA with full tape
recording. `flash_attn_pipeline` with 5 bindings (Q+K+V+O+params).

**M18 — FSDP (Complete):** Fully Sharded Data Parallelism — each rank stores
only 1/N of every parameter. `FSDP::new(model, device_ids, rank)` slices
parameters along dim 0. `forward_linear` All-Gathers full weight from shard
storages, computes inside `no_grad()` (prevents autograd leak), drops gathered
weight immediately, pushes `FsdpLinearBackward` to tape. Backward re-gathers
weight, computes grad_X and grad_W locally, then mid-tape Reduce-Scatter via
`FsdpSync` barrier (`Mutex` + `Condvar`): all ranks push local gradients, last
arrival sums and averages, `notify_all`, each rank slices its shard using exact
`weight_shard_offset`. True FSDP memory profile — peak VRAM = one layer's worth
of params, not the entire model.

**M19 — Custom Ops Extension API (Complete):** `CustomOp` trait for user-defined
WGSL kernels (`wgsl_source`, `entry_point`, `output_shape`). `custom_forward()`
dynamically compiles and caches pipelines via `CustomOpCache` (separate from core
`PipelineCache`, namespace-isolated). `CustomBackward` trait for user-defined
gradient math. `BackwardOp::Custom(CustomBackwardOp)` — the single
`Arc<dyn CustomBackward>` escape hatch in the concrete enum. Saved tensors
captured via `save_for_backward`. GPU feature-gated. Zero new dependencies.

**M20 — Inference Server (Complete):** `rumus-serve` binary crate — high-throughput
LLM inference with continuous batching. `axum` + `tokio` HTTP server with
`POST /v1/generate` endpoint. Async↔sync bridge: bounded `tokio::sync::mpsc`
channel connects HTTP handlers to a dedicated GPU `std::thread`. Per-request
`oneshot` channel for zero-lock response routing. Deadline-timer batch collection
(5ms default). `pad_batch` generates both `input_ids: [B, max_seq]` and
`attention_mask: [B, max_seq]` (1.0 real, 0.0 padding). Two-phase generation:
Prefill (full padded sequence + mask, populates KV-cache per layer) → Decode loop
(only `[B, 1]` new tokens + KV-cache, O(1) per step instead of O(N²)).
`MockTransformer` with `prefill()` / `decode_step()` demonstrating mask-aware
attention and KV-cache extension. Scatter-return via `oneshot_tx` per request.

**M21 — Sparse Graph Engine (Complete):** `rumus-graph` library crate for GNNs.
`SparseTensor` in CSR format (`row_ptr`, `col_indices`, `values` as GPU buffers
via `BufferPool`). `Graph` holds both forward (A) and backward (A^T) CSR
representations — transpose pre-computed at construction to bypass WGSL's lack
of f32 atomics. Fused SpMM WGSL kernel (`spmm.wgsl`): 1 thread per node,
edge-outer / dim-inner loop for cache locality, `workgroup_size(256)`.
`SpMMOp` implements `CustomOp` (M19 plugin API) with 6 dynamic bindings.
`SpMMBackward` implements `CustomBackward` — runs SpMM on transposed graph for
gradient routing. Fully differentiable `graph.spmm(features)` API. Nodes sorted
by degree to mitigate subgroup divergence.

**M22 — Spatial Engine (Complete):** `rumus-vision` library crate for CNNs.
Direct sliding-window Conv2d WGSL kernel — 1 thread per output pixel, triple
nested loop (C_in × K_h × K_w), handles stride/padding/dilation, zero im2col
VRAM. Three backward kernels: `conv2d_backward_data` (transposed convolution for
grad_input), `conv2d_backward_weight` (gradient accumulation over batch ×
spatial), `conv2d_backward_bias` (channel-wise reduction). MaxPool2d with
F16-safe local window argmax: stores `ky * kernel_w + kx` (bounded by K²,
lossless in f16) in concatenated output buffer. Backward reconstructs global
input coordinates from local index + output position. `assert!(K_h * K_w <=
2048)` enforces the f16 precision bound. Tracked `slice_range` + `reshape_tracked`
preserve autograd chain from combined output to user-facing tensor. All ops via
M19 `CustomOp` / `CustomBackward` plugin API.

**M23 — Ultra-Low Precision (Complete):** INT4 weight-only quantization
(AWQ/GPTQ-style). `QuantizedTensor` packs 8 × 4-bit unsigned weights per `u32`,
K-major layout. K padded to `group_size` (multiple of 8) — eliminates word
overflow. Group-wise scales/zero-points `[num_groups, N]` as f32, grouped along
K. `qmatmul_int4.wgsl`: fused dequant-matmul, 1 thread per output element,
group-outer / word-inner loop, 8-nibble unpacking via `(packed >> (j*4)) & 0xF`,
dequant `(q - zp) * scale`. `qmatmul_int4_transpose.wgsl` for activation
gradient `grad_x = grad_y @ dequant(W)^T`. `QLinear` module: frozen INT4 weights
(`AutogradState::None` — no GradId, no optimizer) + optional trainable bias
(LoRA-compatible). `QLinearBackward` returns `vec![grad_x]` only — frozen inputs
skipped. `from_linear(linear, group_size)` quantizes pre-trained weights. 8x
compression vs F32, 4x vs F16.

**M24 — 3D Parallelism (Complete):** `rumus-distributed` library crate.
`GpuContext` refactored to `Arc<Device>` + `Arc<Queue>` for comm thread sharing.
`CommThread` owns device/queue arcs, performs `map_async` + `poll(Wait)` +
barrier reduce + `write_buffer` on a dedicated background thread — compute thread
never blocks. `CollectiveBarrier` (`Mutex` + `Condvar`) for cross-rank AllReduce.
`ColumnParallelLinear` (weight along N, AllReduce on grad_X in backward) +
`RowParallelLinear` (weight along K, AllReduce on output in forward).
`PipelineExecutor` with 1F1B micro-batch schedule: per-micro-batch isolated tapes
via `context::install_tape(Tape::new())`, cross-stage gradient injection via
`backward_with_grad(tensor, grad)`, incoming tensors tracked via
`set_requires_grad(true)` with saved `GradId` for deterministic gradient
extraction. `GradientStore::merge_from()` accumulates parameter gradients.

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
| `DType` | `F32` / `F16` / `Q8 { block_size }` enum with `gpu_buf_size(numel)` (4-byte aligned) |
| `StorageData` | Enum: `Cpu(Vec<f32>)`, `Gpu(wgpu::Buffer)`, `Both { cpu, gpu, dirty }`, `Transferring` |
| `GpuContext` | `OnceLock<Option<...>>` singleton holding `Device`, `Queue`, `PipelineCache`, `BufferPool`, `supports_f16` |
| `PipelineCache` | Named struct fields for 35+ F32 pipelines + `Option<F16Pipelines>` (30 F16 variants) + cast + Q8 pipelines |
| `BufferPool` | Power-of-2 bucketed GPU buffer cache (`Mutex<HashMap<PoolKey, Vec<Buffer>>>`) |
| `WeakStorageHandle` | Non-owning ref for `VersionSnapshot` (dead tensor = provably unmutated) |
| `Layout` | Shape, strides, offset — views share storage with different layouts |
| `AutogradState` | `None` (inference) or `Tracked(Arc<TensorMeta>)` |
| `TensorMeta` | `requires_grad`, `grad_id`, `creator`, `total_grads` (atomic), `is_leaf` |
| `Tensor` | Composes `StorageHandle` + `Layout` + `AutogradState` |
| `Parameter` | `Tensor` + globally unique `ParamId` (auto `requires_grad`), implements `Module` |
| `Backend` trait | Stateless associated fns (no `&self`) — `CpuBackend` is zero-sized |
| `Tape` | Append-only Wengert list of `TapeEntry` nodes |
| `GradientStore` | `HashMap<GradId, Tensor>` — accumulate-only, shape-checked, `replace()` for clip |
| `BackwardOp` | Concrete enum (31 variants incl. `Cast`) — no closures, `Send + Sync` |
| `VersionSnapshot` | `WeakStorageHandle` + recorded version — upgrade-or-dead check |
| `Module` trait | State-only: `parameters`, `train`/`eval`, `state_dict`/`load_state_dict` — `forward` is inherent |
| `#[derive(Module)]` | Proc macro generating all `Module` methods by iterating struct fields |
| `Optimizer` trait | `step(&mut self, &mut GradientStore)` + `set_lr`/`get_lr` — drain pattern |
| `SGD` | CPU: block-scoped `RwLock` guards. GPU: fused `sgd_step` WGSL kernel |
| `Adam` | CPU: block-scoped locks. GPU: fused `adam_step` WGSL kernel (m + v + param in one dispatch) |
| `Tensor::to_gpu()` | Triggers H2D transfer; `ModuleToGpu` blanket trait pushes all params |
| `Linear` | `[in, out]` weight layout, Kaiming init, `add_bias` for 1D bias broadcasting |
| `Conv2d` | im2col + matmul forward, `[C_out, C_in*K*K]` weight, Kaiming init, per-batch tracked loop |
| `MaxPool2d` | f32 argmax indexing, `stride >= kernel_size`, WGSL forward+backward |
| `Flatten` | Zero-copy tracked reshape `[B,C,H,W] → [B,C*H*W]` |
| `Dropout` | Inverted scaling `1/(1-p)`, PCG hash PRNG (CPU+GPU), `train`/`eval` toggle |
| `AdamW` | Decoupled weight decay, GPU-native moment init via `clear_buffer`, fused WGSL kernel |
| `Sigmoid/Tanh/GeLU/LeakyReLU` | Activation functions with CPU + GPU forward/backward, save-output or save-input for backward |
| `LayerNorm` | Layer normalization over last dim, 3-phase WGSL kernels, saves mean+invstd for backward |
| `Embedding` | Lookup table `[vocab, dim]`, CPU sparse scatter backward (no f32 atomics in WGSL) |
| `Bmm` | Batched matmul `[B,M,K]@[B,K,N]→[B,M,N]` via Z-axis dispatch |
| `Softmax` | Row-wise Log-Sum-Exp softmax with WGSL forward/backward kernels |
| `MultiheadAttention` | Tracked head splitting + per-head SDPA + output projection |
| `TransformerBlock` | MHA + LayerNorm + 2-layer FFN with residual connections, causal masking |
| `BatchNorm2d` | Per-channel normalization `[B,C,H,W]`, RefCell running stats, train/eval toggle, WGSL kernel |
| `ConvTranspose2d` | Transposed convolution via `W^T @ x → col2im`, composition of tracked ops |
| `AdaptiveAvgPool2d` | Dynamic-bin average pooling to fixed output size, WGSL forward/backward |
| `Trainer<O>` | Closure-based `train_step()`, epoch loss tracking, no `zero_grad` needed |
| `Dataset` trait | `len()` + `get(index)` → `DataItem`, `Send + Sync` for worker threads |
| `DataLoader` | Multithreaded batching with bounded `mpsc` channels, Fisher-Yates shuffle, deadlock-free `Drop` |
| `StepLR` | Step-decay scheduler: `lr *= gamma` every `step_size` epochs |
| `CosineAnnealingLR` | Cosine annealing from `initial_lr` to `eta_min` over `t_max` epochs |
| `clip_grad_norm_` | 3-pass GPU-safe gradient clipping: reduce_sum_sq → read norms → scale |
| `Tensor::to_dtype()` | GPU cast `F32↔F16` via WGSL kernels, tracked with `CastBackward` |
| `preprocess_shader()` | WGSL metaprogramming: `alias scalar = f32/f16` injection for dual F32/F16 pipelines |
| `Tensor::quantize()` | Symmetric block INT8 quantization with column-major repacking for matmul locality |
| `matmul_q8` | Mixed-precision WGSL kernel: scalar A x Q8 B -> scalar C, in-register dequantization |
| `onnx::Tracer` | Thread-local graph tracer: records `TracedNode` entries, fusion scoping for modules |
| `export_onnx()` | Single-call ONNX export: trace → `prost` Protobuf serialization → `.onnx` file |
| `RecordWriter` | Append-only `.rrec` writer: sequential tensor serialization + trailing index + header patching |
| `RecordDataset` | `memmap2::Mmap`-backed `.rrec` reader: O(1) index lookup, `Dataset` trait, `Send + Sync` |
| `jit::compile()` | JIT fusion scope: traces element-wise ops into `FusedOp` IR, codegen → cached WGSL pipeline |
| `JitCache` | `FusionKey`-hashed pipeline cache: O(1) hit, one-time compilation per unique op sequence |
| `MultiGpuContext` | Process-global singleton: enumerates all GPUs, per-device `BufferPool` + `PipelineCache` |
| `DataParallel<M>` | Scatter-forward-gather wrapper: `std::thread::scope` for concurrent per-GPU forward passes |
| `AllReduceSync` | 4-phase WebGPU gradient averaging: copy → map_async → poll(Wait) → CPU average |
| `slice_range` / `cat` | Tracked batch splitting and concatenation along any dimension |
| `flash_attention` | O(N) VRAM FlashAttention: tiled online softmax, auto-fallback to SDPA for training |
| `FSDP` | Fully Sharded Data Parallelism: 1/N params per rank, All-Gather + `FsdpSync` Reduce-Scatter |
| `CustomOp` / `CustomBackward` | Plugin API: user-defined WGSL kernels + autograd, `CustomOpCache` pipeline caching |
| `SparseTensor` / `Graph` | CSR format GPU graph with forward + transposed adjacency for differentiable SpMM |
| `SpMMOp` | Fused Sparse-Dense MatMul via M19 plugin: 1 thread/node, edge-outer loop, zero intermediate VRAM |
| `conv2d` (vision) | Direct sliding-window Conv2d: zero im2col VRAM, stride/padding/dilation, 3 backward kernels |
| `max_pool2d` (vision) | F16-safe local-window argmax, concatenated output, `assert!(K² ≤ 2048)` precision guard |
| `QLinear` / `QuantizedTensor` | INT4 weight-only quantization: 8 per u32, group-wise K-major, fused dequant-matmul |
| `CommThread` | Dedicated comm thread with `Arc<Device/Queue>` for non-blocking async AllReduce |
| `ColumnParallelLinear` / `RowParallelLinear` | Tensor Parallelism: weight sharded along N or K, AllReduce via `CollectiveBarrier` |
| `PipelineExecutor` | 1F1B pipeline schedule: per-micro-batch tapes, `backward_with_grad` gradient injection |
| `save/load_safetensors` | Dot-path state dict serialization via `bytemuck` + `safetensors` (zero `unsafe`) |

### Backward Engine

Kahn's algorithm in reverse tape order:
- **Pending map** built by counting input appearances across all tape entries.
- **Strict zero-ready gate** — `pending != 0` means unreachable, skip.
- **Dead-branch decrementing** — ready but no gradient → decrement parents before skip (prevents upstream starvation in branching graphs).
- **Version checks** via `VersionSnapshot::check()` — `Weak` upgrade failure = dead = `Ok(())`.

## Building

The project is a Cargo workspace with six crates:

```
RUMUS/
├── rumus/              # core framework (lib)
├── rumus-macros/       # #[derive(Module)] proc macro (lib)
├── rumus-serve/        # inference server (bin)
├── rumus-graph/        # sparse graph engine for GNNs (lib)
├── rumus-vision/       # spatial CNN engine (lib)
└── rumus-distributed/  # 3D parallelism: TP + PP (lib)
```

```bash
cargo build                        # CPU-only build
cargo build --features gpu         # with WGPU GPU backend
cargo build -p rumus-serve         # inference server
cargo build -p rumus-graph         # graph engine
cargo build -p rumus-vision        # spatial CNN engine
cargo build -p rumus-distributed   # 3D parallelism
cargo test                         # runs all tests (CPU)
cargo test --features gpu          # runs CPU + GPU tests
```

External dependencies: `syn`/`quote`/`proc-macro2` (macro crate),
`safetensors` + `bytemuck` + `parking_lot` + `memmap2` (core crate).
GPU-only (behind `--features gpu`): `wgpu` + `pollster`.
ONNX export (behind `--features onnx`): `prost` + `prost-build`.
Inference server (`rumus-serve`): `axum` + `tokio` + `serde` + `tower-http`.
Graph engine (`rumus-graph`): `rumus` with `gpu` feature + `wgpu` + `bytemuck`.
Distributed (`rumus-distributed`): `rumus` with `gpu` + `multi_gpu` features.

## License

Licensed under either of

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this project by you, as defined in the Apache-2.0 license,
shall be dual licensed as above, without any additional terms or conditions.
