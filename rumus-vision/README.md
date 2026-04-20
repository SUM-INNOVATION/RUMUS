# rumus-vision

Spatial CNN engine for the **RUMUS** deep learning framework — direct sliding-window convolution and max-pooling via WebGPU. Zero im2col VRAM overhead.

## Why

Standard im2col convolution unfolds image patches into a matrix, consuming `C_in × K² × H_out × W_out × 4` bytes of intermediate VRAM per batch element. For a 224×224 image with 64 channels and 3×3 kernel, that's ~28MB per sample. `rumus-vision` computes each output pixel in-place via nested loops — zero intermediate buffers.

## Operations

### Conv2d (Direct Sliding-Window)

```text
output[b, co, oh, ow] = Σ_{ci, ky, kx} input[b, ci, ih, iw] × weight[co, ci, ky, kx] + bias[co]
    where ih = oh × stride_h + ky × dilation_h - pad_h
```

- 1 thread = 1 output pixel, `workgroup_size(256)`
- Triple nested loop: `C_in × K_h × K_w`
- Supports stride, padding, dilation
- Three backward kernels:
  - **grad_input** (`conv2d_backward_data`): transposed convolution
  - **grad_weight** (`conv2d_backward_weight`): accumulation over batch × spatial
  - **grad_bias**: channel-wise reduction

### MaxPool2d (F16-Safe Local Argmax)

- 1 thread = 1 output pixel
- Concatenated output: first half = max values, second half = local window indices
- **Local window index** `ky × kernel_w + kx` (bounded by K², lossless in f16)
- Backward reconstructs global coordinates from local index + output position
- `assert!(K_h × K_w ≤ 2048)` enforces the f16 precision bound at the Rust API level
- Tracked `slice_range` + `reshape_tracked` preserve the autograd chain

## Quick Start

```rust
use rumus_vision::ops;
use rumus::tensor::Tensor;

// Conv2d
let input = Tensor::new(vec![0.0; 1 * 3 * 32 * 32], vec![1, 3, 32, 32]);
let weight = Tensor::new(vec![0.01; 16 * 3 * 3 * 3], vec![16, 3, 3, 3]);
let bias = Tensor::new(vec![0.0; 16], vec![16]);
let output = ops::conv2d(&input, &weight, Some(&bias), (1,1), (1,1), (1,1));
// output: [1, 16, 32, 32]

// MaxPool2d
let pooled = ops::max_pool2d(&output, (2,2), (2,2), (0,0));
// pooled: [1, 16, 16, 16]
```

## API

### `conv2d(input, weight, bias, stride, padding, dilation) -> Tensor`

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `&Tensor` | `[B, C_in, H_in, W_in]` |
| `weight` | `&Tensor` | `[C_out, C_in, K_h, K_w]` |
| `bias` | `Option<&Tensor>` | `[C_out]` or `None` |
| `stride` | `(usize, usize)` | `(stride_h, stride_w)` |
| `padding` | `(usize, usize)` | `(pad_h, pad_w)` |
| `dilation` | `(usize, usize)` | `(dilation_h, dilation_w)` |

### `max_pool2d(input, kernel_size, stride, padding) -> Tensor`

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `&Tensor` | `[B, C, H_in, W_in]` |
| `kernel_size` | `(usize, usize)` | `(K_h, K_w)` — must satisfy `K_h × K_w ≤ 2048` |
| `stride` | `(usize, usize)` | `(stride_h, stride_w)` |
| `padding` | `(usize, usize)` | `(pad_h, pad_w)` |

## WGSL Shaders

| Shader | Bindings | Thread Mapping |
|--------|----------|----------------|
| `conv2d_direct.wgsl` | input + weight + bias + output + params (64B) | 1 thread / output pixel |
| `conv2d_backward_data.wgsl` | grad_out + weight + grad_in + params (64B) | 1 thread / input pixel |
| `conv2d_backward_weight.wgsl` | grad_out + input + grad_w + params (64B) | 1 thread / weight element |
| `maxpool2d_direct.wgsl` | input + combined_output + params (64B) | 1 thread / output pixel |
| `maxpool2d_backward.wgsl` | grad_out + combined + grad_in + params (48B) | 1 thread / output pixel |

### INT4 Weight-Only Quantization (AWQ/GPTQ)

8x compression vs F32 for serving massive LLMs on limited VRAM.

- `QuantizedTensor`: packs 8 × 4-bit unsigned weights per `u32`, K-major layout
- K padded to `group_size` (must be multiple of 8, e.g., 128) — clean word boundaries
- Group-wise `scales` and `zero_points` as `[num_groups, N]` f32, grouped along K
- `qmatmul_int4.wgsl`: fused dequant-matmul — unpacks 8 nibbles per u32 via `(packed >> (j*4)) & 0xF`, dequantizes `(q - zp) × scale`, accumulates dot product in registers
- `qmatmul_int4_transpose.wgsl`: transpose variant for activation gradients (`grad_x`)
- `QLinear::from_linear(linear, group_size)`: quantizes a pre-trained `Linear` layer
- Frozen weights (`AutogradState::None`) + optional trainable bias (LoRA-compatible)
- `QLinearBackward` returns only `grad_x` — frozen INT4 inputs get no gradients

```rust
use rumus_vision::QLinear;

let qlinear = QLinear::from_linear(&pretrained_linear, 128);
let output = qlinear.forward(&input);  // fused INT4 dequant-matmul
```

## Dependencies

- `rumus` (v0.3.0) — core framework with GPU backend and M19 Custom Ops API
- `wgpu` — WebGPU buffer types
- `bytemuck` — safe byte reinterpretation

## License

Licensed under either of

- [Apache License, Version 2.0](../LICENSE-APACHE)
- [MIT License](../LICENSE-MIT)

at your option.
