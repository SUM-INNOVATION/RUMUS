// SPDX-License-Identifier: Apache-2.0 OR MIT
//! INT4 weight-only quantization: QuantizedTensor and QLinear.

use std::sync::Arc;

use rumus::autograd::CustomBackward;
use rumus::backend::gpu::context::{GpuContext, STORAGE_USAGE};
use rumus::ext::{self, CustomOp};
use rumus::nn::Parameter;
use rumus::tensor::{AutogradState, Layout, StorageHandle, Tensor};

// Embedded WGSL.
const QMATMUL_INT4_WGSL: &str = include_str!("shaders/qmatmul_int4.wgsl");
const QMATMUL_INT4_T_WGSL: &str = include_str!("shaders/qmatmul_int4_transpose.wgsl");

// ---------------------------------------------------------------------------
// QuantizedTensor
// ---------------------------------------------------------------------------

/// INT4 weight tensor with group-wise quantization metadata.
///
/// Weights packed as 8 × 4-bit unsigned values per `u32`, grouped along K.
/// K is padded to the nearest multiple of `group_size`.
pub struct QuantizedTensor {
    pub k: usize,           // original (unpadded) K
    pub n: usize,
    pub padded_k: usize,
    pub group_size: usize,
    pub num_groups: usize,  // padded_k / group_size
    // GPU buffers (owned, returned to BufferPool on Drop).
    packed_buf: wgpu::Buffer,
    scales_buf: wgpu::Buffer,
    zp_buf: wgpu::Buffer,
    // Tensor wrappers with AutogradState::None (frozen — no gradients).
    pub packed_tensor: Tensor,
    pub scales_tensor: Tensor,
    pub zp_tensor: Tensor,
}

impl QuantizedTensor {
    /// Quantize an f32 weight matrix `[K, N]` to INT4 with group-wise metadata.
    pub fn from_f32(data: &[f32], k: usize, n: usize, group_size: usize) -> Self {
        assert!(group_size % 8 == 0, "group_size must be a multiple of 8");
        assert!(k * n == data.len(), "data length mismatch");

        let padded_k = ((k + group_size - 1) / group_size) * group_size;
        let num_groups = padded_k / group_size;
        let num_words = (padded_k / 8) * n;

        // Pad the weight data along K with zeros.
        let mut padded = vec![0.0f32; padded_k * n];
        for row in 0..k {
            for col in 0..n {
                padded[row * n + col] = data[row * n + col];
            }
        }

        // Compute per-group scales and zero-points.
        let mut scales = vec![0.0f32; num_groups * n];
        let mut zero_points = vec![0.0f32; num_groups * n];

        for col in 0..n {
            for g in 0..num_groups {
                let start = g * group_size;
                let end = start + group_size;

                // Find min and max in this group.
                let mut min_val = f32::MAX;
                let mut max_val = f32::MIN;
                for row in start..end {
                    let v = padded[row * n + col];
                    if v < min_val { min_val = v; }
                    if v > max_val { max_val = v; }
                }

                // Asymmetric quantization: map [min, max] → [0, 15].
                let range = max_val - min_val;
                let scale = if range > 1e-10 { range / 15.0 } else { 1.0 };
                let zp = if range > 1e-10 { -min_val / scale } else { 0.0 };

                scales[g * n + col] = scale;
                zero_points[g * n + col] = zp;
            }
        }

        // Pack weights into u32 words (8 INT4 per word, K-major).
        let mut packed = vec![0u32; num_words];
        for col in 0..n {
            for row in 0..padded_k {
                let g = row / group_size;
                let scale = scales[g * n + col];
                let zp = zero_points[g * n + col];
                let val = padded[row * n + col];

                // Quantize: q = clamp(round(val / scale + zp), 0, 15).
                let q = ((val / scale + zp).round() as i32).clamp(0, 15) as u32;

                let word_idx = (row / 8) * n + col;
                let nibble = row % 8;
                packed[word_idx] |= q << (nibble as u32 * 4);
            }
        }

        // Upload to GPU.
        let ctx = GpuContext::get().expect("GPU required for INT4 quantization");

        let packed_buf = ctx.pool.acquire(&ctx.device, (num_words * 4) as u64, STORAGE_USAGE);
        ctx.queue.write_buffer(&packed_buf, 0, bytemuck::cast_slice(&packed));

        let scales_buf = ctx.pool.acquire(&ctx.device, (num_groups * n * 4) as u64, STORAGE_USAGE);
        ctx.queue.write_buffer(&scales_buf, 0, bytemuck::cast_slice(&scales));

        let zp_buf = ctx.pool.acquire(&ctx.device, (num_groups * n * 4) as u64, STORAGE_USAGE);
        ctx.queue.write_buffer(&zp_buf, 0, bytemuck::cast_slice(&zero_points));

        // Create Tensor wrappers with AutogradState::None (frozen).
        let packed_tensor = make_frozen_tensor(&packed_buf, num_words);
        let scales_tensor = make_frozen_tensor(&scales_buf, num_groups * n);
        let zp_tensor = make_frozen_tensor(&zp_buf, num_groups * n);

        Self {
            k, n, padded_k, group_size, num_groups,
            packed_buf, scales_buf, zp_buf,
            packed_tensor, scales_tensor, zp_tensor,
        }
    }
}

impl Drop for QuantizedTensor {
    fn drop(&mut self) {
        if let Some(ctx) = GpuContext::get() {
            // Swap out buffers with tiny placeholders and release to pool.
            let placeholder = || ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: 4, usage: STORAGE_USAGE, mapped_at_creation: false,
            });
            ctx.pool.release(std::mem::replace(&mut self.packed_buf, placeholder()));
            ctx.pool.release(std::mem::replace(&mut self.scales_buf, placeholder()));
            ctx.pool.release(std::mem::replace(&mut self.zp_buf, placeholder()));
        }
    }
}

/// Create a GPU Tensor with AutogradState::None (no gradients).
fn make_frozen_tensor(buf: &wgpu::Buffer, num_elements: usize) -> Tensor {
    // Download raw bytes → reinterpret as f32 → upload as new Tensor.
    // This creates an independent StorageHandle (doesn't alias the source buf).
    let ctx = GpuContext::get().expect("GPU required");
    let byte_size = (num_elements * 4) as u64;
    let raw = ctx.download_raw_bytes(buf, byte_size);
    let f32_data: Vec<f32> = bytemuck::cast_slice(&raw).to_vec();
    let t = Tensor::new(f32_data, vec![num_elements]);
    t.to_gpu();
    t
    // Tensor::new sets state = AutogradState::None by default.
}

// ---------------------------------------------------------------------------
// QLinear
// ---------------------------------------------------------------------------

/// Linear layer with frozen INT4 weights and optional trainable bias.
///
/// Forward: `y = x @ dequant(W_int4)^T + bias`
/// (the kernel fuses dequantization into the matmul — no intermediate buffer).
pub struct QLinear {
    pub weight: QuantizedTensor,
    pub bias: Option<Parameter>,
    pub in_features: usize,   // K
    pub out_features: usize,  // N
}

impl QLinear {
    /// Quantize a pre-trained `Linear` layer to INT4.
    pub fn from_linear(linear: &rumus::nn::Linear, group_size: usize) -> Self {
        let k = linear.weight.tensor.shape()[0];  // [in, out]
        let n = linear.weight.tensor.shape()[1];

        let guard = linear.weight.tensor.data();
        let weight = QuantizedTensor::from_f32(&guard, k, n, group_size);
        drop(guard);

        Self {
            weight,
            bias: linear.bias.clone(),
            in_features: k,
            out_features: n,
        }
    }

    /// Forward pass: fused INT4 dequant-matmul + optional bias.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let m = x.shape()[0];

        let backward = Arc::new(QLinearBackward {
            packed: self.weight.packed_tensor.clone(),
            scales: self.weight.scales_tensor.clone(),
            zp: self.weight.zp_tensor.clone(),
            m, k: self.in_features, n: self.out_features,
            padded_k: self.weight.padded_k,
            group_size: self.weight.group_size,
            num_groups: self.weight.num_groups,
        });

        let op = QMatMulINT4FwdOp {
            m, k: self.in_features, n: self.out_features,
            padded_k: self.weight.padded_k,
            group_size: self.weight.group_size,
            num_groups: self.weight.num_groups,
            backward,
        };

        let y = ext::custom_forward(
            &op,
            &[x, &self.weight.packed_tensor, &self.weight.scales_tensor, &self.weight.zp_tensor],
        );

        match &self.bias {
            Some(bias) => y.add_bias(&bias.tensor),
            None => y,
        }
    }
}

// ---------------------------------------------------------------------------
// CustomOp: Forward
// ---------------------------------------------------------------------------

struct QMatMulINT4FwdOp {
    m: usize, k: usize, n: usize,
    padded_k: usize, group_size: usize, num_groups: usize,
    backward: Arc<QLinearBackward>,
}

impl CustomOp for QMatMulINT4FwdOp {
    fn op_name(&self) -> &str { "rumus_vision_qmatmul_int4" }
    fn wgsl_source(&self) -> &str { QMATMUL_INT4_WGSL }
    fn entry_point(&self) -> &str { "qmatmul_int4_kernel" }
    fn num_inputs(&self) -> usize { 4 }
    fn output_shape(&self, _: &[&[usize]]) -> Vec<usize> { vec![self.m, self.n] }
    fn dispatch(&self, _: usize) -> (u32, u32, u32) {
        ((self.n as u32 + 15) / 16, (self.m as u32 + 15) / 16, 1)
    }
    fn uniform_data(&self, _: &[&Tensor]) -> Vec<u8> {
        let vals: [u32; 8] = [
            self.m as u32, self.k as u32, self.n as u32, self.padded_k as u32,
            self.group_size as u32, self.num_groups as u32, 0, 0,
        ];
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
    fn backward_handler(&self) -> Option<Arc<dyn CustomBackward>> {
        Some(self.backward.clone())
    }
    fn save_for_backward<'a>(&self, _: &[&'a Tensor], _: &'a Tensor) -> Vec<&'a Tensor> {
        vec![]  // backward holds its own references to the frozen weight buffers
    }
}

// ---------------------------------------------------------------------------
// CustomOp: Transpose (for grad_x)
// ---------------------------------------------------------------------------

struct QMatMulINT4TransposeOp {
    m: usize, k: usize, n: usize,
    padded_k: usize, group_size: usize, num_groups: usize,
}

impl CustomOp for QMatMulINT4TransposeOp {
    fn op_name(&self) -> &str { "rumus_vision_qmatmul_int4_t" }
    fn wgsl_source(&self) -> &str { QMATMUL_INT4_T_WGSL }
    fn entry_point(&self) -> &str { "qmatmul_int4_transpose_kernel" }
    fn num_inputs(&self) -> usize { 4 }
    fn output_shape(&self, _: &[&[usize]]) -> Vec<usize> { vec![self.m, self.k] }
    fn dispatch(&self, _: usize) -> (u32, u32, u32) {
        ((self.k as u32 + 15) / 16, (self.m as u32 + 15) / 16, 1)
    }
    fn uniform_data(&self, _: &[&Tensor]) -> Vec<u8> {
        let vals: [u32; 8] = [
            self.m as u32, self.k as u32, self.n as u32, self.padded_k as u32,
            self.group_size as u32, self.num_groups as u32, 0, 0,
        ];
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

// ---------------------------------------------------------------------------
// CustomBackward: grad_x only (frozen weights get no gradients)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct QLinearBackward {
    packed: Tensor,
    scales: Tensor,
    zp: Tensor,
    m: usize, k: usize, n: usize,
    padded_k: usize, group_size: usize, num_groups: usize,
}

impl CustomBackward for QLinearBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Vec<Tensor> {
        let m = grad_output.shape()[0];

        // grad_x = grad_y @ dequant(W)^T
        let grad_x = ext::custom_forward(
            &QMatMulINT4TransposeOp {
                m, k: self.k, n: self.n,
                padded_k: self.padded_k,
                group_size: self.group_size,
                num_groups: self.num_groups,
            },
            &[grad_output, &self.packed, &self.scales, &self.zp],
        );

        // Return exactly 1 gradient (for the activation input at position 0).
        // Inputs 1..3 (packed, scales, zp) have AutogradState::None — the
        // backward engine skips them because vec length < entry.inputs.len().
        vec![grad_x]
    }
}
