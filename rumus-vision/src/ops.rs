// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Conv2d and MaxPool2d operations via the RUMUS CustomOp plugin API.

use std::sync::Arc;

use rumus::autograd::CustomBackward;
use rumus::ext::{self, CustomOp};
use rumus::tensor::Tensor;

// Embedded WGSL sources.
const CONV2D_FWD_WGSL: &str = include_str!("shaders/conv2d_direct.wgsl");
const CONV2D_BWD_DATA_WGSL: &str = include_str!("shaders/conv2d_backward_data.wgsl");
const CONV2D_BWD_WEIGHT_WGSL: &str = include_str!("shaders/conv2d_backward_weight.wgsl");
const MAXPOOL2D_FWD_WGSL: &str = include_str!("shaders/maxpool2d_direct.wgsl");
const MAXPOOL2D_BWD_WGSL: &str = include_str!("shaders/maxpool2d_backward.wgsl");

// =====================================================================
// Conv2d
// =====================================================================

/// Convolution hyperparameters.
#[derive(Clone, Debug)]
pub struct ConvParams {
    pub batch: usize,
    pub c_in: usize,
    pub c_out: usize,
    pub h_in: usize,
    pub w_in: usize,
    pub h_out: usize,
    pub w_out: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_h: usize,
    pub pad_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
    pub has_bias: bool,
}

fn pack_conv2d_fwd_uniform(p: &ConvParams) -> Vec<u8> {
    let total = p.batch * p.c_out * p.h_out * p.w_out;
    let vals: [u32; 16] = [
        total as u32, p.c_in as u32, p.c_out as u32, p.h_in as u32,
        p.w_in as u32, p.h_out as u32, p.w_out as u32, p.kernel_h as u32,
        p.kernel_w as u32, p.stride_h as u32, p.stride_w as u32, p.pad_h as u32,
        p.pad_w as u32, p.dilation_h as u32, p.dilation_w as u32,
        if p.has_bias { 1 } else { 0 },
    ];
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

// --- Forward ---

struct Conv2dFwdOp {
    params: ConvParams,
    backward: Arc<Conv2dBackward>,
}

impl CustomOp for Conv2dFwdOp {
    fn op_name(&self) -> &str { "rumus_vision_conv2d_fwd" }
    fn wgsl_source(&self) -> &str { CONV2D_FWD_WGSL }
    fn entry_point(&self) -> &str { "conv2d_forward_kernel" }
    fn num_inputs(&self) -> usize { 3 }
    fn output_shape(&self, _: &[&[usize]]) -> Vec<usize> {
        let p = &self.params;
        vec![p.batch, p.c_out, p.h_out, p.w_out]
    }
    fn dispatch(&self, _: usize) -> (u32, u32, u32) {
        let n = self.params.batch * self.params.c_out * self.params.h_out * self.params.w_out;
        ((n as u32 + 255) / 256, 1, 1)
    }
    fn uniform_data(&self, _: &[&Tensor]) -> Vec<u8> {
        pack_conv2d_fwd_uniform(&self.params)
    }
    fn backward_handler(&self) -> Option<Arc<dyn CustomBackward>> {
        Some(self.backward.clone())
    }
    fn save_for_backward<'a>(&self, inputs: &[&'a Tensor], _: &'a Tensor) -> Vec<&'a Tensor> {
        vec![inputs[0], inputs[1]]  // save input and weight
    }
}

// --- Backward: grad_input ---

struct Conv2dBwdDataOp { params: ConvParams }

impl CustomOp for Conv2dBwdDataOp {
    fn op_name(&self) -> &str { "rumus_vision_conv2d_bwd_data" }
    fn wgsl_source(&self) -> &str { CONV2D_BWD_DATA_WGSL }
    fn entry_point(&self) -> &str { "conv2d_backward_data_kernel" }
    fn num_inputs(&self) -> usize { 2 }
    fn output_shape(&self, _: &[&[usize]]) -> Vec<usize> {
        let p = &self.params;
        vec![p.batch, p.c_in, p.h_in, p.w_in]
    }
    fn dispatch(&self, _: usize) -> (u32, u32, u32) {
        let n = self.params.batch * self.params.c_in * self.params.h_in * self.params.w_in;
        ((n as u32 + 255) / 256, 1, 1)
    }
    fn uniform_data(&self, _: &[&Tensor]) -> Vec<u8> {
        let p = &self.params;
        let total = p.batch * p.c_in * p.h_in * p.w_in;
        let vals: [u32; 16] = [
            total as u32, p.c_in as u32, p.c_out as u32, p.h_in as u32,
            p.w_in as u32, p.h_out as u32, p.w_out as u32, p.kernel_h as u32,
            p.kernel_w as u32, p.stride_h as u32, p.stride_w as u32, p.pad_h as u32,
            p.pad_w as u32, p.dilation_h as u32, p.dilation_w as u32, 0,
        ];
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

// --- Backward: grad_weight ---

struct Conv2dBwdWeightOp { params: ConvParams }

impl CustomOp for Conv2dBwdWeightOp {
    fn op_name(&self) -> &str { "rumus_vision_conv2d_bwd_weight" }
    fn wgsl_source(&self) -> &str { CONV2D_BWD_WEIGHT_WGSL }
    fn entry_point(&self) -> &str { "conv2d_backward_weight_kernel" }
    fn num_inputs(&self) -> usize { 2 }
    fn output_shape(&self, _: &[&[usize]]) -> Vec<usize> {
        let p = &self.params;
        vec![p.c_out, p.c_in, p.kernel_h, p.kernel_w]
    }
    fn dispatch(&self, _: usize) -> (u32, u32, u32) {
        let n = self.params.c_out * self.params.c_in * self.params.kernel_h * self.params.kernel_w;
        ((n as u32 + 255) / 256, 1, 1)
    }
    fn uniform_data(&self, _: &[&Tensor]) -> Vec<u8> {
        let p = &self.params;
        let total = p.c_out * p.c_in * p.kernel_h * p.kernel_w;
        let vals: [u32; 16] = [
            total as u32, p.batch as u32, p.c_in as u32, p.c_out as u32,
            p.h_in as u32, p.w_in as u32, p.h_out as u32, p.w_out as u32,
            p.kernel_h as u32, p.kernel_w as u32, p.stride_h as u32, p.stride_w as u32,
            p.pad_h as u32, p.pad_w as u32, p.dilation_h as u32, p.dilation_w as u32,
        ];
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

// --- Backward: grad_bias ---

struct Conv2dBwdBiasOp { c_out: usize, total_spatial: usize }

impl CustomOp for Conv2dBwdBiasOp {
    fn op_name(&self) -> &str { "rumus_vision_conv2d_bwd_bias" }
    fn wgsl_source(&self) -> &str {
        // Inline: 1 thread per output channel, sums over batch × spatial.
        "struct P { c_out: u32, spatial: u32, _p0: u32, _p1: u32, }
@group(0) @binding(0) var<storage, read> gb_in: array<scalar>;
@group(0) @binding(1) var<storage, read_write> gb_out: array<scalar>;
@group(0) @binding(2) var<uniform> gb_p: P;
@compute @workgroup_size(256)
fn conv2d_bwd_bias_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let co = gid.x;
    if (co >= gb_p.c_out) { return; }
    var s: scalar = scalar(0.0);
    for (var i: u32 = 0u; i < gb_p.spatial; i++) {
        s += gb_in[i * gb_p.c_out + co];
    }
    gb_out[co] = s;
}"
    }
    fn entry_point(&self) -> &str { "conv2d_bwd_bias_kernel" }
    fn num_inputs(&self) -> usize { 1 }
    fn output_shape(&self, _: &[&[usize]]) -> Vec<usize> { vec![self.c_out] }
    fn dispatch(&self, _: usize) -> (u32, u32, u32) {
        ((self.c_out as u32 + 255) / 256, 1, 1)
    }
    fn uniform_data(&self, _: &[&Tensor]) -> Vec<u8> {
        let vals: [u32; 4] = [self.c_out as u32, self.total_spatial as u32, 0, 0];
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

// --- Conv2dBackward ---

#[derive(Debug)]
struct Conv2dBackward { params: ConvParams }

impl CustomBackward for Conv2dBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let saved_input = &saved[0];
        let saved_weight = &saved[1];

        let grad_input = ext::custom_forward(
            &Conv2dBwdDataOp { params: self.params.clone() },
            &[grad_output, saved_weight],
        );

        let grad_weight = ext::custom_forward(
            &Conv2dBwdWeightOp { params: self.params.clone() },
            &[grad_output, saved_input],
        );

        if self.params.has_bias {
            // Reshape grad_output to [B*H_out*W_out, C_out] for the bias reduction.
            let p = &self.params;
            let spatial = p.batch * p.h_out * p.w_out;
            let go_flat = grad_output.reshape(vec![spatial, p.c_out]);
            let grad_bias = ext::custom_forward(
                &Conv2dBwdBiasOp { c_out: p.c_out, total_spatial: spatial },
                &[&go_flat],
            );
            vec![grad_input, grad_weight, grad_bias]
        } else {
            let zero = Tensor::new(vec![0.0], vec![1]);
            vec![grad_input, grad_weight, zero]
        }
    }
}

// =====================================================================
// MaxPool2d
// =====================================================================

#[derive(Clone, Debug)]
pub struct PoolParams {
    pub batch: usize,
    pub channels: usize,
    pub h_in: usize,
    pub w_in: usize,
    pub h_out: usize,
    pub w_out: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_h: usize,
    pub pad_w: usize,
}

// --- Forward ---

struct MaxPool2dFwdOp {
    params: PoolParams,
    backward: Arc<MaxPool2dBackward>,
}

impl CustomOp for MaxPool2dFwdOp {
    fn op_name(&self) -> &str { "rumus_vision_maxpool2d_fwd" }
    fn wgsl_source(&self) -> &str { MAXPOOL2D_FWD_WGSL }
    fn entry_point(&self) -> &str { "maxpool2d_forward_kernel" }
    fn num_inputs(&self) -> usize { 1 }
    fn output_shape(&self, _: &[&[usize]]) -> Vec<usize> {
        let p = &self.params;
        // Concatenated: first half = values, second half = argmax.
        vec![2 * p.batch * p.channels * p.h_out * p.w_out]
    }
    fn dispatch(&self, _: usize) -> (u32, u32, u32) {
        let n = self.params.batch * self.params.channels * self.params.h_out * self.params.w_out;
        ((n as u32 + 255) / 256, 1, 1)
    }
    fn uniform_data(&self, _: &[&Tensor]) -> Vec<u8> {
        let p = &self.params;
        let n = p.batch * p.channels * p.h_out * p.w_out;
        let vals: [u32; 16] = [
            n as u32, p.channels as u32, p.h_in as u32, p.w_in as u32,
            p.h_out as u32, p.w_out as u32, p.kernel_h as u32, p.kernel_w as u32,
            p.stride_h as u32, p.stride_w as u32, p.pad_h as u32, p.pad_w as u32,
            0, 0, 0, 0,
        ];
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
    fn backward_handler(&self) -> Option<Arc<dyn CustomBackward>> {
        Some(self.backward.clone())
    }
    fn save_for_backward<'a>(&self, _: &[&'a Tensor], output: &'a Tensor) -> Vec<&'a Tensor> {
        // Save the ENTIRE combined output (values + argmax).
        vec![output]
    }
}

// --- Backward ---

struct MaxPool2dBwdOp { params: PoolParams }

impl CustomOp for MaxPool2dBwdOp {
    fn op_name(&self) -> &str { "rumus_vision_maxpool2d_bwd" }
    fn wgsl_source(&self) -> &str { MAXPOOL2D_BWD_WGSL }
    fn entry_point(&self) -> &str { "maxpool2d_backward_kernel" }
    fn num_inputs(&self) -> usize { 2 }  // grad_output, combined_forward
    fn output_shape(&self, _: &[&[usize]]) -> Vec<usize> {
        let p = &self.params;
        vec![p.batch * p.channels * p.h_in * p.w_in]
    }
    fn dispatch(&self, _: usize) -> (u32, u32, u32) {
        let n = self.params.batch * self.params.channels * self.params.h_out * self.params.w_out;
        ((n as u32 + 255) / 256, 1, 1)
    }
    fn uniform_data(&self, _: &[&Tensor]) -> Vec<u8> {
        let p = &self.params;
        let n = p.batch * p.channels * p.h_out * p.w_out;
        // Must match MaxPool2dBwParams: num_output_pixels, channels, h_in, w_in,
        // h_out, w_out, kernel_w, stride_h, stride_w, pad_h, pad_w, _pad0.
        let vals: [u32; 12] = [
            n as u32, p.channels as u32, p.h_in as u32, p.w_in as u32,
            p.h_out as u32, p.w_out as u32, p.kernel_w as u32,
            p.stride_h as u32, p.stride_w as u32, p.pad_h as u32, p.pad_w as u32, 0,
        ];
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

#[derive(Debug)]
struct MaxPool2dBackward { params: PoolParams }

impl CustomBackward for MaxPool2dBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        // saved[0] = combined forward output (values + argmax indices).
        let combined = &saved[0];
        let grad_input = ext::custom_forward(
            &MaxPool2dBwdOp { params: self.params.clone() },
            &[grad_output, combined],
        );
        vec![grad_input]
    }
}

// =====================================================================
// Public API
// =====================================================================

/// Direct 2D convolution (no im2col — zero intermediate VRAM).
///
/// `input`:  `[B, C_in, H_in, W_in]`
/// `weight`: `[C_out, C_in, K_h, K_w]`
/// `bias`:   `[C_out]` or `None`
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Tensor {
    let shape = input.shape();
    let (batch, c_in, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
    let wshape = weight.shape();
    let (c_out, kernel_h, kernel_w) = (wshape[0], wshape[2], wshape[3]);

    let h_out = (h_in + 2 * padding.0 - dilation.0 * (kernel_h - 1) - 1) / stride.0 + 1;
    let w_out = (w_in + 2 * padding.1 - dilation.1 * (kernel_w - 1) - 1) / stride.1 + 1;

    let has_bias = bias.is_some();
    let bias_tensor = bias.cloned().unwrap_or_else(|| Tensor::new(vec![0.0], vec![1]));

    let params = ConvParams {
        batch, c_in, c_out, h_in, w_in, h_out, w_out,
        kernel_h, kernel_w,
        stride_h: stride.0, stride_w: stride.1,
        pad_h: padding.0, pad_w: padding.1,
        dilation_h: dilation.0, dilation_w: dilation.1,
        has_bias,
    };

    let backward = Arc::new(Conv2dBackward { params: params.clone() });
    let op = Conv2dFwdOp { params, backward };

    ext::custom_forward(&op, &[input, weight, &bias_tensor])
}

/// Max pooling 2D.
///
/// `input`: `[B, C, H_in, W_in]`
/// Returns: `[B, C, H_out, W_out]` (max values only; argmax is internal).
pub fn max_pool2d(
    input: &Tensor,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Tensor {
    assert!(
        kernel_size.0 * kernel_size.1 <= 2048,
        "MaxPool2d: kernel window size {}x{} = {} exceeds the f16-safe argmax limit (2048)",
        kernel_size.0, kernel_size.1, kernel_size.0 * kernel_size.1,
    );

    let shape = input.shape();
    let (batch, channels, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);

    let h_out = (h_in + 2 * padding.0 - kernel_size.0) / stride.0 + 1;
    let w_out = (w_in + 2 * padding.1 - kernel_size.1) / stride.1 + 1;

    let params = PoolParams {
        batch, channels, h_in, w_in, h_out, w_out,
        kernel_h: kernel_size.0, kernel_w: kernel_size.1,
        stride_h: stride.0, stride_w: stride.1,
        pad_h: padding.0, pad_w: padding.1,
    };

    let backward = Arc::new(MaxPool2dBackward { params: params.clone() });
    let op = MaxPool2dFwdOp { params, backward };

    // custom_forward returns the combined [2 * N] tensor (values + argmax).
    let combined = ext::custom_forward(&op, &[input]);

    // Slice out the max values (first half) using TRACKED slice_range
    // (records SliceRangeBackward — preserves the autograd chain).
    let n = batch * channels * h_out * w_out;
    let values = combined.slice_range(0, 0, n);

    // Reshape using TRACKED reshape_tracked (records ReshapeBackward —
    // preserves the autograd chain back through slice_range → CustomOp).
    values.reshape_tracked(vec![batch, channels, h_out, w_out])
}
