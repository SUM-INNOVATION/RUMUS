// MaxPool2d backward: scatter grad_output to argmax positions.
//
// The combined tensor (from forward) stores LOCAL window indices
// (ky * kernel_w + kx) in the second half — lossless in f16.
// This kernel reconstructs global input coordinates from the local
// index + the output pixel's (oh, ow) position.
//
// Bindings (CustomOp: 2 inputs + 1 output + 1 uniform):
//   0: grad_output       [B * C * H_out * W_out]          (scalar, read)
//   1: combined_forward  [2 * B * C * H_out * W_out]      (scalar, read)
//   2: grad_input        [B * C * H_in * W_in]            (scalar, rw, pre-zeroed)
//   3: params            (uniform)

struct MaxPool2dBwParams {
    num_output_pixels: u32,  // B * C * H_out * W_out
    channels: u32,
    h_in: u32,
    w_in: u32,
    h_out: u32,
    w_out: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    _pad0: u32,
}
// 48 bytes (12 × u32) — multiple of 16 ✓

@group(0) @binding(0) var<storage, read>       mpbw_grad_out: array<scalar>;
@group(0) @binding(1) var<storage, read>       mpbw_combined: array<scalar>;
@group(0) @binding(2) var<storage, read_write> mpbw_grad_in:  array<scalar>;
@group(0) @binding(3) var<uniform>             mpbw_params:   MaxPool2dBwParams;

@compute @workgroup_size(256)
fn maxpool2d_backward_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= mpbw_params.num_output_pixels) { return; }

    let p = mpbw_params;

    // Decompose idx → (b, c, oh, ow).
    let ow = idx % p.w_out;
    let tmp = idx / p.w_out;
    let oh = tmp % p.h_out;
    let tmp2 = tmp / p.h_out;
    let c = tmp2 % p.channels;
    let b = tmp2 / p.channels;

    // Read the LOCAL window index from the second half.
    let local_idx = u32(mpbw_combined[p.num_output_pixels + idx]);

    // Decompose local index → kernel coordinates.
    let ky = local_idx / p.kernel_w;
    let kx = local_idx % p.kernel_w;

    // Reconstruct global input coordinates.
    let ih = oh * p.stride_h + ky - p.pad_h;
    let iw = ow * p.stride_w + kx - p.pad_w;

    // Compute the flat index into grad_input.
    let gi_base = b * p.channels * p.h_in * p.w_in + c * p.h_in * p.w_in;
    let gi_idx = gi_base + ih * p.w_in + iw;

    // Scatter the gradient.  Safe without atomics when stride >= kernel_size.
    mpbw_grad_in[gi_idx] += mpbw_grad_out[idx];
}
