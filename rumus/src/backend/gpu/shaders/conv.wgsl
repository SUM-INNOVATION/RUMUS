// Convolution helper kernels: im2col, col2im, channel-wise bias.
//
// All operate on a SINGLE batch element.  The Rust dispatch layer
// loops over the batch dimension.

struct Im2ColParams {
    c_in: u32,
    h: u32,
    w: u32,
    k: u32,
    stride: u32,
    pad: u32,
    out_h: u32,
    out_w: u32,
}
// 8 * 4 = 32 bytes — multiple of 16 ✓

// --- im2col -----------------------------------------------------------------
// Input:  [C_in, H, W]          (binding 0)
// Output: [col_height, num_patches]  (binding 1)
//   col_height  = C_in * K * K
//   num_patches = out_h * out_w
// Each thread writes one element of the output matrix.

@group(0) @binding(0) var<storage, read>       im2col_input: array<scalar>;
@group(0) @binding(1) var<storage, read_write> im2col_output: array<scalar>;
@group(0) @binding(2) var<uniform>             im2col_params: Im2ColParams;

@compute @workgroup_size(64)
fn im2col_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = im2col_params;
    let col_height = p.c_in * p.k * p.k;
    let num_patches = p.out_h * p.out_w;
    let idx = gid.x;
    if (idx >= col_height * num_patches) { return; }

    let row = idx / num_patches;
    let col = idx % num_patches;

    let kw_val = row % p.k;
    let kh_val = (row / p.k) % p.k;
    let c      = row / (p.k * p.k);

    let oh = col / p.out_w;
    let ow = col % p.out_w;

    let ih_signed: i32 = i32(oh * p.stride + kh_val) - i32(p.pad);
    let iw_signed: i32 = i32(ow * p.stride + kw_val) - i32(p.pad);

    if (ih_signed >= 0 && ih_signed < i32(p.h) && iw_signed >= 0 && iw_signed < i32(p.w)) {
        let ih = u32(ih_signed);
        let iw = u32(iw_signed);
        im2col_output[idx] = im2col_input[c * p.h * p.w + ih * p.w + iw];
    } else {
        im2col_output[idx] = 0.0;
    }
}

// --- col2im -----------------------------------------------------------------
// Input:  [col_height, num_patches]  (binding 0)
// Output: [C_in, H, W]              (binding 1, pre-zeroed)
// Each thread handles one pixel of the output, accumulating from all
// overlapping patches.

@group(0) @binding(0) var<storage, read>       col2im_input: array<scalar>;
@group(0) @binding(1) var<storage, read_write> col2im_output: array<scalar>;
@group(0) @binding(2) var<uniform>             col2im_params: Im2ColParams;

@compute @workgroup_size(64)
fn col2im_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = col2im_params;
    let total = p.c_in * p.h * p.w;
    let idx = gid.x;
    if (idx >= total) { return; }

    let c  = idx / (p.h * p.w);
    let hw = idx % (p.h * p.w);
    let ih = hw / p.w;
    let iw = hw % p.w;

    let num_patches = p.out_h * p.out_w;
    var sum: scalar = scalar(0.0);

    for (var kh: u32 = 0u; kh < p.k; kh++) {
        for (var kw: u32 = 0u; kw < p.k; kw++) {
            let ih_shifted_signed: i32 = i32(ih) + i32(p.pad) - i32(kh);
            let iw_shifted_signed: i32 = i32(iw) + i32(p.pad) - i32(kw);
            if (ih_shifted_signed < 0 || iw_shifted_signed < 0) { continue; }
            let ih_shifted = u32(ih_shifted_signed);
            let iw_shifted = u32(iw_shifted_signed);
            if (ih_shifted % p.stride != 0u || iw_shifted % p.stride != 0u) { continue; }
            let oh = ih_shifted / p.stride;
            let ow = iw_shifted / p.stride;
            if (oh >= p.out_h || ow >= p.out_w) { continue; }

            let row = c * p.k * p.k + kh * p.k + kw;
            let col_idx = oh * p.out_w + ow;
            sum += col2im_input[row * num_patches + col_idx];
        }
    }
    col2im_output[idx] = sum;
}

// --- add_channel_bias -------------------------------------------------------
// src:  [C, spatial]  (binding 0)   where spatial = H*W
// bias: [C]           (binding 1)
// out:  [C, spatial]  (binding 2)
// out[c * spatial + s] = src[c * spatial + s] + bias[c]

struct ChannelBiasParams {
    channels: u32,
    spatial: u32,
    _pad0: u32,
    _pad1: u32,
}
// 4 * 4 = 16 bytes ✓

@group(0) @binding(0) var<storage, read>       chb_src:  array<scalar>;
@group(0) @binding(1) var<storage, read>       chb_bias: array<scalar>;
@group(0) @binding(2) var<storage, read_write> chb_out:  array<scalar>;
@group(0) @binding(3) var<uniform>             chb_params: ChannelBiasParams;

@compute @workgroup_size(64)
fn add_channel_bias_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = chb_params.channels * chb_params.spatial;
    let idx = gid.x;
    if (idx >= total) { return; }
    let c = idx / chb_params.spatial;
    chb_out[idx] = chb_src[idx] + chb_bias[c];
}

// --- sum_channel_bias_grad --------------------------------------------------
// Reduces [C, spatial] → [C] by summing over the spatial dimension.
// Each thread handles one channel.

@group(0) @binding(0) var<storage, read>       scbg_src: array<scalar>;
@group(0) @binding(1) var<storage, read>       scbg_dummy: array<scalar>;
@group(0) @binding(2) var<storage, read_write> scbg_out: array<scalar>;
@group(0) @binding(3) var<uniform>             scbg_params: ChannelBiasParams;

@compute @workgroup_size(64)
fn sum_channel_bias_grad_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= scbg_params.channels) { return; }
    var sum: scalar = scalar(0.0);
    for (var s: u32 = 0u; s < scbg_params.spatial; s++) {
        sum += scbg_src[c * scbg_params.spatial + s];
    }
    scbg_out[c] = sum;
}
