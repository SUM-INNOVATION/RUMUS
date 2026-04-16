// Direct sliding-window 2D convolution.
//
// 1 thread = 1 output pixel.  No im2col — zero intermediate VRAM.
// Handles stride, padding, and dilation.
//
// Bindings (CustomOp: 3 inputs + 1 output + 1 uniform):
//   0: input   [B, C_in, H_in, W_in]    (scalar, read)
//   1: weight  [C_out, C_in, K_h, K_w]  (scalar, read)
//   2: bias    [C_out] or [1] dummy      (scalar, read)
//   3: output  [B, C_out, H_out, W_out]  (scalar, rw)
//   4: params  (uniform)

struct Conv2dParams {
    total_output: u32,
    c_in: u32,
    c_out: u32,
    h_in: u32,
    w_in: u32,
    h_out: u32,
    w_out: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    has_bias: u32,
}
// 64 bytes (16 × u32) ✓

@group(0) @binding(0) var<storage, read>       conv_input:  array<scalar>;
@group(0) @binding(1) var<storage, read>       conv_weight: array<scalar>;
@group(0) @binding(2) var<storage, read>       conv_bias:   array<scalar>;
@group(0) @binding(3) var<storage, read_write> conv_output: array<scalar>;
@group(0) @binding(4) var<uniform>             conv_params: Conv2dParams;

@compute @workgroup_size(256)
fn conv2d_forward_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= conv_params.total_output) { return; }

    let p = conv_params;
    let ow = idx % p.w_out;
    let tmp = idx / p.w_out;
    let oh = tmp % p.h_out;
    let tmp2 = tmp / p.h_out;
    let co = tmp2 % p.c_out;
    let b = tmp2 / p.c_out;

    var sum: scalar = scalar(0.0);

    for (var ci: u32 = 0u; ci < p.c_in; ci++) {
        for (var ky: u32 = 0u; ky < p.kernel_h; ky++) {
            for (var kx: u32 = 0u; kx < p.kernel_w; kx++) {
                let ih_s = i32(oh * p.stride_h + ky * p.dilation_h) - i32(p.pad_h);
                let iw_s = i32(ow * p.stride_w + kx * p.dilation_w) - i32(p.pad_w);

                if (ih_s >= 0 && ih_s < i32(p.h_in) && iw_s >= 0 && iw_s < i32(p.w_in)) {
                    let ih = u32(ih_s);
                    let iw = u32(iw_s);
                    let in_idx = b * p.c_in * p.h_in * p.w_in
                               + ci * p.h_in * p.w_in
                               + ih * p.w_in + iw;
                    let w_idx = co * p.c_in * p.kernel_h * p.kernel_w
                              + ci * p.kernel_h * p.kernel_w
                              + ky * p.kernel_w + kx;
                    sum += conv_input[in_idx] * conv_weight[w_idx];
                }
            }
        }
    }

    if (p.has_bias == 1u) {
        sum += conv_bias[co];
    }

    conv_output[idx] = sum;
}
