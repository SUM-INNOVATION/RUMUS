// Conv2d backward: grad_input (transposed convolution).
//
// 1 thread = 1 input pixel.  Accumulates contributions from all output
// positions that touch this input pixel through the convolution.
//
// Bindings (CustomOp: 2 inputs + 1 output + 1 uniform):
//   0: grad_output  [B, C_out, H_out, W_out]  (scalar, read)
//   1: weight       [C_out, C_in, K_h, K_w]   (scalar, read)
//   2: grad_input   [B, C_in, H_in, W_in]     (scalar, rw)
//   3: params       (uniform)

struct Conv2dBwDataParams {
    total_input: u32,
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
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       bwd_grad_out: array<scalar>;
@group(0) @binding(1) var<storage, read>       bwd_weight:   array<scalar>;
@group(0) @binding(2) var<storage, read_write> bwd_grad_in:  array<scalar>;
@group(0) @binding(3) var<uniform>             bwd_params:   Conv2dBwDataParams;

@compute @workgroup_size(256)
fn conv2d_backward_data_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= bwd_params.total_input) { return; }

    let p = bwd_params;
    let iw = idx % p.w_in;
    let tmp = idx / p.w_in;
    let ih = tmp % p.h_in;
    let tmp2 = tmp / p.h_in;
    let ci = tmp2 % p.c_in;
    let b = tmp2 / p.c_in;

    var grad_sum: scalar = scalar(0.0);

    for (var co: u32 = 0u; co < p.c_out; co++) {
        for (var ky: u32 = 0u; ky < p.kernel_h; ky++) {
            for (var kx: u32 = 0u; kx < p.kernel_w; kx++) {
                let oh_num = i32(ih) + i32(p.pad_h) - i32(ky * p.dilation_h);
                let ow_num = i32(iw) + i32(p.pad_w) - i32(kx * p.dilation_w);

                if (oh_num < 0 || ow_num < 0) { continue; }
                if (oh_num % i32(p.stride_h) != 0 || ow_num % i32(p.stride_w) != 0) { continue; }

                let oh = u32(oh_num / i32(p.stride_h));
                let ow = u32(ow_num / i32(p.stride_w));
                if (oh >= p.h_out || ow >= p.w_out) { continue; }

                let go_idx = b * p.c_out * p.h_out * p.w_out
                           + co * p.h_out * p.w_out
                           + oh * p.w_out + ow;
                let w_idx = co * p.c_in * p.kernel_h * p.kernel_w
                          + ci * p.kernel_h * p.kernel_w
                          + ky * p.kernel_w + kx;
                grad_sum += bwd_grad_out[go_idx] * bwd_weight[w_idx];
            }
        }
    }

    bwd_grad_in[idx] = grad_sum;
}
