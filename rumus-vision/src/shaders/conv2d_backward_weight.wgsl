// Conv2d backward: grad_weight.
//
// 1 thread = 1 weight element (co, ci, ky, kx).
// Sums grad_output * input over all batch elements and spatial positions.
//
// Bindings (CustomOp: 2 inputs + 1 output + 1 uniform):
//   0: grad_output  [B, C_out, H_out, W_out]  (scalar, read)
//   1: input        [B, C_in, H_in, W_in]     (scalar, read)
//   2: grad_weight  [C_out, C_in, K_h, K_w]   (scalar, rw)
//   3: params       (uniform)

struct Conv2dBwWeightParams {
    total_weights: u32,
    batch: u32,
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
}

@group(0) @binding(0) var<storage, read>       bww_grad_out: array<scalar>;
@group(0) @binding(1) var<storage, read>       bww_input:    array<scalar>;
@group(0) @binding(2) var<storage, read_write> bww_grad_w:   array<scalar>;
@group(0) @binding(3) var<uniform>             bww_params:   Conv2dBwWeightParams;

@compute @workgroup_size(256)
fn conv2d_backward_weight_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= bww_params.total_weights) { return; }

    let p = bww_params;
    let kx = idx % p.kernel_w;
    let tmp = idx / p.kernel_w;
    let ky = tmp % p.kernel_h;
    let tmp2 = tmp / p.kernel_h;
    let ci = tmp2 % p.c_in;
    let co = tmp2 / p.c_in;

    var grad_sum: scalar = scalar(0.0);

    for (var b: u32 = 0u; b < p.batch; b++) {
        for (var oh: u32 = 0u; oh < p.h_out; oh++) {
            for (var ow: u32 = 0u; ow < p.w_out; ow++) {
                let ih_s = i32(oh * p.stride_h + ky * p.dilation_h) - i32(p.pad_h);
                let iw_s = i32(ow * p.stride_w + kx * p.dilation_w) - i32(p.pad_w);

                if (ih_s >= 0 && ih_s < i32(p.h_in) && iw_s >= 0 && iw_s < i32(p.w_in)) {
                    let go_idx = b * p.c_out * p.h_out * p.w_out
                               + co * p.h_out * p.w_out
                               + oh * p.w_out + ow;
                    let in_idx = b * p.c_in * p.h_in * p.w_in
                               + ci * p.h_in * p.w_in
                               + u32(ih_s) * p.w_in + u32(iw_s);
                    grad_sum += bww_grad_out[go_idx] * bww_input[in_idx];
                }
            }
        }
    }

    bww_grad_w[idx] = grad_sum;
}
