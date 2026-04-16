// MaxPool2d forward: max value + LOCAL window argmax index.
//
// Output is CONCATENATED: first half = max values, second half = local indices.
// Total output size = 2 * B * C * H_out * W_out.
//
// The local index is `ky * kernel_w + kx`, bounded by K*K (≤ 49 for 7×7).
// This fits losslessly in f16 (exact integers up to 2048), unlike global
// flat indices which easily exceed that limit.
//
// Bindings (CustomOp: 1 input + 1 output + 1 uniform):
//   0: input   [B, C, H_in, W_in]             (scalar, read)
//   1: output  [2 * B * C * H_out * W_out]    (scalar, rw)
//   2: params  (uniform)

struct MaxPool2dParams {
    num_output_pixels: u32,  // B * C * H_out * W_out
    channels: u32,
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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}
// 64 bytes ✓

@group(0) @binding(0) var<storage, read>       mp_input:  array<scalar>;
@group(0) @binding(1) var<storage, read_write> mp_output: array<scalar>;
@group(0) @binding(2) var<uniform>             mp_params: MaxPool2dParams;

@compute @workgroup_size(256)
fn maxpool2d_forward_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= mp_params.num_output_pixels) { return; }

    let p = mp_params;
    let ow = idx % p.w_out;
    let tmp = idx / p.w_out;
    let oh = tmp % p.h_out;
    let tmp2 = tmp / p.h_out;
    let c = tmp2 % p.channels;
    let b = tmp2 / p.channels;

    let in_base = b * p.channels * p.h_in * p.w_in + c * p.h_in * p.w_in;

    // Initialize from the FIRST valid pixel — no bitcast, F16-safe.
    var max_val: scalar = scalar(0.0);
    var max_local_idx: u32 = 0u;
    var initialized: bool = false;

    for (var ky: u32 = 0u; ky < p.kernel_h; ky++) {
        for (var kx: u32 = 0u; kx < p.kernel_w; kx++) {
            let ih_s = i32(oh * p.stride_h + ky) - i32(p.pad_h);
            let iw_s = i32(ow * p.stride_w + kx) - i32(p.pad_w);

            if (ih_s >= 0 && ih_s < i32(p.h_in) && iw_s >= 0 && iw_s < i32(p.w_in)) {
                let in_idx = in_base + u32(ih_s) * p.w_in + u32(iw_s);
                let val = mp_input[in_idx];

                if (!initialized) {
                    max_val = val;
                    max_local_idx = ky * p.kernel_w + kx;
                    initialized = true;
                } else if (val > max_val) {
                    max_val = val;
                    max_local_idx = ky * p.kernel_w + kx;
                }
            }
        }
    }

    // First half: max values.
    mp_output[idx] = max_val;
    // Second half: LOCAL window index (≤ K*K, lossless in f16).
    mp_output[p.num_output_pixels + idx] = scalar(max_local_idx);
}
