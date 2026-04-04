// MaxPool2d compute kernels.
//
// Forward: slide K×K window, extract max + argmax index per patch.
// Backward: scatter out_grad to the saved argmax positions.
//
// Operates on a single batch element: [channels, H, W].

struct MaxPool2dParams {
    channels: u32,
    h: u32,
    w: u32,
    k: u32,
    stride: u32,
    out_h: u32,
    out_w: u32,
    _pad: u32,
}
// 8 * 4 = 32 bytes — multiple of 16 ✓

// --- Forward ----------------------------------------------------------------
// input:   [channels, H, W]           (binding 0, read)
// output:  [channels, out_h, out_w]   (binding 1, rw)
// indices: [channels, out_h, out_w]   (binding 2, rw) — stored as f32
// Each thread handles one output element.

@group(0) @binding(0) var<storage, read>       pool_input:   array<scalar>;
@group(0) @binding(1) var<storage, read_write> pool_output:  array<scalar>;
@group(0) @binding(2) var<storage, read_write> pool_indices: array<scalar>;
@group(0) @binding(3) var<uniform>             pool_params:  MaxPool2dParams;

@compute @workgroup_size(64)
fn max_pool2d_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = pool_params;
    let total = p.channels * p.out_h * p.out_w;
    let idx = gid.x;
    if (idx >= total) { return; }

    let c   = idx / (p.out_h * p.out_w);
    let rem = idx % (p.out_h * p.out_w);
    let oh  = rem / p.out_w;
    let ow  = rem % p.out_w;

    var max_val: scalar = scalar(-3.402823e+38);
    var max_idx: u32 = 0u;

    for (var kh: u32 = 0u; kh < p.k; kh++) {
        for (var kw: u32 = 0u; kw < p.k; kw++) {
            let ih = oh * p.stride + kh;
            let iw = ow * p.stride + kw;
            let val = pool_input[c * p.h * p.w + ih * p.w + iw];
            if (val > max_val) {
                max_val = val;
                max_idx = ih * p.w + iw;
            }
        }
    }

    pool_output[idx] = max_val;
    pool_indices[idx] = scalar(max_idx);
}

// --- Backward ---------------------------------------------------------------
// out_grad:   [channels, out_h, out_w]  (binding 0, read)
// indices:    [channels, out_h, out_w]  (binding 1, read)
// grad_input: [channels, H, W]         (binding 2, rw, pre-zeroed)
// Each thread handles one output element, scatters to the argmax position.
// Safe when stride >= kernel_size (no overlapping windows).

@group(0) @binding(0) var<storage, read>       pool_bw_out_grad:   array<scalar>;
@group(0) @binding(1) var<storage, read>       pool_bw_indices:    array<scalar>;
@group(0) @binding(2) var<storage, read_write> pool_bw_grad_input: array<scalar>;
@group(0) @binding(3) var<uniform>             pool_bw_params:     MaxPool2dParams;

@compute @workgroup_size(64)
fn max_pool2d_backward_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = pool_bw_params;
    let total = p.channels * p.out_h * p.out_w;
    let idx = gid.x;
    if (idx >= total) { return; }

    let c = idx / (p.out_h * p.out_w);
    let src_idx = u32(pool_bw_indices[idx]);

    pool_bw_grad_input[c * p.h * p.w + src_idx] += pool_bw_out_grad[idx];
}
