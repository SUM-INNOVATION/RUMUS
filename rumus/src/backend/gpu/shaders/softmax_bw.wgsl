// Softmax backward: grad_input = softmax * (grad_output - dot)
// where dot = Σ_j grad_output[j] * softmax[j] (per row).
//
// One workgroup per row.  Two phases:
//   1. Reduce → dot
//   2. Each thread: grad_input[j] = saved_out[j] * (grad_out[j] - dot)
//
// Reuses binary_layout: saved_out(read) + grad_out(read) + grad_in(rw) + uniform.

struct SoftmaxBwParams {
    num_rows: u32,
    row_size: u32,
    _pad0: u32,
    _pad1: u32,
}
// 16 bytes ✓

@group(0) @binding(0) var<storage, read>       smbw_saved:    array<scalar>; // softmax output
@group(0) @binding(1) var<storage, read>       smbw_grad_out: array<scalar>;
@group(0) @binding(2) var<storage, read_write> smbw_grad_in:  array<scalar>;
@group(0) @binding(3) var<uniform>             smbw_params:   SoftmaxBwParams;

var<workgroup> shared_dot: array<scalar, 64>;

@compute @workgroup_size(64)
fn softmax_backward_kernel(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let row = wgid.x;
    if (row >= smbw_params.num_rows) { return; }
    let D = smbw_params.row_size;
    let tid = lid.x;
    let base = row * D;

    // Phase 1: dot = Σ grad_out[j] * saved_out[j]
    var local_dot: scalar = scalar(0.0);
    var j = tid;
    while (j < D) {
        local_dot += smbw_grad_out[base + j] * smbw_saved[base + j];
        j += 64u;
    }
    shared_dot[tid] = local_dot;
    workgroupBarrier();
    var s: u32 = 32u;
    while (s > 0u) {
        if (tid < s) { shared_dot[tid] += shared_dot[tid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    let dot = shared_dot[0];
    workgroupBarrier();

    // Phase 2: grad_input[j] = saved_out[j] * (grad_out[j] - dot)
    j = tid;
    while (j < D) {
        smbw_grad_in[base + j] = smbw_saved[base + j] * (smbw_grad_out[base + j] - dot);
        j += 64u;
    }
}
