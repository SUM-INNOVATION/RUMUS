// Row-wise Softmax with Log-Sum-Exp stability.
//
// One workgroup per row.  Three phases:
//   1. Reduce → max
//   2. Each thread: exp(x - max), reduce → sum_exp
//   3. Each thread: output = exp(x - max) / sum_exp
//
// Reuses unary_layout: input(read) + output(rw) + uniform.

struct SoftmaxParams {
    num_rows: u32,
    row_size: u32,
    _pad0: u32,
    _pad1: u32,
}
// 16 bytes ✓

@group(0) @binding(0) var<storage, read>       sm_input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> sm_output: array<f32>;
@group(0) @binding(2) var<uniform>             sm_params: SoftmaxParams;

var<workgroup> shared_val: array<f32, 64>;

@compute @workgroup_size(64)
fn softmax_forward_kernel(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let row = wgid.x;
    if (row >= sm_params.num_rows) { return; }
    let D = sm_params.row_size;
    let tid = lid.x;
    let base = row * D;

    // Phase 1: max
    var local_max: f32 = -3.402823e+38;
    var j = tid;
    while (j < D) {
        local_max = max(local_max, sm_input[base + j]);
        j += 64u;
    }
    shared_val[tid] = local_max;
    workgroupBarrier();
    var s: u32 = 32u;
    while (s > 0u) {
        if (tid < s) { shared_val[tid] = max(shared_val[tid], shared_val[tid + s]); }
        workgroupBarrier();
        s = s >> 1u;
    }
    let max_val = shared_val[0];
    workgroupBarrier();

    // Phase 2: sum of exp
    var local_sum: f32 = 0.0;
    j = tid;
    while (j < D) {
        local_sum += exp(sm_input[base + j] - max_val);
        j += 64u;
    }
    shared_val[tid] = local_sum;
    workgroupBarrier();
    s = 32u;
    while (s > 0u) {
        if (tid < s) { shared_val[tid] += shared_val[tid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    let sum_exp = shared_val[0];
    workgroupBarrier();

    // Phase 3: normalize
    j = tid;
    while (j < D) {
        sm_output[base + j] = exp(sm_input[base + j] - max_val) / sum_exp;
        j += 64u;
    }
}
