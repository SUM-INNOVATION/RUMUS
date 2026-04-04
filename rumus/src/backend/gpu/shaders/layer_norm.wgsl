// LayerNorm forward: fused mean + variance + normalize + affine.
//
// One workgroup per normalization instance (B*S workgroups).
// Three phases separated by workgroupBarrier():
//   1. Reduce → mean
//   2. Reduce → variance
//   3. Normalize + affine: y = γ * (x - mean) / sqrt(var + ε) + β
//
// Saves mean + invstd per instance for backward.

struct LayerNormParams {
    num_instances: u32,
    norm_size: u32,
    epsilon: f32,
    _pad: u32,
}
// 16 bytes ✓

@group(0) @binding(0) var<storage, read>       ln_input:  array<scalar>;
@group(0) @binding(1) var<storage, read>       ln_weight: array<scalar>; // γ [D]
@group(0) @binding(2) var<storage, read>       ln_bias:   array<scalar>; // β [D]
@group(0) @binding(3) var<storage, read_write> ln_output: array<scalar>;
@group(0) @binding(4) var<storage, read_write> ln_save:   array<scalar>; // [num_instances, 2]
@group(0) @binding(5) var<uniform>             ln_params: LayerNormParams;

var<workgroup> shared_val: array<scalar, 64>;

@compute @workgroup_size(64)
fn layer_norm_forward_kernel(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let inst = wgid.x;
    if (inst >= ln_params.num_instances) { return; }
    let D = ln_params.norm_size;
    let tid = lid.x;
    let base = inst * D;

    // ---- Phase 1: Mean ----
    var local_sum: scalar = scalar(0.0);
    var j = tid;
    while (j < D) {
        local_sum += ln_input[base + j];
        j += 64u;
    }
    shared_val[tid] = local_sum;
    workgroupBarrier();

    var s: u32 = 32u;
    while (s > 0u) {
        if (tid < s) { shared_val[tid] += shared_val[tid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    let mean = shared_val[0] / scalar(D);
    workgroupBarrier();

    // ---- Phase 2: Variance ----
    var local_var: scalar = scalar(0.0);
    j = tid;
    while (j < D) {
        let diff = ln_input[base + j] - mean;
        local_var += diff * diff;
        j += 64u;
    }
    shared_val[tid] = local_var;
    workgroupBarrier();

    s = 32u;
    while (s > 0u) {
        if (tid < s) { shared_val[tid] += shared_val[tid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    let variance = shared_val[0] / scalar(D);
    let invstd = scalar(1.0) / sqrt(variance + scalar(ln_params.epsilon));
    workgroupBarrier();

    // Save mean + invstd for backward.
    if (tid == 0u) {
        ln_save[inst * 2u] = mean;
        ln_save[inst * 2u + 1u] = invstd;
    }

    // ---- Phase 3: Normalize + Affine ----
    j = tid;
    while (j < D) {
        let x_hat = (ln_input[base + j] - mean) * invstd;
        ln_output[base + j] = ln_weight[j] * x_hat + ln_bias[j];
        j += 64u;
    }
}
