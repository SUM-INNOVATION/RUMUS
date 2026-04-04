// BatchNorm2d forward: per-channel reduction over B*H*W.
//
// One workgroup per channel.  Three phases:
//   1. Reduce → mean[c]
//   2. Reduce → var[c]
//   3. Normalize + affine + update running stats
//
// During eval (is_training==0), uses running_mean/running_var instead.

struct BatchNorm2dParams {
    batch: u32,
    channels: u32,
    height: u32,
    width: u32,
    epsilon: f32,
    momentum: f32,
    is_training: u32,
    _pad: u32,
}
// 32 bytes ✓

@group(0) @binding(0) var<storage, read>       bn_input:        array<scalar>;
@group(0) @binding(1) var<storage, read>       bn_weight:       array<scalar>; // γ [C]
@group(0) @binding(2) var<storage, read>       bn_bias:         array<scalar>; // β [C]
@group(0) @binding(3) var<storage, read_write> bn_running_mean: array<scalar>; // [C]
@group(0) @binding(4) var<storage, read_write> bn_running_var:  array<scalar>; // [C]
@group(0) @binding(5) var<storage, read_write> bn_output:       array<scalar>;
@group(0) @binding(6) var<storage, read_write> bn_save:         array<scalar>; // [C, 2]
@group(0) @binding(7) var<uniform>             bn_params:       BatchNorm2dParams;

var<workgroup> shared_val: array<scalar, 64>;

@compute @workgroup_size(64)
fn batch_norm_forward_kernel(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let c = wgid.x;
    if (c >= bn_params.channels) { return; }
    let tid = lid.x;
    let spatial = bn_params.height * bn_params.width;
    let n = bn_params.batch * spatial;

    var use_mean: scalar;
    var use_invstd: scalar;

    if (bn_params.is_training == 1u) {
        // ---- Phase 1: Mean ----
        var local_sum: scalar = scalar(0.0);
        var idx = tid;
        while (idx < n) {
            let b = idx / spatial;
            let hw = idx % spatial;
            let flat = b * bn_params.channels * spatial + c * spatial + hw;
            local_sum += bn_input[flat];
            idx += 64u;
        }
        shared_val[tid] = local_sum;
        workgroupBarrier();
        var s: u32 = 32u;
        while (s > 0u) {
            if (tid < s) { shared_val[tid] += shared_val[tid + s]; }
            workgroupBarrier();
            s = s >> 1u;
        }
        let mean = shared_val[0] / scalar(n);
        workgroupBarrier();

        // ---- Phase 2: Variance ----
        var local_var: scalar = scalar(0.0);
        idx = tid;
        while (idx < n) {
            let b = idx / spatial;
            let hw = idx % spatial;
            let flat = b * bn_params.channels * spatial + c * spatial + hw;
            let diff = bn_input[flat] - mean;
            local_var += diff * diff;
            idx += 64u;
        }
        shared_val[tid] = local_var;
        workgroupBarrier();
        s = 32u;
        while (s > 0u) {
            if (tid < s) { shared_val[tid] += shared_val[tid + s]; }
            workgroupBarrier();
            s = s >> 1u;
        }
        let variance = shared_val[0] / scalar(n);
        let invstd = scalar(1.0) / sqrt(variance + scalar(bn_params.epsilon));
        workgroupBarrier();

        use_mean = mean;
        use_invstd = invstd;

        // Save + update running stats (thread 0 only).
        if (tid == 0u) {
            bn_save[c * 2u] = mean;
            bn_save[c * 2u + 1u] = invstd;
            let m = scalar(bn_params.momentum);
            bn_running_mean[c] = (scalar(1.0) - m) * bn_running_mean[c] + m * mean;
            bn_running_var[c]  = (scalar(1.0) - m) * bn_running_var[c]  + m * variance;
        }
    } else {
        // Eval mode: use running stats.
        use_mean = bn_running_mean[c];
        use_invstd = scalar(1.0) / sqrt(bn_running_var[c] + scalar(bn_params.epsilon));
    }
    workgroupBarrier();

    // ---- Phase 3: Normalize + Affine ----
    let gamma = bn_weight[c];
    let beta = bn_bias[c];
    var idx2 = tid;
    while (idx2 < n) {
        let b = idx2 / spatial;
        let hw = idx2 % spatial;
        let flat = b * bn_params.channels * spatial + c * spatial + hw;
        let x_hat = (bn_input[flat] - use_mean) * use_invstd;
        bn_output[flat] = gamma * x_hat + beta;
        idx2 += 64u;
    }
}
