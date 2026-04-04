// LayerNorm backward: grad_input (per-instance workgroup reduction).
//
// One workgroup per instance. Computes:
//   grad_norm = grad_output * γ
//   c1 = (1/D) * Σ grad_norm[j]
//   c2 = (1/D) * Σ grad_norm[j] * x_hat[j]
//   grad_input[j] = invstd * (grad_norm[j] - c1 - x_hat[j] * c2)
//
// grad_weight and grad_bias are computed by a separate reduce pass.

struct LayerNormBwParams {
    num_instances: u32,
    norm_size: u32,
    _pad0: u32,
    _pad1: u32,
}
// 16 bytes ✓

@group(0) @binding(0) var<storage, read>       lnbw_grad_out:  array<scalar>; // [N, D]
@group(0) @binding(1) var<storage, read>       lnbw_input:     array<scalar>; // [N, D]
@group(0) @binding(2) var<storage, read>       lnbw_weight:    array<scalar>; // [D]
@group(0) @binding(3) var<storage, read>       lnbw_save:      array<scalar>; // [N, 2]
@group(0) @binding(4) var<storage, read_write> lnbw_grad_in:   array<scalar>; // [N, D]
@group(0) @binding(5) var<uniform>             lnbw_params:    LayerNormBwParams;

var<workgroup> shared_c1: array<scalar, 64>;
var<workgroup> shared_c2: array<scalar, 64>;

@compute @workgroup_size(64)
fn layer_norm_backward_kernel(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let inst = wgid.x;
    if (inst >= lnbw_params.num_instances) { return; }
    let D = lnbw_params.norm_size;
    let tid = lid.x;
    let base = inst * D;
    let mean = lnbw_save[inst * 2u];
    let invstd = lnbw_save[inst * 2u + 1u];

    // ---- Reduction 1: c1 = (1/D) * Σ grad_norm[j] ----
    // ---- Reduction 2: c2 = (1/D) * Σ grad_norm[j] * x_hat[j] ----
    var local_c1: scalar = scalar(0.0);
    var local_c2: scalar = scalar(0.0);
    var j = tid;
    while (j < D) {
        let grad_norm_j = lnbw_grad_out[base + j] * lnbw_weight[j];
        let x_hat_j = (lnbw_input[base + j] - mean) * invstd;
        local_c1 += grad_norm_j;
        local_c2 += grad_norm_j * x_hat_j;
        j += 64u;
    }
    shared_c1[tid] = local_c1;
    shared_c2[tid] = local_c2;
    workgroupBarrier();

    var s: u32 = 32u;
    while (s > 0u) {
        if (tid < s) {
            shared_c1[tid] += shared_c1[tid + s];
            shared_c2[tid] += shared_c2[tid + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    let c1 = shared_c1[0] / scalar(D);
    let c2 = shared_c2[0] / scalar(D);
    workgroupBarrier();

    // ---- Element-wise: grad_input ----
    j = tid;
    while (j < D) {
        let grad_norm_j = lnbw_grad_out[base + j] * lnbw_weight[j];
        let x_hat_j = (lnbw_input[base + j] - mean) * invstd;
        lnbw_grad_in[base + j] = invstd * (grad_norm_j - c1 - x_hat_j * c2);
        j += 64u;
    }
}
