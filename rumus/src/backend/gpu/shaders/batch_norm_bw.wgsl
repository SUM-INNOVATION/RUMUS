// BatchNorm2d backward: per-channel grad_input.
//
// One workgroup per channel. Two reductions → c1, c2 → element-wise grad_input.

struct BatchNormBwParams {
    batch: u32,
    channels: u32,
    height: u32,
    width: u32,
}
// 16 bytes ✓

@group(0) @binding(0) var<storage, read>       bnbw_grad_out: array<scalar>;
@group(0) @binding(1) var<storage, read>       bnbw_input:    array<scalar>;
@group(0) @binding(2) var<storage, read>       bnbw_weight:   array<scalar>;
@group(0) @binding(3) var<storage, read>       bnbw_save:     array<scalar>;
@group(0) @binding(4) var<storage, read_write> bnbw_grad_in:  array<scalar>;
@group(0) @binding(5) var<uniform>             bnbw_params:   BatchNormBwParams;

var<workgroup> shared_c1: array<scalar, 64>;
var<workgroup> shared_c2: array<scalar, 64>;

@compute @workgroup_size(64)
fn batch_norm_backward_kernel(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let c = wgid.x;
    if (c >= bnbw_params.channels) { return; }
    let tid = lid.x;
    let spatial = bnbw_params.height * bnbw_params.width;
    let n = bnbw_params.batch * spatial;
    let mean = bnbw_save[c * 2u];
    let invstd = bnbw_save[c * 2u + 1u];
    let gamma = bnbw_weight[c];

    // Reductions: c1 = (1/N) Σ grad_norm, c2 = (1/N) Σ grad_norm * x_hat
    var lc1: scalar = scalar(0.0);
    var lc2: scalar = scalar(0.0);
    var idx = tid;
    while (idx < n) {
        let b = idx / spatial;
        let hw = idx % spatial;
        let flat = b * bnbw_params.channels * spatial + c * spatial + hw;
        let grad_norm = bnbw_grad_out[flat] * gamma;
        let x_hat = (bnbw_input[flat] - mean) * invstd;
        lc1 += grad_norm;
        lc2 += grad_norm * x_hat;
        idx += 64u;
    }
    shared_c1[tid] = lc1;
    shared_c2[tid] = lc2;
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
    let c1 = shared_c1[0] / scalar(n);
    let c2 = shared_c2[0] / scalar(n);
    workgroupBarrier();

    // Element-wise grad_input
    idx = tid;
    while (idx < n) {
        let b = idx / spatial;
        let hw = idx % spatial;
        let flat = b * bnbw_params.channels * spatial + c * spatial + hw;
        let grad_norm = bnbw_grad_out[flat] * gamma;
        let x_hat = (bnbw_input[flat] - mean) * invstd;
        bnbw_grad_in[flat] = invstd * (grad_norm - c1 - x_hat * c2);
        idx += 64u;
    }
}
