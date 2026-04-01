// Optimizer compute kernels.
//
// sgd_step: param -= lr * (momentum * vel + grad)
// adam_step: fused m/v update + bias-corrected weight update

// ---------------------------------------------------------------------------
// SGD with momentum
// ---------------------------------------------------------------------------

struct SgdHyperparams {
    lr: f32,
    momentum: f32,
    numel: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       grad:  array<f32>;
@group(0) @binding(1) var<storage, read_write> vel:   array<f32>;
@group(0) @binding(2) var<storage, read_write> param: array<f32>;
@group(0) @binding(3) var<uniform>             sgd_hp: SgdHyperparams;

@compute @workgroup_size(64)
fn sgd_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sgd_hp.numel) { return; }

    vel[i] = sgd_hp.momentum * vel[i] + grad[i];
    param[i] -= sgd_hp.lr * vel[i];
}

// ---------------------------------------------------------------------------
// Adam
// ---------------------------------------------------------------------------

struct AdamHyperparams {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    bc1: f32,       // 1 - beta1^t  (pre-computed on CPU)
    bc2: f32,       // 1 - beta2^t  (pre-computed on CPU)
    numel: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       adam_grad:  array<f32>;
@group(0) @binding(1) var<storage, read_write> m:         array<f32>;
@group(0) @binding(2) var<storage, read_write> v:         array<f32>;
@group(0) @binding(3) var<storage, read_write> adam_param: array<f32>;
@group(0) @binding(4) var<uniform>             adam_hp:    AdamHyperparams;

@compute @workgroup_size(64)
fn adam_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= adam_hp.numel) { return; }

    let g = adam_grad[i];

    // Update moments.
    m[i] = adam_hp.beta1 * m[i] + (1.0 - adam_hp.beta1) * g;
    v[i] = adam_hp.beta2 * v[i] + (1.0 - adam_hp.beta2) * g * g;

    // Bias-corrected estimates.
    let m_hat = m[i] / adam_hp.bc1;
    let v_hat = v[i] / adam_hp.bc2;

    // Apply update.
    adam_param[i] -= adam_hp.lr * m_hat / (sqrt(v_hat) + adam_hp.epsilon);
}
