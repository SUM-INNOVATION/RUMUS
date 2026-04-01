// AdamW optimizer kernel (decoupled weight decay).
//
// Reads grad, reads+writes m, v, param in a single dispatch.
// Weight decay applied directly to weights (decoupled from gradient),
// before the adaptive gradient step.

struct AdamWHyperparams {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    bc1: f32,           // 1 - beta1^t
    bc2: f32,           // 1 - beta2^t
    weight_decay: f32,
    numel: u32,
}
// 8 * 4 = 32 bytes ✓

@group(0) @binding(0) var<storage, read>       adamw_grad:  array<f32>;
@group(0) @binding(1) var<storage, read_write> adamw_m:     array<f32>;
@group(0) @binding(2) var<storage, read_write> adamw_v:     array<f32>;
@group(0) @binding(3) var<storage, read_write> adamw_param: array<f32>;
@group(0) @binding(4) var<uniform>             adamw_hp:    AdamWHyperparams;

@compute @workgroup_size(64)
fn adamw_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= adamw_hp.numel) { return; }

    let g = adamw_grad[i];

    // Update moments.
    adamw_m[i] = adamw_hp.beta1 * adamw_m[i] + (1.0 - adamw_hp.beta1) * g;
    adamw_v[i] = adamw_hp.beta2 * adamw_v[i] + (1.0 - adamw_hp.beta2) * g * g;

    // Bias-corrected estimates.
    let m_hat = adamw_m[i] / adamw_hp.bc1;
    let v_hat = adamw_v[i] / adamw_hp.bc2;

    // Decoupled weight decay (applied before gradient step).
    adamw_param[i] -= adamw_hp.lr * adamw_hp.weight_decay * adamw_param[i];

    // Adaptive gradient step.
    adamw_param[i] -= adamw_hp.lr * m_hat / (sqrt(v_hat) + adamw_hp.epsilon);
}
