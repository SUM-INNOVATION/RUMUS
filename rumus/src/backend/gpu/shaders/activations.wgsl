// Advanced activation kernels (forward + backward).
//
// Forward ops use unary_layout: input(read) + out(rw) + uniform.
// Backward ops use binary_layout: saved(read) + out_grad(read) + dst(rw) + uniform.

struct Params {
    numel: u32,
    scalar: f32,   // used by leaky_relu (alpha)
    _pad0: u32,
    _pad1: u32,
}

// === Forward (unary_layout) ================================================

@group(0) @binding(0) var<storage, read>       fwd_input: array<scalar>;
@group(0) @binding(1) var<storage, read_write> fwd_out:   array<scalar>;
@group(0) @binding(2) var<uniform>             fwd_params: Params;

@compute @workgroup_size(64)
fn sigmoid_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= fwd_params.numel) { return; }
    fwd_out[i] = 1.0 / (1.0 + exp(-fwd_input[i]));
}

@compute @workgroup_size(64)
fn tanh_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= fwd_params.numel) { return; }
    fwd_out[i] = tanh(fwd_input[i]);
}

@compute @workgroup_size(64)
fn gelu_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= fwd_params.numel) { return; }
    let x = fwd_input[i];
    let inner = 0.7978845608 * (x + 0.044715 * x * x * x); // sqrt(2/pi) ≈ 0.7978845608
    fwd_out[i] = 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(64)
fn leaky_relu_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= fwd_params.numel) { return; }
    let x = fwd_input[i];
    fwd_out[i] = select(scalar(fwd_params.scalar) * x, x, x > 0.0);
}

// === Backward (binary_layout) ==============================================
// binding 0 = saved tensor (read), binding 1 = out_grad (read),
// binding 2 = grad_input (rw), binding 3 = uniform

@group(0) @binding(0) var<storage, read>       bw_saved:    array<scalar>;
@group(0) @binding(1) var<storage, read>       bw_out_grad: array<scalar>;
@group(0) @binding(2) var<storage, read_write> bw_dst:      array<scalar>;
@group(0) @binding(3) var<uniform>             bw_params:   Params;

// Sigmoid backward: grad = out_grad * saved_out * (1 - saved_out)
@compute @workgroup_size(64)
fn sigmoid_backward_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= bw_params.numel) { return; }
    let s = bw_saved[i];
    bw_dst[i] = bw_out_grad[i] * s * (1.0 - s);
}

// Tanh backward: grad = out_grad * (1 - saved_out^2)
@compute @workgroup_size(64)
fn tanh_backward_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= bw_params.numel) { return; }
    let t = bw_saved[i];
    bw_dst[i] = bw_out_grad[i] * (1.0 - t * t);
}

// GELU backward (tanh approx): grad = out_grad * gelu'(saved_input)
@compute @workgroup_size(64)
fn gelu_backward_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= bw_params.numel) { return; }
    let x = bw_saved[i];
    let c = 0.7978845608; // sqrt(2/pi)
    let inner = c * (x + 0.044715 * x * x * x);
    let t = tanh(inner);
    let sech2 = 1.0 - t * t;
    let d_inner = c * (1.0 + 3.0 * 0.044715 * x * x);
    let gelu_prime = 0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner;
    bw_dst[i] = bw_out_grad[i] * gelu_prime;
}

// LeakyReLU backward: grad = out_grad * (x > 0 ? 1 : alpha)
@compute @workgroup_size(64)
fn leaky_relu_backward_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= bw_params.numel) { return; }
    let x = bw_saved[i];
    bw_dst[i] = bw_out_grad[i] * select(scalar(bw_params.scalar), scalar(1.0), x > 0.0);
}
