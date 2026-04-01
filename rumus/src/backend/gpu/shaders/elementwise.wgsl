// Element-wise compute kernels.
//
// Binary ops (add, sub, mul, relu_backward) use bindings 0-3:
//   @binding(0) a      — read
//   @binding(1) b      — read
//   @binding(2) out    — read_write
//   @binding(3) params — uniform (16-byte aligned)
//
// Unary ops (relu, scale) use bindings 0-2:
//   @binding(0) input  — read
//   @binding(1) out    — read_write
//   @binding(2) params — uniform (16-byte aligned)

// --- Binary ops ---

struct Params {
    numel: u32,
    scalar: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             params: Params;

@compute @workgroup_size(64)
fn add_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.numel) { return; }
    out[i] = a[i] + b[i];
}

@compute @workgroup_size(64)
fn sub_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.numel) { return; }
    out[i] = a[i] - b[i];
}

@compute @workgroup_size(64)
fn mul_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.numel) { return; }
    out[i] = a[i] * b[i];
}

@compute @workgroup_size(64)
fn relu_backward_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.numel) { return; }
    // a = saved_input, b = out_grad
    out[i] = select(0.0, b[i], a[i] > 0.0);
}
