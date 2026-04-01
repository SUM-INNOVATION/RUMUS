// Unary element-wise compute kernels.
//
// Bind group layout:
//   @binding(0) input  — storage, read
//   @binding(1) out    — storage, read_write
//   @binding(2) params — uniform (16-byte aligned)

struct Params {
    numel: u32,
    scalar: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read>       input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;

@compute @workgroup_size(64)
fn relu_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.numel) { return; }
    out[i] = max(0.0, input[i]);
}

@compute @workgroup_size(64)
fn scale_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.numel) { return; }
    out[i] = params.scalar * input[i];
}
