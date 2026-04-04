// Cast kernels: convert between f32 and f16 data types.
// These kernels use concrete types (not the scalar alias) because
// they bridge the two dtype worlds.

enable f16;

struct CastParams {
    numel: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
// 16 bytes ✓

// --- F32 → F16 ---
@group(0) @binding(0) var<storage, read>       cast_f32_input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> cast_f16_output: array<f16>;
@group(0) @binding(2) var<uniform>             cast_down_params: CastParams;

@compute @workgroup_size(256)
fn cast_f32_to_f16_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= cast_down_params.numel) { return; }
    cast_f16_output[idx] = f16(cast_f32_input[idx]);
}

// --- F16 → F32 ---
@group(0) @binding(0) var<storage, read>       cast_f16_input:  array<f16>;
@group(0) @binding(1) var<storage, read_write> cast_f32_output: array<f32>;
@group(0) @binding(2) var<uniform>             cast_up_params: CastParams;

@compute @workgroup_size(256)
fn cast_f16_to_f32_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= cast_up_params.numel) { return; }
    cast_f32_output[idx] = f32(cast_f16_input[idx]);
}
