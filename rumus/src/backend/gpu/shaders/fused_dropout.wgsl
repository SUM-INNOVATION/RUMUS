// Fused stride-aware dropout kernel.
//
// Combines strided input reading with PCG PRNG dropout in a single pass.
// Uses vec4<u32> packing for uniform array alignment.

struct FusedDropoutParams {
    numel: u32,
    seed: u32,
    p_threshold: u32,
    scale: f32,
    ndim: u32,
    offset: u32,
    _pad0: u32,
    _pad1: u32,
    shape_lo:   vec4<u32>,
    shape_hi:   vec4<u32>,
    strides_lo: vec4<u32>,
    strides_hi: vec4<u32>,
    suffix_lo:  vec4<u32>,
    suffix_hi:  vec4<u32>,
}
// 16 + 16 + 16*6 = 128 bytes ✓

fn get_val(lo: vec4<u32>, hi: vec4<u32>, idx: u32) -> u32 {
    switch idx {
        case 0u: { return lo.x; }
        case 1u: { return lo.y; }
        case 2u: { return lo.z; }
        case 3u: { return lo.w; }
        case 4u: { return hi.x; }
        case 5u: { return hi.y; }
        case 6u: { return hi.z; }
        case 7u: { return hi.w; }
        default: { return 0u; }
    }
}

@group(0) @binding(0) var<storage, read>       fd_input:  array<scalar>;
@group(0) @binding(1) var<storage, read_write> fd_output: array<scalar>;
@group(0) @binding(2) var<storage, read_write> fd_mask:   array<scalar>;
@group(0) @binding(3) var<uniform>             fd_params: FusedDropoutParams;

fn pcg_hash(input_val: u32) -> u32 {
    var state = input_val;
    state = state * 747796405u + 2891336453u;
    state = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (state >> 22u) ^ state;
}

@compute @workgroup_size(64)
fn fused_dropout_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= fd_params.numel) { return; }

    var src_idx = fd_params.offset;
    var remainder = i;
    for (var d: u32 = 0u; d < fd_params.ndim; d++) {
        let dim_size = get_val(fd_params.suffix_lo, fd_params.suffix_hi, d);
        let stride = get_val(fd_params.strides_lo, fd_params.strides_hi, d);
        let coord = remainder / dim_size;
        remainder = remainder % dim_size;
        src_idx += coord * stride;
    }

    let hash = pcg_hash(fd_params.seed ^ i);

    if (hash < fd_params.p_threshold) {
        fd_output[i] = 0.0;
        fd_mask[i] = 0.0;
    } else {
        fd_output[i] = fd_input[src_idx] * scalar(fd_params.scale);
        fd_mask[i] = scalar(fd_params.scale);
    }
}
