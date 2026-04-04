// Fused stride-aware scale kernel.
//
// Reads from a potentially non-contiguous source buffer using strides,
// multiplies by a scalar, and writes to a dense output buffer.
// Zero intermediate VRAM allocation — no contiguous() needed.
//
// Uses the same struct layout as FusedDropoutParams but repurposes
// seed/p_threshold as unused padding.  The scalar field is the scale factor.
//
// Bind group: unary_layout (input(read) + output(rw) + uniform).

struct FusedScaleParams {
    numel: u32,
    scalar: f32,
    _pad0: u32,
    _pad1: u32,
    ndim: u32,
    offset: u32,
    _pad2: u32,
    _pad3: u32,
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

@group(0) @binding(0) var<storage, read>       fs_input:  array<scalar>;
@group(0) @binding(1) var<storage, read_write> fs_output: array<scalar>;
@group(0) @binding(2) var<uniform>             fs_params: FusedScaleParams;

@compute @workgroup_size(64)
fn fused_scale_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= fs_params.numel) { return; }

    // Compute strided source index from dense output index.
    var src_idx = fs_params.offset;
    var remainder = i;
    for (var d: u32 = 0u; d < fs_params.ndim; d++) {
        let dim_size = get_val(fs_params.suffix_lo, fs_params.suffix_hi, d);
        let stride = get_val(fs_params.strides_lo, fs_params.strides_hi, d);
        let coord = remainder / dim_size;
        remainder = remainder % dim_size;
        src_idx += coord * stride;
    }

    fs_output[i] = fs_input[src_idx] * scalar(fs_params.scalar);
}
