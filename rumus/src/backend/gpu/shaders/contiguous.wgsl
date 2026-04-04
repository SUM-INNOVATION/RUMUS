// Strided-to-contiguous copy kernel.
//
// Reads from a strided source buffer, writes to a dense destination.
// Supports up to 8 dimensions via shape/strides/suffix packed as vec4<u32>.

struct ContiguousParams {
    numel: u32,
    ndim: u32,
    offset: u32,
    _pad: u32,
    shape_lo:   vec4<u32>,  // shape[0..4]
    shape_hi:   vec4<u32>,  // shape[4..8]
    strides_lo: vec4<u32>,  // strides[0..4]
    strides_hi: vec4<u32>,  // strides[4..8]
    suffix_lo:  vec4<u32>,  // suffix[0..4]
    suffix_hi:  vec4<u32>,  // suffix[4..8]
}
// 16 + 16*6 = 112 bytes — multiple of 16 ✓

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

@group(0) @binding(0) var<storage, read>       cont_src: array<scalar>;
@group(0) @binding(1) var<storage, read_write> cont_dst: array<scalar>;
@group(0) @binding(2) var<uniform>             cont_params: ContiguousParams;

@compute @workgroup_size(64)
fn contiguous_copy_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_idx = gid.x;
    if (dst_idx >= cont_params.numel) { return; }

    var src_idx = cont_params.offset;
    var remainder = dst_idx;

    for (var d: u32 = 0u; d < cont_params.ndim; d++) {
        let dim_size = get_val(cont_params.suffix_lo, cont_params.suffix_hi, d);
        let stride = get_val(cont_params.strides_lo, cont_params.strides_hi, d);
        let coord = remainder / dim_size;
        remainder = remainder % dim_size;
        src_idx += coord * stride;
    }

    cont_dst[dst_idx] = cont_src[src_idx];
}
