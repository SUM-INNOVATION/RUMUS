// N-dimensional broadcasted binary ops + reduce_sum.
//
// Each thread handles one element of the output.  It decomposes the
// linear output index into a multi-index via suffix products, then
// computes separate source indices for a and b using their broadcast
// strides (stride 0 = broadcasted dimension).

struct BroadcastBinaryParams {
    numel: u32,
    ndim: u32,
    _pad0: u32,
    _pad1: u32,
    suffix_lo:    vec4<u32>,
    suffix_hi:    vec4<u32>,
    a_strides_lo: vec4<u32>,
    a_strides_hi: vec4<u32>,
    b_strides_lo: vec4<u32>,
    b_strides_hi: vec4<u32>,
}
// 16 + 16*5 = 96 bytes — multiple of 16 ✓

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

@group(0) @binding(0) var<storage, read>       bc_a:      array<scalar>;
@group(0) @binding(1) var<storage, read>       bc_b:      array<scalar>;
@group(0) @binding(2) var<storage, read_write> bc_out:    array<scalar>;
@group(0) @binding(3) var<uniform>             bc_params: BroadcastBinaryParams;

fn compute_indices(i: u32) -> vec2<u32> {
    var a_idx: u32 = 0u;
    var b_idx: u32 = 0u;
    var remainder = i;
    for (var d: u32 = 0u; d < bc_params.ndim; d++) {
        let s = get_val(bc_params.suffix_lo, bc_params.suffix_hi, d);
        let coord = remainder / s;
        remainder = remainder % s;
        a_idx += coord * get_val(bc_params.a_strides_lo, bc_params.a_strides_hi, d);
        b_idx += coord * get_val(bc_params.b_strides_lo, bc_params.b_strides_hi, d);
    }
    return vec2<u32>(a_idx, b_idx);
}

@compute @workgroup_size(64)
fn broadcast_add_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= bc_params.numel) { return; }
    let idx = compute_indices(i);
    bc_out[i] = bc_a[idx.x] + bc_b[idx.y];
}

@compute @workgroup_size(64)
fn broadcast_sub_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= bc_params.numel) { return; }
    let idx = compute_indices(i);
    bc_out[i] = bc_a[idx.x] - bc_b[idx.y];
}

@compute @workgroup_size(64)
fn broadcast_mul_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= bc_params.numel) { return; }
    let idx = compute_indices(i);
    bc_out[i] = bc_a[idx.x] * bc_b[idx.y];
}

// === reduce_sum =============================================================
// Reduces a tensor along specified dimensions (broadcast backward).
// Each thread computes one element of the reduced output by summing
// over the reduced dimensions.

struct ReduceSumParams {
    out_numel: u32,   // elements in reduced output
    ndim: u32,
    _pad0: u32,
    _pad1: u32,
    in_shape_lo:  vec4<u32>,    // full input shape
    in_shape_hi:  vec4<u32>,
    in_suffix_lo: vec4<u32>,    // input suffix products
    in_suffix_hi: vec4<u32>,
    out_strides_lo: vec4<u32>,  // output strides (0 for reduced dims)
    out_strides_hi: vec4<u32>,
    reduce_extents_lo: vec4<u32>,  // size of each dim (1 if not reduced)
    reduce_extents_hi: vec4<u32>,
}
// 16 + 16*7 = 128 bytes ✓

@group(0) @binding(0) var<storage, read>       rs_input:  array<scalar>;
@group(0) @binding(1) var<storage, read_write> rs_output: array<scalar>;
@group(0) @binding(2) var<uniform>             rs_params: ReduceSumParams;

@compute @workgroup_size(64)
fn reduce_sum_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_idx = gid.x;
    if (out_idx >= rs_params.out_numel) { return; }

    // Decompose out_idx into multi-index of the output.
    // For reduced dims, the coordinate is 0; for kept dims, use the coordinate.
    // Then iterate over all combinations of the reduced dims.

    // First, compute the base input index from the output index.
    var base_idx: u32 = 0u;
    var remainder = out_idx;

    // We need output suffix products.  Compute them from out_strides.
    // Actually, out_strides already encode the mapping: for kept dims,
    // out_strides[d] is the output stride; for reduced dims, it's 0.
    // We need to decompose out_idx using the OUTPUT shape (not input shape).
    // For simplicity, the output suffix products are just the out_strides
    // since the output is contiguous.

    // Simpler approach: iterate all input elements, check if they map to
    // this output index.  O(input_numel / out_numel) per thread.
    // This is correct and simple for the MVP.

    // Total input elements that map to one output element.
    var total_reduce: u32 = 1u;
    for (var d: u32 = 0u; d < rs_params.ndim; d++) {
        total_reduce *= get_val(rs_params.reduce_extents_lo, rs_params.reduce_extents_hi, d);
    }

    // Decompose out_idx into output coordinates.
    // Output is contiguous with its own shape.
    // We need output suffix products — derive from out_strides.
    // Actually, let's just iterate over the reduced dims.

    // For each reduction combo, compute the full input index.
    var sum_val: scalar = scalar(0.0);
    for (var r: u32 = 0u; r < total_reduce; r++) {
        // Decompose out_idx into kept-dim coords, r into reduced-dim coords.
        var in_idx: u32 = 0u;
        var out_rem = out_idx;
        var red_rem = r;

        for (var d: i32 = i32(rs_params.ndim) - 1; d >= 0; d--) {
            let du = u32(d);
            let dim_size = get_val(rs_params.in_shape_lo, rs_params.in_shape_hi, du);
            let reduce_ext = get_val(rs_params.reduce_extents_lo, rs_params.reduce_extents_hi, du);
            let in_suf = get_val(rs_params.in_suffix_lo, rs_params.in_suffix_hi, du);

            var coord: u32;
            if (reduce_ext > 1u) {
                // This is a reduced dimension.
                coord = red_rem % dim_size;
                red_rem = red_rem / dim_size;
            } else {
                // This is a kept dimension.
                let out_stride = get_val(rs_params.out_strides_lo, rs_params.out_strides_hi, du);
                if (out_stride > 0u) {
                    coord = out_rem / out_stride;
                    out_rem = out_rem % out_stride;
                } else {
                    coord = 0u;
                }
            }
            in_idx += coord * in_suf;
        }
        sum_val += rs_input[in_idx];
    }
    rs_output[out_idx] = sum_val;
}
