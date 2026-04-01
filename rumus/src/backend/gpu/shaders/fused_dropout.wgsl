// Fused stride-aware dropout kernel.
//
// Combines strided input reading (from contiguous_copy) with PCG PRNG
// dropout (from dropout) in a single pass.  Eliminates the intermediate
// contiguous buffer allocation — reads directly from a potentially
// non-contiguous source and writes dense output + mask.
//
// Bind group: input(read) + output(rw) + mask(rw) + uniform.
// Reuses pool_layout.

struct FusedDropoutParams {
    // Dropout parameters
    numel: u32,
    seed: u32,
    p_threshold: u32,
    scale: f32,
    // Stride parameters (from ContiguousParams)
    ndim: u32,
    offset: u32,
    _pad0: u32,
    _pad1: u32,
    shape:   array<u32, 8>,
    strides: array<u32, 8>,
    suffix:  array<u32, 8>,
}
// 16 + 16 + 32 + 32 + 32 = 128 bytes — multiple of 16 ✓

@group(0) @binding(0) var<storage, read>       fd_input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> fd_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> fd_mask:   array<f32>;
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

    // Compute strided source index from dense output index.
    var src_idx = fd_params.offset;
    var remainder = i;
    for (var d: u32 = 0u; d < fd_params.ndim; d++) {
        let dim_size = fd_params.suffix[d];
        let coord = remainder / dim_size;
        remainder = remainder % dim_size;
        src_idx += coord * fd_params.strides[d];
    }

    // PCG PRNG dropout.
    let hash = pcg_hash(fd_params.seed ^ i);

    if (hash < fd_params.p_threshold) {
        fd_output[i] = 0.0;
        fd_mask[i] = 0.0;
    } else {
        fd_output[i] = fd_input[src_idx] * fd_params.scale;
        fd_mask[i] = fd_params.scale;
    }
}
