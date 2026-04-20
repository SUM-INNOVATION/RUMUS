// Fused INT4 dequantize-matmul: Y = A × dequant(W_int4)
//
// A: [M, K] activations in scalar (f16/f32).
// W: [K, N] weights packed as 8 × INT4 per u32, K-major.
// scales/zero_points: [num_groups, N] f32, grouped along K (group_size elements per group).
//
// Dequantization: w_f16 = (u4_val - zero_point) × scale
//
// K is padded to a multiple of group_size (which is a multiple of 8).
// Packed word index: (k / 8) * N + col.
// Group index: k / group_size.  Metadata index: group * N + col.
//
// Bindings (CustomOp: 4 inputs + 1 output + 1 uniform):
//   0: activations   [M, K]                 (scalar, read)
//   1: w_packed      [(padded_k/8) * N]     (u32, read)
//   2: w_scales      [num_groups, N]         (scalar, read)
//   3: w_zero_points [num_groups, N]         (scalar, read)
//   4: output        [M, N]                  (scalar, rw)
//   5: params        (uniform)

struct QMatMulINT4Params {
    m: u32,
    k: u32,            // original (unpadded) K
    n: u32,
    padded_k: u32,
    group_size: u32,
    num_groups: u32,
    _pad0: u32,
    _pad1: u32,
}
// 32 bytes ✓

@group(0) @binding(0) var<storage, read>       qm_activations: array<scalar>;
@group(0) @binding(1) var<storage, read>       qm_packed:      array<u32>;
@group(0) @binding(2) var<storage, read>       qm_scales:      array<scalar>;
@group(0) @binding(3) var<storage, read>       qm_zp:          array<scalar>;
@group(0) @binding(4) var<storage, read_write> qm_output:      array<scalar>;
@group(0) @binding(5) var<uniform>             qm_params:      QMatMulINT4Params;

@compute @workgroup_size(16, 16)
fn qmatmul_int4_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;  // M dimension (activation row)
    let col = gid.x;  // N dimension (weight column)
    if (row >= qm_params.m || col >= qm_params.n) { return; }

    let p = qm_params;
    var sum: scalar = scalar(0.0);

    // Iterate over groups along K.
    for (var g: u32 = 0u; g < p.num_groups; g++) {
        // Fetch per-group scale and zero-point for this column.
        let meta_idx = g * p.n + col;
        let scale = qm_scales[meta_idx];
        let zp = qm_zp[meta_idx];

        let k_start = g * p.group_size;
        let words_per_group = p.group_size / 8u;

        // Iterate over packed u32 words within this group.
        for (var w: u32 = 0u; w < words_per_group; w++) {
            let k_base = k_start + w * 8u;

            // Fetch one u32 word containing 8 INT4 weights for column `col`.
            let word_idx = (k_base / 8u) * p.n + col;
            let packed = qm_packed[word_idx];

            // Unpack 8 nibbles and accumulate dot product.
            for (var j: u32 = 0u; j < 8u; j++) {
                let k = k_base + j;

                // Skip padding region beyond original K.
                if (k >= p.k) { break; }

                // Extract 4-bit unsigned value.
                let q = (packed >> (j * 4u)) & 0xFu;

                // Dequantize: w = (q - zero_point) × scale.
                let w_val = (scalar(q) - zp) * scale;

                // Dot product with activation.
                sum += qm_activations[row * p.k + k] * w_val;
            }
        }
    }

    qm_output[row * p.n + col] = sum;
}
