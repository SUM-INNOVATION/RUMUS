// Fused INT4 dequantize-matmul TRANSPOSE: Y = A × dequant(W_int4)^T
//
// Used for activation gradients: grad_x = grad_y × dequant(W)^T
// where grad_y is [M, N] and W is [K, N], so W^T is [N, K].
// Result: [M, K].
//
// Thread (row, k_col) computes: Y[row, k_col] = Σ_n A[row, n] × dequant(W[k_col, n])
// The summation is over N (the weight's column dimension).
//
// Bindings (CustomOp: 4 inputs + 1 output + 1 uniform):
//   0: activations   [M, N]                 (scalar, read)  — grad_output
//   1: w_packed      [(padded_k/8) * N]     (u32, read)
//   2: w_scales      [num_groups, N]         (scalar, read)
//   3: w_zero_points [num_groups, N]         (scalar, read)
//   4: output        [M, K]                  (scalar, rw)   — grad_input
//   5: params        (uniform)

struct QMatMulINT4TParams {
    m: u32,
    k: u32,            // original K (output cols)
    n: u32,            // summation dimension
    padded_k: u32,
    group_size: u32,
    num_groups: u32,
    _pad0: u32,
    _pad1: u32,
}
// 32 bytes ✓

@group(0) @binding(0) var<storage, read>       qt_a:      array<scalar>;
@group(0) @binding(1) var<storage, read>       qt_packed: array<u32>;
@group(0) @binding(2) var<storage, read>       qt_scales: array<scalar>;
@group(0) @binding(3) var<storage, read>       qt_zp:     array<scalar>;
@group(0) @binding(4) var<storage, read_write> qt_output: array<scalar>;
@group(0) @binding(5) var<uniform>             qt_params: QMatMulINT4TParams;

@compute @workgroup_size(16, 16)
fn qmatmul_int4_transpose_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;    // M dimension
    let k_col = gid.x;  // K dimension (output column)
    if (row >= qt_params.m || k_col >= qt_params.k) { return; }

    let p = qt_params;
    var sum: scalar = scalar(0.0);

    // For output position (row, k_col), we need W[k_col, n] for all n.
    // The group for k_col is k_col / group_size.
    // But each u32 word packs 8 consecutive K values for a SINGLE n-column.
    // So W[k_col, n] is in word (k_col / 8) * N + n, nibble k_col % 8.

    let word_row = k_col / 8u;
    let nibble = k_col % 8u;
    let shift = nibble * 4u;
    let group = k_col / p.group_size;

    // Sum over all N columns.
    for (var n: u32 = 0u; n < p.n; n++) {
        // Fetch the packed word and extract the nibble for k_col.
        let word_idx = word_row * p.n + n;
        let packed = qt_packed[word_idx];
        let q = (packed >> shift) & 0xFu;

        // Fetch per-group scale/zp for this (group, n).
        let meta_idx = group * p.n + n;
        let scale = qt_scales[meta_idx];
        let zp = qt_zp[meta_idx];

        // Dequantize and accumulate.
        let w_val = (scalar(q) - zp) * scale;
        sum += qt_a[row * p.n + n] * w_val;
    }

    qt_output[row * p.k + k_col] = sum;
}
