// Fused Sparse-Dense Matrix Multiplication (SpMM).
//
// output[i, :] = Σ_{j ∈ neighbors(i)} A[i,j] * features[j, :]
//
// CSR format: row_ptr[N+1], col_indices[E], values[E].
// 1 thread = 1 node.  Edge-outer / dim-inner loop for cache locality.
//
// Bindings (CustomOp convention: 4 inputs + 1 output + 1 uniform = 6):
//   0: row_ptr      (u32, read)
//   1: col_indices   (u32, read)
//   2: values        (scalar, read)   — edge weights (or dummy if unweighted)
//   3: features      (scalar, read)   — [N, D] dense node features
//   4: output        (scalar, rw)     — [N, D] output
//   5: params        (uniform)

struct SpMMParams {
    num_nodes: u32,
    num_edges: u32,
    hidden_dim: u32,
    has_values: u32,
}
// 16 bytes ✓

@group(0) @binding(0) var<storage, read>       spmm_row_ptr:     array<u32>;
@group(0) @binding(1) var<storage, read>       spmm_col_indices: array<u32>;
@group(0) @binding(2) var<storage, read>       spmm_values:      array<scalar>;
@group(0) @binding(3) var<storage, read>       spmm_features:    array<scalar>;
@group(0) @binding(4) var<storage, read_write> spmm_output:      array<scalar>;
@group(0) @binding(5) var<uniform>             spmm_params:      SpMMParams;

@compute @workgroup_size(256)
fn spmm_forward_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let node = gid.x;
    if (node >= spmm_params.num_nodes) { return; }

    let D = spmm_params.hidden_dim;
    let start = spmm_row_ptr[node];
    let end = spmm_row_ptr[node + 1u];
    let out_base = node * D;

    // Zero the output row.
    for (var d: u32 = 0u; d < D; d++) {
        spmm_output[out_base + d] = scalar(0.0);
    }

    // Edge-outer loop: for each neighbor, accumulate weight * features.
    // This reads col_indices[e] and values[e] once per edge, then sweeps
    // the inner D dimension with stride-1 access on features — cache-friendly.
    for (var e: u32 = start; e < end; e++) {
        let neighbor = spmm_col_indices[e];
        let w = select(scalar(1.0), spmm_values[e], spmm_params.has_values == 1u);
        let feat_base = neighbor * D;

        for (var d: u32 = 0u; d < D; d++) {
            spmm_output[out_base + d] += w * spmm_features[feat_base + d];
        }
    }
}
