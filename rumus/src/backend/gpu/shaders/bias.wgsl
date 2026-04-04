// Bias broadcasting and reduction kernels.
//
// add_bias: matrix[m,n] + bias[n] → out[m,n]
// sum_rows: matrix[m,n] → out[n]  (sum over rows)

struct BiasParams {
    m: u32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read>       matrix: array<scalar>;
@group(0) @binding(1) var<storage, read>       bias: array<scalar>;
@group(0) @binding(2) var<storage, read_write> out: array<scalar>;
@group(0) @binding(3) var<uniform>             params: BiasParams;

// Dispatch: ((m*n + 63) / 64, 1, 1)
@compute @workgroup_size(64)
fn add_bias_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.m * params.n) { return; }
    let col = i % params.n;
    out[i] = matrix[i] + bias[col];
}

// Dispatch: ((n + 63) / 64, 1, 1)
// Each thread handles one column, sums across all rows.
@compute @workgroup_size(64)
fn sum_rows_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= params.n) { return; }
    var sum: scalar = scalar(0.0);
    for (var row: u32 = 0u; row < params.m; row = row + 1u) {
        sum = sum + matrix[row * params.n + col];
    }
    out[col] = sum;
}
