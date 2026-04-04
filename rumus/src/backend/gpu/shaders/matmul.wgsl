// Naive O(M*K*N) matrix multiply.
//
// Each thread computes one element of the output matrix.
// A is (m x k), B is (k x n), C is (m x n), all row-major.
//
// Dispatch: ((n + 15) / 16, (m + 15) / 16, 1)
// gid.x → column (n), gid.y → row (m)

struct MatmulParams {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       a: array<scalar>;
@group(0) @binding(1) var<storage, read>       b: array<scalar>;
@group(0) @binding(2) var<storage, read_write> out: array<scalar>;
@group(0) @binding(3) var<uniform>             params: MatmulParams;

@compute @workgroup_size(16, 16)
fn matmul_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= params.m || col >= params.n) { return; }

    var sum: scalar = scalar(0.0);
    for (var p: u32 = 0u; p < params.k; p = p + 1u) {
        sum = sum + a[row * params.k + p] * b[p * params.n + col];
    }
    out[row * params.n + col] = sum;
}
