// Batched matrix multiplication.
//
// C[b, i, j] = Σ_p A[b, i, p] * B[b, p, j]
//
// gid.x → column (n), gid.y → row (m), gid.z → batch (B).
// Dispatch: ((n+15)/16, (m+15)/16, batch).

struct BmmParams {
    batch: u32,
    m: u32,
    k: u32,
    n: u32,
}
// 16 bytes ✓

@group(0) @binding(0) var<storage, read>       bmm_a:      array<scalar>;
@group(0) @binding(1) var<storage, read>       bmm_b:      array<scalar>;
@group(0) @binding(2) var<storage, read_write> bmm_out:    array<scalar>;
@group(0) @binding(3) var<uniform>             bmm_params: BmmParams;

@compute @workgroup_size(16, 16)
fn bmm_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b   = gid.z;
    let row = gid.y;
    let col = gid.x;
    if (b >= bmm_params.batch || row >= bmm_params.m || col >= bmm_params.n) { return; }

    let a_off = b * bmm_params.m * bmm_params.k;
    let b_off = b * bmm_params.k * bmm_params.n;
    let c_off = b * bmm_params.m * bmm_params.n;

    var sum: scalar = scalar(0.0);
    for (var p: u32 = 0u; p < bmm_params.k; p++) {
        sum += bmm_a[a_off + row * bmm_params.k + p] * bmm_b[b_off + p * bmm_params.n + col];
    }
    bmm_out[c_off + row * bmm_params.n + col] = sum;
}
