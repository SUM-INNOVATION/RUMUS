// Mixed-precision matmul: C = A @ B where A is scalar, B is Q8.
//
// A: [M, K] in scalar (F32 or F16) — activations
// B: [K, N] packed as Q8 in column-major block order
// C: [M, N] in scalar — output
//
// B layout: for each column n, blocks of K elements are stored contiguously:
//   col n: [header_0][i8 × BS][header_1][i8 × BS]...
//   col n+1: [header_0][i8 × BS]...
//
// Each workgroup computes a 16×16 tile of C.
// The K dimension is traversed in steps of block_size.

struct MatmulQ8Params {
    m: u32,
    k: u32,
    n: u32,
    block_size: u32,
    block_stride_u32: u32,  // (4 + block_size) / 4
    blocks_per_col: u32,    // ceil(K / block_size)
    _pad0: u32,
    _pad1: u32,
}
// 32 bytes ✓

@group(0) @binding(0) var<storage, read>       mq_a:      array<scalar>; // [M, K]
@group(0) @binding(1) var<storage, read>       mq_b:      array<u32>;    // Q8 packed [K, N]
@group(0) @binding(2) var<storage, read_write> mq_c:      array<scalar>; // [M, N]
@group(0) @binding(3) var<uniform>             mq_params: MatmulQ8Params;

/// Extract a signed i8 from a u32 word at the given byte position (0..3).
fn extract_i8(word: u32, byte_pos: u32) -> i32 {
    let raw = (word >> (byte_pos * 8u)) & 0xFFu;
    return select(i32(raw), i32(raw) - 256, raw > 127u);
}

@compute @workgroup_size(16, 16)
fn matmul_q8_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;  // M dimension
    let col = gid.x;  // N dimension
    if (row >= mq_params.m || col >= mq_params.n) { return; }

    var sum: scalar = scalar(0.0);

    // Column `col` of B: its Q8 blocks start at
    // col * blocks_per_col * block_stride_u32 (in u32 words).
    let col_base = col * mq_params.blocks_per_col * mq_params.block_stride_u32;

    // Traverse K in block-sized steps.
    var k_offset: u32 = 0u;
    for (var blk: u32 = 0u; blk < mq_params.blocks_per_col; blk++) {
        let block_word_base = col_base + blk * mq_params.block_stride_u32;

        // Read f16 scale from the 4-byte header.
        let header = mq_b[block_word_base];
        let scale_f32 = unpack2x16float(header).x;
        let blk_scale = scalar(scale_f32);

        // Data words start at block_word_base + 1.
        let data_base = block_word_base + 1u;

        // Iterate over elements in this block.
        let bs = min(mq_params.block_size, mq_params.k - k_offset);

        // Process 4 elements at a time for vectorized i8 unpacking.
        let full_quads = bs / 4u;
        for (var q: u32 = 0u; q < full_quads; q++) {
            let packed = mq_b[data_base + q];
            let k0 = k_offset + q * 4u;

            // Unpack 4 i8 values and accumulate dot product.
            let q0 = extract_i8(packed, 0u);
            let q1 = extract_i8(packed, 1u);
            let q2 = extract_i8(packed, 2u);
            let q3 = extract_i8(packed, 3u);

            sum += mq_a[row * mq_params.k + k0 + 0u] * scalar(f32(q0)) * blk_scale;
            sum += mq_a[row * mq_params.k + k0 + 1u] * scalar(f32(q1)) * blk_scale;
            sum += mq_a[row * mq_params.k + k0 + 2u] * scalar(f32(q2)) * blk_scale;
            sum += mq_a[row * mq_params.k + k0 + 3u] * scalar(f32(q3)) * blk_scale;
        }

        // Handle remaining elements (< 4).
        let remainder_start = full_quads * 4u;
        if (remainder_start < bs) {
            let packed = mq_b[data_base + full_quads];
            for (var r: u32 = remainder_start; r < bs; r++) {
                let byte_pos = r - remainder_start;
                let q_val = extract_i8(packed, byte_pos);
                sum += mq_a[row * mq_params.k + k_offset + r] * scalar(f32(q_val)) * blk_scale;
            }
        }

        k_offset += mq_params.block_size;
    }

    mq_c[row * mq_params.n + col] = sum;
}
