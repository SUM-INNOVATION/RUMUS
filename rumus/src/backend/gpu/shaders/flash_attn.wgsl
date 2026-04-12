// FlashAttention: memory-efficient scaled dot-product attention.
//
// 1D workgroup: each thread owns one query row.  Threads cooperatively
// load K/V blocks into shared memory, then each thread independently
// computes its row's attention via online softmax.  No cross-thread
// reductions needed.
//
// Dispatch: (ceil(N / B_r), batch_heads, 1)

struct FlashAttnParams {
    batch_heads: u32,
    seq_len: u32,
    head_dim: u32,
    b_r: u32,
    b_c: u32,
    num_col_blocks: u32,
    scale: f32,
    causal: u32,
}
// 32 bytes ✓

@group(0) @binding(0) var<storage, read>       fa_q: array<scalar>;
@group(0) @binding(1) var<storage, read>       fa_k: array<scalar>;
@group(0) @binding(2) var<storage, read>       fa_v: array<scalar>;
@group(0) @binding(3) var<storage, read_write> fa_o: array<scalar>;
@group(0) @binding(4) var<uniform>             fa_params: FlashAttnParams;

// Shared memory for K and V blocks.
// Max allocation: B_c=16, d=128 → 2048 scalars each.
// F32: 2048×4×2 = 16384 bytes = exactly 16KB budget.
var<workgroup> k_shared: array<scalar, 2048>;
var<workgroup> v_shared: array<scalar, 2048>;

@compute @workgroup_size(64, 1, 1)
fn flash_attn_kernel(
    @builtin(local_invocation_id)  lid:  vec3<u32>,
    @builtin(workgroup_id)         wgid: vec3<u32>,
) {
    let p = fa_params;
    let ty = lid.x;
    let row_block = wgid.x;
    let bh = wgid.y;
    let query_row = row_block * p.b_r + ty;

    // Threads beyond B_r still participate in cooperative loads
    // but skip all compute.
    let active = ty < p.b_r && query_row < p.seq_len;

    let N = p.seq_len;
    let d = p.head_dim;
    let scale = scalar(p.scale);
    let bh_offset = bh * N * d;

    // ---- Load Q row into private registers ----
    var q_row: array<scalar, 128>;
    if (active) {
        for (var i: u32 = 0u; i < d; i++) {
            q_row[i] = fa_q[bh_offset + query_row * d + i];
        }
    }

    // ---- Online softmax state (per-thread private) ----
    var m_i: scalar = scalar(-1e30);
    var l_i: scalar = scalar(0.0);
    var o_i: array<scalar, 128>;
    for (var i: u32 = 0u; i < d; i++) {
        o_i[i] = scalar(0.0);
    }

    // ---- Iterate over column blocks ----
    for (var col_block: u32 = 0u; col_block < p.num_col_blocks; col_block++) {
        let col_start = col_block * p.b_c;

        // == Phase 1: Cooperative load K_block and V_block ==
        // All threads (including inactive ones) participate to ensure
        // the shared memory is fully populated.
        let total_elems = p.b_c * d;
        var load_idx = ty;
        while (load_idx < total_elems) {
            let kv_row = load_idx / d;
            let kv_col = load_idx % d;
            let global_row = col_start + kv_row;

            if (global_row < N) {
                let src = bh_offset + global_row * d + kv_col;
                k_shared[load_idx] = fa_k[src];
                v_shared[load_idx] = fa_v[src];
            } else {
                k_shared[load_idx] = scalar(0.0);
                v_shared[load_idx] = scalar(0.0);
            }
            load_idx += 64u;  // stride by workgroup_size
        }
        workgroupBarrier();

        // == Phase 2: Compute scores + online softmax (active threads only) ==
        if (active) {
            // 2a: Compute dot products and find block-local max.
            var scores: array<scalar, 16>;  // B_c max = 16
            var m_block: scalar = scalar(-1e30);
            var valid_count: u32 = 0u;

            for (var tx: u32 = 0u; tx < p.b_c; tx++) {
                let col_idx = col_start + tx;
                if (col_idx >= N) { break; }
                if (p.causal == 1u && col_idx > query_row) { break; }

                var dot: scalar = scalar(0.0);
                let k_base = tx * d;
                for (var i: u32 = 0u; i < d; i++) {
                    dot += q_row[i] * k_shared[k_base + i];
                }
                dot *= scale;
                scores[tx] = dot;
                m_block = max(m_block, dot);
                valid_count = tx + 1u;
            }

            // 2b: Online softmax update.
            let m_new = max(m_i, m_block);
            let alpha = exp(m_i - m_new);

            // Rescale previous accumulator ONCE.
            for (var i: u32 = 0u; i < d; i++) {
                o_i[i] = o_i[i] * alpha;
            }
            l_i = l_i * alpha;

            // 2c: Accumulate new block's contribution.
            for (var tx: u32 = 0u; tx < valid_count; tx++) {
                let w = exp(scores[tx] - m_new);
                l_i += w;

                let v_base = tx * d;
                for (var i: u32 = 0u; i < d; i++) {
                    o_i[i] += w * v_shared[v_base + i];
                }
            }

            m_i = m_new;
        }

        workgroupBarrier();
    }

    // ---- Final normalization: O = o_i / l_i ----
    if (active && l_i > scalar(0.0)) {
        let inv_l = scalar(1.0) / l_i;
        for (var i: u32 = 0u; i < d; i++) {
            fa_o[bh_offset + query_row * d + i] = o_i[i] * inv_l;
        }
    }
}
