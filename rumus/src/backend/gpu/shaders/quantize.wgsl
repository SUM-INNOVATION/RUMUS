// Symmetric block quantization: F32/F16 → Q8.
//
// One workgroup per block.  Two phases:
//   1. Parallel reduction for abs_max within the block.
//   2. Each thread quantizes its element and packs i8 values.
//
// Output layout (4-byte header per block):
//   [u32: f16_scale in lower 16 bits, 2B padding] [i8 × block_size]
//
// Bindings (unary_layout):
//   @binding(0) input:  array<scalar> (read)
//   @binding(1) output: array<u32>    (read_write) — packed Q8
//   @binding(2) params: uniform

struct QuantizeParams {
    numel: u32,
    block_size: u32,
    num_blocks: u32,
    block_stride_u32: u32,  // (4 + block_size) / 4, rounded up
}
// 16 bytes ✓

@group(0) @binding(0) var<storage, read>       q_input:  array<scalar>;
@group(0) @binding(1) var<storage, read_write> q_output: array<u32>;
@group(0) @binding(2) var<uniform>             q_params: QuantizeParams;

var<workgroup> shared_abs: array<f32, 64>;

@compute @workgroup_size(64)
fn quantize_kernel(
    @builtin(local_invocation_id)  lid:  vec3<u32>,
    @builtin(workgroup_id)         wgid: vec3<u32>,
) {
    let block_idx = wgid.x;
    if (block_idx >= q_params.num_blocks) { return; }
    let tid = lid.x;
    let bs = q_params.block_size;
    let elem_start = block_idx * bs;

    // Phase 1: find abs_max via parallel reduction.
    // Each thread handles elements with stride 64 within the block.
    var local_max: f32 = 0.0;
    var i = tid;
    while (i < bs) {
        let global_idx = elem_start + i;
        if (global_idx < q_params.numel) {
            local_max = max(local_max, abs(f32(q_input[global_idx])));
        }
        i += 64u;
    }
    shared_abs[tid] = local_max;
    workgroupBarrier();

    var s: u32 = 32u;
    while (s > 0u) {
        if (tid < s) {
            shared_abs[tid] = max(shared_abs[tid], shared_abs[tid + s]);
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    let abs_max = shared_abs[0];
    workgroupBarrier();

    // Compute scale and inv_scale.
    let scale: f32 = select(abs_max / 127.0, 1.0, abs_max == 0.0);
    let inv_scale: f32 = select(127.0 / abs_max, 0.0, abs_max == 0.0);

    // Write the 4-byte header: f16 scale in lower 16 bits of word 0.
    let out_block_base = block_idx * q_params.block_stride_u32;
    if (tid == 0u) {
        let scale_f16_bits = pack2x16float(vec2<f32>(scale, 0.0));
        q_output[out_block_base] = scale_f16_bits;
    }
    workgroupBarrier();

    // Phase 2: quantize + pack i8 values.
    // Data starts at byte offset 4 within the block → word offset 1.
    // Pack 4 i8 values per u32 word.
    let data_word_offset = out_block_base + 1u;
    let num_data_words = (bs + 3u) / 4u;

    // Each thread handles words with stride 64.
    var w = tid;
    while (w < num_data_words) {
        var packed: u32 = 0u;
        for (var byte_idx: u32 = 0u; byte_idx < 4u; byte_idx++) {
            let in_block_pos = w * 4u + byte_idx;
            let global_idx = elem_start + in_block_pos;
            var q_val: i32 = 0;
            if (in_block_pos < bs && global_idx < q_params.numel) {
                let val = f32(q_input[global_idx]);
                q_val = clamp(i32(round(val * inv_scale)), -128, 127);
            }
            // Pack as unsigned byte into the u32 word.
            packed = packed | ((u32(q_val) & 0xFFu) << (byte_idx * 8u));
        }
        q_output[data_word_offset + w] = packed;
        w += 64u;
    }
}
