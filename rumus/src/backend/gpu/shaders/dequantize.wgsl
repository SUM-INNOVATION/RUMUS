// Dequantize Q8 → scalar (F32 or F16).
//
// One thread per output element.
//
// Input layout (4-byte header per block):
//   [u32: f16_scale in lower 16 bits, 2B padding] [i8 × block_size]
//
// Bindings (unary_layout):
//   @binding(0) input:  array<u32>    (read)  — packed Q8
//   @binding(1) output: array<scalar> (read_write)
//   @binding(2) params: uniform

struct DequantizeParams {
    numel: u32,
    block_size: u32,
    block_stride_u32: u32,  // (4 + block_size) / 4, rounded up
    _pad: u32,
}
// 16 bytes ✓

@group(0) @binding(0) var<storage, read>       dq_input:  array<u32>;
@group(0) @binding(1) var<storage, read_write> dq_output: array<scalar>;
@group(0) @binding(2) var<uniform>             dq_params: DequantizeParams;

/// Extract a signed i8 from a u32 word at the given byte position (0..3).
fn extract_i8(word: u32, byte_pos: u32) -> i32 {
    let raw = (word >> (byte_pos * 8u)) & 0xFFu;
    // Sign-extend: if bit 7 is set, value is negative.
    return select(i32(raw), i32(raw) - 256, raw > 127u);
}

@compute @workgroup_size(256)
fn dequantize_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= dq_params.numel) { return; }

    let block_idx = idx / dq_params.block_size;
    let in_block_pos = idx % dq_params.block_size;

    let block_base = block_idx * dq_params.block_stride_u32;

    // Read f16 scale from lower 16 bits of header word.
    let header_word = dq_input[block_base];
    let scale_vec = unpack2x16float(header_word);
    let scale = scale_vec.x;  // f32

    // Read the i8 value from the data region (starts at word offset 1).
    let data_word_idx = block_base + 1u + in_block_pos / 4u;
    let byte_pos = in_block_pos % 4u;
    let q_val = extract_i8(dq_input[data_word_idx], byte_pos);

    dq_output[idx] = scalar(f32(q_val) * scale);
}
