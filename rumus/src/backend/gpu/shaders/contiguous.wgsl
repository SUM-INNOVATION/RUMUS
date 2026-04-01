// Strided-to-contiguous copy kernel.
//
// Reads elements from a strided source buffer and writes them into a
// dense (row-major) destination buffer.  Used by Tensor::contiguous()
// to keep non-contiguous GPU tensors on-device.
//
// Each thread handles one output element.  It decomposes the linear
// output index into a multi-index using precomputed suffix products,
// then computes the strided source index.
//
// Supports up to 8 dimensions (sufficient for [batch, C, H, W] and beyond).

struct ContiguousParams {
    numel: u32,
    ndim: u32,
    offset: u32,
    _pad: u32,
    // Followed by shape[8] + strides[8] + suffix[8] packed as u32 arrays.
    shape:   array<u32, 8>,
    strides: array<u32, 8>,
    suffix:  array<u32, 8>,
}
// 4 + 4 + 4 + 4 + 32 + 32 + 32 = 112 bytes
// 112 is a multiple of 16 ✓

@group(0) @binding(0) var<storage, read>       cont_src: array<f32>;
@group(0) @binding(1) var<storage, read_write> cont_dst: array<f32>;
@group(0) @binding(2) var<uniform>             cont_params: ContiguousParams;

@compute @workgroup_size(64)
fn contiguous_copy_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_idx = gid.x;
    if (dst_idx >= cont_params.numel) { return; }

    var src_idx = cont_params.offset;
    var remainder = dst_idx;

    for (var d: u32 = 0u; d < cont_params.ndim; d++) {
        let dim_size = cont_params.suffix[d];
        let coord = remainder / dim_size;
        remainder = remainder % dim_size;
        src_idx += coord * cont_params.strides[d];
    }

    cont_dst[dst_idx] = cont_src[src_idx];
}
