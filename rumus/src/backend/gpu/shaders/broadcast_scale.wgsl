// Broadcast-scale kernel: dst[i] = src[i] * scalar_buf[0].
//
// Reads a single scalar from a storage buffer (not a uniform) and
// multiplies every element of the source tensor.  Used by
// CrossEntropyBackward to scale the pre-computed gradient by the
// incoming out_grad scalar — entirely on-device, zero host reads.
//
// Bind group: scalar(read) + src(read) + dst(rw) + uniform(numel).
// Reuses binary_layout: binding 0 (read), 1 (read), 2 (rw), 3 (uniform).

struct BroadcastScaleParams {
    numel: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
// 16 bytes ✓

@group(0) @binding(0) var<storage, read>       bs_scalar: array<scalar>;
@group(0) @binding(1) var<storage, read>       bs_src:    array<scalar>;
@group(0) @binding(2) var<storage, read_write> bs_dst:    array<scalar>;
@group(0) @binding(3) var<uniform>             bs_params: BroadcastScaleParams;

@compute @workgroup_size(64)
fn broadcast_scale_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= bs_params.numel) { return; }
    bs_dst[i] = bs_src[i] * bs_scalar[0];
}
