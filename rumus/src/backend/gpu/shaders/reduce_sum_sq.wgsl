// reduce_sum_sq: compute sum of squares of all elements.
//
// Output[0] = sum(input[i]^2) for i in 0..numel.
// Single workgroup, shared memory tree reduction.

struct ReduceSumSqParams {
    numel: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
// 16 bytes ✓

@group(0) @binding(0) var<storage, read>       rss_input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> rss_output: array<f32>;
@group(0) @binding(2) var<uniform>             rss_params: ReduceSumSqParams;

var<workgroup> rss_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce_sum_sq_kernel(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    var local_sum: f32 = 0.0;

    // Grid-stride loop: each thread accumulates multiple elements.
    var i = tid;
    loop {
        if (i >= rss_params.numel) { break; }
        let val = rss_input[i];
        local_sum += val * val;
        i += 256u;
    }

    rss_shared[tid] = local_sum;
    workgroupBarrier();

    // Tree reduction in shared memory.
    var stride: u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            rss_shared[tid] += rss_shared[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (tid == 0u) {
        rss_output[0] = rss_shared[0];
    }
}
