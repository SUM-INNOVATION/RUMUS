// Dropout forward kernel.
//
// Uses a PCG hash for deterministic per-element PRNG.
// The seed (passed from CPU) + global_invocation_id produces a uniform
// u32 that is compared against an integer threshold (p * 2^32) to
// decide if the element is dropped.  No float conversion needed for
// the comparison — exact.
//
// Bindings reuse pool_layout: input(read) + output(rw) + mask(rw) + uniform.

struct DropoutParams {
    numel: u32,
    seed: u32,
    p_threshold: u32,  // floor(p * 2^32)
    scale: f32,        // 1.0 / (1.0 - p)
}
// 4 * 4 = 16 bytes ✓

@group(0) @binding(0) var<storage, read>       dropout_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> dropout_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> dropout_mask: array<f32>;
@group(0) @binding(3) var<uniform>             dropout_params: DropoutParams;

/// PCG-style hash: mix a single u32 into a pseudorandom u32.
fn pcg_hash(input_val: u32) -> u32 {
    var state = input_val;
    state = state * 747796405u + 2891336453u;
    state = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (state >> 22u) ^ state;
}

@compute @workgroup_size(64)
fn dropout_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dropout_params.numel) { return; }

    let hash = pcg_hash(dropout_params.seed ^ i);

    if (hash < dropout_params.p_threshold) {
        // Dropped.
        dropout_output[i] = 0.0;
        dropout_mask[i] = 0.0;
    } else {
        // Kept, scaled by 1/(1-p).
        dropout_output[i] = dropout_input[i] * dropout_params.scale;
        dropout_mask[i] = dropout_params.scale;
    }
}
