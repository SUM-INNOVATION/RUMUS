// Compute grad_output * x_hat element-wise (for grad_weight reduction).
//
// Reconstructs x_hat = (input - mean) * invstd per instance, then
// computes output[i,j] = grad_out[i,j] * x_hat[i,j].
// The result is then reduced over the instance dimension externally.
//
// Uses the same 6-binding layout as layer_norm_backward (ln_bw_layout):
//   binding 0: grad_out (read)
//   binding 1: input (read)
//   binding 2: (unused, weight placeholder — read)
//   binding 3: save (read) — mean+invstd
//   binding 4: output (rw) — grad_out * x_hat
//   binding 5: uniform

struct LnGradWeightParams {
    num_instances: u32,
    norm_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read>       gw_grad_out: array<scalar>;
@group(0) @binding(1) var<storage, read>       gw_input:    array<scalar>;
@group(0) @binding(2) var<storage, read>       gw_unused:   array<scalar>;
@group(0) @binding(3) var<storage, read>       gw_save:     array<scalar>;
@group(0) @binding(4) var<storage, read_write> gw_output:   array<scalar>;
@group(0) @binding(5) var<uniform>             gw_params:   LnGradWeightParams;

@compute @workgroup_size(64)
fn layer_norm_grad_weight_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = gw_params.num_instances * gw_params.norm_size;
    if (idx >= total) { return; }

    let inst = idx / gw_params.norm_size;
    let mean = gw_save[inst * 2u];
    let invstd = gw_save[inst * 2u + 1u];
    let x_hat = (gw_input[idx] - mean) * invstd;

    gw_output[idx] = gw_grad_out[idx] * x_hat;
}
