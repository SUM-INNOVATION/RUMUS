// AdaptiveAvgPool2d: dynamic-window average pooling.
//
// One thread per output element. Computes dynamic bin boundaries
// using ceiling division for end indices to avoid dropping pixels.

struct AdaptivePoolParams {
    batch: u32,
    channels: u32,
    h_in: u32,
    w_in: u32,
    h_out: u32,
    w_out: u32,
    _pad0: u32,
    _pad1: u32,
}
// 32 bytes ✓

// --- Forward ---

@group(0) @binding(0) var<storage, read>       ap_input:  array<scalar>;
@group(0) @binding(1) var<storage, read_write> ap_output: array<scalar>;
@group(0) @binding(2) var<uniform>             ap_params: AdaptivePoolParams;

@compute @workgroup_size(64)
fn adaptive_avg_pool2d_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = ap_params.batch * ap_params.channels * ap_params.h_out * ap_params.w_out;
    if (idx >= total) { return; }

    let ow = idx % ap_params.w_out;
    let oh = (idx / ap_params.w_out) % ap_params.h_out;
    let c  = (idx / (ap_params.w_out * ap_params.h_out)) % ap_params.channels;
    let b  = idx / (ap_params.w_out * ap_params.h_out * ap_params.channels);

    // Floor for start, ceiling for end.
    let h_start = oh * ap_params.h_in / ap_params.h_out;
    let h_end   = ((oh + 1u) * ap_params.h_in + ap_params.h_out - 1u) / ap_params.h_out;
    let w_start = ow * ap_params.w_in / ap_params.w_out;
    let w_end   = ((ow + 1u) * ap_params.w_in + ap_params.w_out - 1u) / ap_params.w_out;

    let in_spatial = ap_params.h_in * ap_params.w_in;
    let base = b * ap_params.channels * in_spatial + c * in_spatial;
    let count = (h_end - h_start) * (w_end - w_start);

    var sum: scalar = scalar(0.0);
    for (var h: u32 = h_start; h < h_end; h++) {
        for (var w: u32 = w_start; w < w_end; w++) {
            sum += ap_input[base + h * ap_params.w_in + w];
        }
    }
    ap_output[idx] = sum / scalar(count);
}

// --- Backward: thread per input element ---
// Each input element determines which output bins contain it and accumulates.

@group(0) @binding(0) var<storage, read>       apbw_grad_out: array<scalar>;
@group(0) @binding(1) var<storage, read_write> apbw_grad_in:  array<scalar>;
@group(0) @binding(2) var<uniform>             apbw_params:   AdaptivePoolParams;

@compute @workgroup_size(64)
fn adaptive_avg_pool2d_backward_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let in_total = apbw_params.batch * apbw_params.channels * apbw_params.h_in * apbw_params.w_in;
    if (idx >= in_total) { return; }

    let in_spatial = apbw_params.h_in * apbw_params.w_in;
    let iw = idx % apbw_params.w_in;
    let ih = (idx / apbw_params.w_in) % apbw_params.h_in;
    let c  = (idx / in_spatial) % apbw_params.channels;
    let b  = idx / (apbw_params.channels * in_spatial);

    let out_spatial = apbw_params.h_out * apbw_params.w_out;
    let out_base = b * apbw_params.channels * out_spatial + c * out_spatial;

    var grad_val: scalar = scalar(0.0);
    // Find which output bins cover this input pixel.
    for (var oh: u32 = 0u; oh < apbw_params.h_out; oh++) {
        let h_start = oh * apbw_params.h_in / apbw_params.h_out;
        let h_end = ((oh + 1u) * apbw_params.h_in + apbw_params.h_out - 1u) / apbw_params.h_out;
        if (ih < h_start || ih >= h_end) { continue; }
        for (var ow: u32 = 0u; ow < apbw_params.w_out; ow++) {
            let w_start = ow * apbw_params.w_in / apbw_params.w_out;
            let w_end = ((ow + 1u) * apbw_params.w_in + apbw_params.w_out - 1u) / apbw_params.w_out;
            if (iw < w_start || iw >= w_end) { continue; }
            let count = (h_end - h_start) * (w_end - w_start);
            grad_val += apbw_grad_out[out_base + oh * apbw_params.w_out + ow] / scalar(count);
        }
    }
    apbw_grad_in[idx] = grad_val;
}
