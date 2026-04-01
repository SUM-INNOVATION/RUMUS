// GPU-fused Cross-Entropy Loss.
//
// Pass 1 (cross_entropy_forward_kernel):
//   For each batch element, computes the per-sample loss AND the gradient
//   in a single pass using the Log-Sum-Exp trick for numerical stability.
//   One workgroup per batch element; reduction via workgroup shared memory.
//
// Pass 2 (reduce_loss_kernel):
//   Sums per-batch losses into a single scalar.

struct CrossEntropyParams {
    batch: u32,
    num_classes: u32,
    _pad0: u32,
    _pad1: u32,
}
// 16 bytes ✓

// --- Pass 1: Forward + Gradient ---

@group(0) @binding(0) var<storage, read>       ce_logits:     array<f32>;  // [B, C]
@group(0) @binding(1) var<storage, read>       ce_targets:    array<f32>;  // [B] (class indices as f32)
@group(0) @binding(2) var<storage, read_write> ce_grad:       array<f32>;  // [B, C]
@group(0) @binding(3) var<storage, read_write> ce_loss_per_b: array<f32>;  // [B]
@group(0) @binding(4) var<uniform>             ce_params:     CrossEntropyParams;

var<workgroup> shared_val: array<f32, 64>;

// Each workgroup handles one batch element.
// gid.x = batch index, local_id.x = thread within workgroup.
@compute @workgroup_size(64)
fn cross_entropy_forward_kernel(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let b = wgid.x;
    if (b >= ce_params.batch) { return; }

    let C = ce_params.num_classes;
    let B = ce_params.batch;
    let tid = lid.x;
    let row_start = b * C;

    // ---- Step 1: Find max logit (reduction) ----
    var local_max: f32 = -3.402823e+38;
    var c = tid;
    while (c < C) {
        local_max = max(local_max, ce_logits[row_start + c]);
        c += 64u;
    }
    shared_val[tid] = local_max;
    workgroupBarrier();

    // Parallel reduction for max.
    var stride: u32 = 32u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_val[tid] = max(shared_val[tid], shared_val[tid + stride]);
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let max_z = shared_val[0];
    workgroupBarrier();

    // ---- Step 2: Compute sum of exp(z - max) ----
    var local_sum: f32 = 0.0;
    c = tid;
    while (c < C) {
        local_sum += exp(ce_logits[row_start + c] - max_z);
        c += 64u;
    }
    shared_val[tid] = local_sum;
    workgroupBarrier();

    stride = 32u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_val[tid] += shared_val[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let sum_exp = shared_val[0];
    workgroupBarrier();

    // ---- Step 3: Write per-batch loss ----
    if (tid == 0u) {
        let target_class = u32(ce_targets[b]);
        let log_sum_exp = max_z + log(sum_exp);
        ce_loss_per_b[b] = (-ce_logits[row_start + target_class] + log_sum_exp) / f32(B);
    }

    // ---- Step 4: Write gradient: softmax - one_hot, scaled by 1/B ----
    let target_class = u32(ce_targets[b]);
    let inv_b = 1.0 / f32(B);
    c = tid;
    while (c < C) {
        let softmax_c = exp(ce_logits[row_start + c] - max_z) / sum_exp;
        let one_hot = select(0.0, 1.0, c == target_class);
        ce_grad[row_start + c] = (softmax_c - one_hot) * inv_b;
        c += 64u;
    }
}

// --- Pass 2: Reduce per-batch losses to scalar ---

struct ReduceParams {
    numel: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
// 16 bytes ✓

@group(0) @binding(0) var<storage, read>       reduce_input:  array<f32>;  // [B]
@group(0) @binding(1) var<storage, read_write> reduce_output: array<f32>;  // [1]
@group(0) @binding(2) var<uniform>             reduce_params: ReduceParams;

var<workgroup> reduce_shared: array<f32, 64>;

@compute @workgroup_size(64)
fn reduce_loss_kernel(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    var local_sum: f32 = 0.0;
    var i = tid;
    while (i < reduce_params.numel) {
        local_sum += reduce_input[i];
        i += 64u;
    }
    reduce_shared[tid] = local_sum;
    workgroupBarrier();

    var stride: u32 = 32u;
    while (stride > 0u) {
        if (tid < stride) {
            reduce_shared[tid] += reduce_shared[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (tid == 0u) {
        reduce_output[0] = reduce_shared[0];
    }
}
