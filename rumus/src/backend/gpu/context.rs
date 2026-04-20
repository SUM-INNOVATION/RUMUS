// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Process-global GPU device context with pre-compiled pipeline cache.

use std::sync::OnceLock;

use super::pool::BufferPool;

/// Standard buffer usage for tensors: storage read/write + staging copies.
pub const STORAGE_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

/// Pre-compiled compute pipelines for all RUMUS operations.
///
/// Compiled once at `GpuContext` init time, then immutable.
/// Named struct fields give compile-time guarantees that every op has a
/// pipeline — no `HashMap` lookup that can fail at runtime.
pub struct PipelineCache {
    // Binary element-wise ops (add, sub, mul, relu_backward)
    pub binary_layout: wgpu::BindGroupLayout,
    pub add_pipeline: wgpu::ComputePipeline,
    pub sub_pipeline: wgpu::ComputePipeline,
    pub mul_pipeline: wgpu::ComputePipeline,
    pub relu_bw_pipeline: wgpu::ComputePipeline,

    // Unary element-wise ops (relu, scale)
    pub unary_layout: wgpu::BindGroupLayout,
    pub relu_pipeline: wgpu::ComputePipeline,
    pub scale_pipeline: wgpu::ComputePipeline,

    // Matrix multiply
    pub matmul_layout: wgpu::BindGroupLayout,
    pub matmul_pipeline: wgpu::ComputePipeline,

    // Bias ops (add_bias, sum_rows)
    pub bias_layout: wgpu::BindGroupLayout,
    pub add_bias_pipeline: wgpu::ComputePipeline,
    pub sum_rows_pipeline: wgpu::ComputePipeline,

    // Optimizer ops
    pub sgd_layout: wgpu::BindGroupLayout,
    pub sgd_pipeline: wgpu::ComputePipeline,
    pub adam_layout: wgpu::BindGroupLayout,
    pub adam_pipeline: wgpu::ComputePipeline,
    pub adamw_pipeline: wgpu::ComputePipeline,

    // Cross-entropy loss
    pub ce_layout: wgpu::BindGroupLayout,
    pub ce_forward_pipeline: wgpu::ComputePipeline,
    pub ce_reduce_pipeline: wgpu::ComputePipeline,

    // Broadcast binary ops + reduce_sum
    pub broadcast_add_pipeline: wgpu::ComputePipeline,
    pub broadcast_sub_pipeline: wgpu::ComputePipeline,
    pub broadcast_mul_pipeline: wgpu::ComputePipeline,
    pub reduce_sum_pipeline: wgpu::ComputePipeline,

    // LayerNorm (6-binding layout: input + weight + bias + output + save + uniform)
    pub ln_layout: wgpu::BindGroupLayout,
    pub ln_forward_pipeline: wgpu::ComputePipeline,
    pub ln_bw_layout: wgpu::BindGroupLayout,
    pub ln_backward_pipeline: wgpu::ComputePipeline,
    pub ln_grad_weight_pipeline: wgpu::ComputePipeline,

    // BatchNorm2d (8-binding layout)
    pub bn_layout: wgpu::BindGroupLayout,
    pub bn_forward_pipeline: wgpu::ComputePipeline,
    pub bn_backward_pipeline: wgpu::ComputePipeline,

    // AdaptiveAvgPool2d (reuses unary_layout)
    pub adaptive_pool_fwd_pipeline: wgpu::ComputePipeline,
    pub adaptive_pool_bw_pipeline: wgpu::ComputePipeline,

    // Bmm (reuses matmul_layout)
    pub bmm_pipeline: wgpu::ComputePipeline,

    // Softmax (forward: unary_layout, backward: binary_layout)
    pub softmax_forward_pipeline: wgpu::ComputePipeline,
    pub softmax_backward_pipeline: wgpu::ComputePipeline,

    // Embedding (reuses binary_layout)
    pub embedding_pipeline: wgpu::ComputePipeline,

    // Fused stride-aware scale (for negate, etc.)
    pub fused_scale_pipeline: wgpu::ComputePipeline,

    // Broadcast scale: dst[i] = src[i] * scalar_buf[0]
    pub broadcast_scale_pipeline: wgpu::ComputePipeline,

    // Pool ops
    pub pool_layout: wgpu::BindGroupLayout,
    pub max_pool2d_pipeline: wgpu::ComputePipeline,
    pub pool_bw_layout: wgpu::BindGroupLayout,
    pub max_pool2d_bw_pipeline: wgpu::ComputePipeline,

    // Contiguous copy (reuses unary_layout: src(read) + dst(rw) + uniform)
    pub contiguous_copy_pipeline: wgpu::ComputePipeline,

    // Dropout (reuses pool_layout: input(read) + output(rw) + mask(rw) + uniform)
    pub dropout_pipeline: wgpu::ComputePipeline,
    pub fused_dropout_pipeline: wgpu::ComputePipeline,

    // Conv ops (im2col, col2im share unary_layout; channel bias uses bias_layout)
    pub im2col_pipeline: wgpu::ComputePipeline,
    pub col2im_pipeline: wgpu::ComputePipeline,
    pub add_channel_bias_pipeline: wgpu::ComputePipeline,
    pub sum_channel_bias_grad_pipeline: wgpu::ComputePipeline,

    // Gradient clipping: reduce sum of squares (reuses unary_layout)
    pub reduce_sum_sq_pipeline: wgpu::ComputePipeline,

    // Cast kernels (f32↔f16, only compiled when supports_f16)
    pub cast_f32_to_f16_pipeline: Option<wgpu::ComputePipeline>,
    pub cast_f16_to_f32_pipeline: Option<wgpu::ComputePipeline>,

    // INT8 quantization (reuses unary_layout for quantize/dequantize)
    pub quantize_pipeline: wgpu::ComputePipeline,
    pub dequantize_pipeline: wgpu::ComputePipeline,

    // Mixed-precision matmul: scalar A × Q8 B → scalar C
    pub q8_matmul_layout: wgpu::BindGroupLayout,
    pub matmul_q8_pipeline: wgpu::ComputePipeline,

    // FlashAttention: Q(read) + K(read) + V(read) + O(rw) + uniform
    pub flash_attn_layout: wgpu::BindGroupLayout,
    pub flash_attn_pipeline: wgpu::ComputePipeline,

    // F16 pipeline variants (None if GPU doesn't support shader-f16)
    pub f16: Option<F16Pipelines>,
}

/// F16 pipeline variants for all data-path operations.
///
/// These are compiled with `enable f16; alias scalar = f16;` prepended
/// to the shader source.  Optimizer pipelines are NOT included — optimizer
/// state always stays F32.
pub struct F16Pipelines {
    // Element-wise
    pub add_pipeline: wgpu::ComputePipeline,
    pub sub_pipeline: wgpu::ComputePipeline,
    pub mul_pipeline: wgpu::ComputePipeline,
    pub relu_bw_pipeline: wgpu::ComputePipeline,
    pub relu_pipeline: wgpu::ComputePipeline,
    pub scale_pipeline: wgpu::ComputePipeline,
    // Matmul
    pub matmul_pipeline: wgpu::ComputePipeline,
    // Bias
    pub add_bias_pipeline: wgpu::ComputePipeline,
    pub sum_rows_pipeline: wgpu::ComputePipeline,
    // Broadcast
    pub broadcast_add_pipeline: wgpu::ComputePipeline,
    pub broadcast_sub_pipeline: wgpu::ComputePipeline,
    pub broadcast_mul_pipeline: wgpu::ComputePipeline,
    pub reduce_sum_pipeline: wgpu::ComputePipeline,
    // Activations
    pub sigmoid_pipeline: wgpu::ComputePipeline,
    pub tanh_pipeline: wgpu::ComputePipeline,
    pub gelu_pipeline: wgpu::ComputePipeline,
    pub leaky_relu_pipeline: wgpu::ComputePipeline,
    pub sigmoid_bw_pipeline: wgpu::ComputePipeline,
    pub tanh_bw_pipeline: wgpu::ComputePipeline,
    pub gelu_bw_pipeline: wgpu::ComputePipeline,
    pub leaky_relu_bw_pipeline: wgpu::ComputePipeline,
    // Softmax
    pub softmax_forward_pipeline: wgpu::ComputePipeline,
    pub softmax_backward_pipeline: wgpu::ComputePipeline,
    // BMM
    pub bmm_pipeline: wgpu::ComputePipeline,
    // Contiguous copy
    pub contiguous_copy_pipeline: wgpu::ComputePipeline,
    // Broadcast scale
    pub broadcast_scale_pipeline: wgpu::ComputePipeline,
    pub fused_scale_pipeline: wgpu::ComputePipeline,
    // Cross-entropy
    pub ce_forward_pipeline: wgpu::ComputePipeline,
    pub ce_reduce_pipeline: wgpu::ComputePipeline,
}

impl PipelineCache {
    fn new(device: &wgpu::Device, supports_f16: bool) -> Self {
        // ---- Bind group layouts ------------------------------------------------

        let binary_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("binary_layout"),
            entries: &[
                bgl_storage(0, true),
                bgl_storage(1, true),
                bgl_storage_rw(2),
                bgl_uniform(3),
            ],
        });

        let unary_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("unary_layout"),
            entries: &[
                bgl_storage(0, true),
                bgl_storage_rw(1),
                bgl_uniform(2),
            ],
        });

        // matmul uses the same shape as binary: 2 read, 1 rw, 1 uniform
        let matmul_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matmul_layout"),
            entries: &[
                bgl_storage(0, true),
                bgl_storage(1, true),
                bgl_storage_rw(2),
                bgl_uniform(3),
            ],
        });

        // bias: matrix(read) + bias(read) + out(rw) + uniform
        let bias_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bias_layout"),
            entries: &[
                bgl_storage(0, true),
                bgl_storage(1, true),
                bgl_storage_rw(2),
                bgl_uniform(3),
            ],
        });

        // ---- Shader modules ----------------------------------------------------
        // Data-path shaders are preprocessed with `alias scalar = f32;` for the
        // F32 pipeline set.  When supports_f16, a second set is compiled with
        // `enable f16; alias scalar = f16;`.

        use crate::tensor::DType;

        let make_module = |device: &wgpu::Device, label: &str, src: &str, dtype: DType| {
            let processed = preprocess_shader(src, dtype);
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(processed.into()),
            })
        };

        // Raw shader sources (without alias prefix).
        let ew_src = include_str!("shaders/elementwise.wgsl");
        let unary_src = include_str!("shaders/unary.wgsl");
        let mm_src = include_str!("shaders/matmul.wgsl");
        let bias_src = include_str!("shaders/bias.wgsl");

        let ew_module = make_module(device, "elementwise", ew_src, DType::F32);
        let unary_module = make_module(device, "unary", unary_src, DType::F32);
        let mm_module = make_module(device, "matmul", mm_src, DType::F32);
        let bias_module = make_module(device, "bias", bias_src, DType::F32);

        // Pool forward: input(read) + output(rw) + indices(rw) + uniform
        let pool_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pool_layout"),
            entries: &[
                bgl_storage(0, true),
                bgl_storage_rw(1),
                bgl_storage_rw(2),
                bgl_uniform(3),
            ],
        });

        // Pool backward: out_grad(read) + indices(read) + grad_input(rw) + uniform
        let pool_bw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pool_bw_layout"),
            entries: &[
                bgl_storage(0, true),
                bgl_storage(1, true),
                bgl_storage_rw(2),
                bgl_uniform(3),
            ],
        });

        let optim_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("optim"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/optim.wgsl").into(),
            ),
        });

        // Data-path shader sources (preprocessed with scalar alias).
        let ce_src = include_str!("shaders/cross_entropy.wgsl");
        let broadcast_src = include_str!("shaders/broadcast.wgsl");
        let bn_src = include_str!("shaders/batch_norm.wgsl");
        let bn_bw_src = include_str!("shaders/batch_norm_bw.wgsl");
        let ap_src = include_str!("shaders/adaptive_pool.wgsl");
        let bmm_src = include_str!("shaders/bmm.wgsl");
        let softmax_src = include_str!("shaders/softmax.wgsl");
        let softmax_bw_src = include_str!("shaders/softmax_bw.wgsl");
        let ln_src = include_str!("shaders/layer_norm.wgsl");
        let ln_bw_src = include_str!("shaders/layer_norm_bw.wgsl");
        let ln_gw_src = include_str!("shaders/layer_norm_grad_weight.wgsl");
        let embedding_src = include_str!("shaders/embedding.wgsl");
        let fused_scale_src = include_str!("shaders/fused_scale.wgsl");
        let broadcast_scale_src = include_str!("shaders/broadcast_scale.wgsl");
        let contiguous_src = include_str!("shaders/contiguous.wgsl");
        let dropout_src = include_str!("shaders/dropout.wgsl");
        let fused_dropout_src = include_str!("shaders/fused_dropout.wgsl");
        let pool_src = include_str!("shaders/pool.wgsl");
        let conv_src = include_str!("shaders/conv.wgsl");
        let activations_src = include_str!("shaders/activations.wgsl");

        // F32 modules (preprocessed with `alias scalar = f32;`)
        let ce_module = make_module(device, "cross_entropy", ce_src, DType::F32);
        let broadcast_module = make_module(device, "broadcast", broadcast_src, DType::F32);
        let bn_module = make_module(device, "batch_norm", bn_src, DType::F32);
        let bn_bw_module = make_module(device, "batch_norm_bw", bn_bw_src, DType::F32);
        let ap_module = make_module(device, "adaptive_pool", ap_src, DType::F32);
        let bmm_module = make_module(device, "bmm", bmm_src, DType::F32);
        let softmax_module = make_module(device, "softmax", softmax_src, DType::F32);
        let softmax_bw_module = make_module(device, "softmax_bw", softmax_bw_src, DType::F32);
        let ln_module = make_module(device, "layer_norm", ln_src, DType::F32);
        let ln_bw_module = make_module(device, "layer_norm_bw", ln_bw_src, DType::F32);
        let ln_gw_module = make_module(device, "layer_norm_grad_weight", ln_gw_src, DType::F32);
        let embedding_module = make_module(device, "embedding", embedding_src, DType::F32);
        let fused_scale_module = make_module(device, "fused_scale", fused_scale_src, DType::F32);
        let broadcast_scale_module = make_module(device, "broadcast_scale", broadcast_scale_src, DType::F32);
        let contiguous_module = make_module(device, "contiguous", contiguous_src, DType::F32);
        let dropout_module = make_module(device, "dropout", dropout_src, DType::F32);
        let fused_dropout_module = make_module(device, "fused_dropout", fused_dropout_src, DType::F32);
        let pool_module = make_module(device, "pool", pool_src, DType::F32);
        let conv_module = make_module(device, "conv", conv_src, DType::F32);
        let _activations_module = make_module(device, "activations", activations_src, DType::F32);

        // Optimizer shaders: NOT preprocessed (always F32).
        let adamw_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("adamw"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/adamw.wgsl").into()),
        });

        let reduce_sum_sq_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reduce_sum_sq"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/reduce_sum_sq.wgsl").into(),
            ),
        });

        // SGD: grad(read) + vel(rw) + param(rw) + uniform
        let sgd_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sgd_layout"),
            entries: &[
                bgl_storage(0, true),
                bgl_storage_rw(1),
                bgl_storage_rw(2),
                bgl_uniform(3),
            ],
        });

        // Adam: grad(read) + m(rw) + v(rw) + param(rw) + uniform
        let adam_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("adam_layout"),
            entries: &[
                bgl_storage(0, true),
                bgl_storage_rw(1),
                bgl_storage_rw(2),
                bgl_storage_rw(3),
                bgl_uniform(4),
            ],
        });

        // LayerNorm forward: input(read) + weight(read) + bias(read) + output(rw) + save(rw) + uniform
        let ln_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ln_layout"),
            entries: &[
                bgl_storage(0, true),  // input
                bgl_storage(1, true),  // weight
                bgl_storage(2, true),  // bias
                bgl_storage_rw(3),     // output
                bgl_storage_rw(4),     // save (mean+invstd)
                bgl_uniform(5),
            ],
        });

        // LayerNorm backward: grad_out(read) + input(read) + weight(read) + save(read) + grad_in(rw) + uniform
        let ln_bw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ln_bw_layout"),
            entries: &[
                bgl_storage(0, true),  // grad_out
                bgl_storage(1, true),  // input
                bgl_storage(2, true),  // weight
                bgl_storage(3, true),  // save
                bgl_storage_rw(4),     // grad_in
                bgl_uniform(5),
            ],
        });

        // BatchNorm2d: 8 bindings
        let bn_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bn_layout"),
            entries: &[
                bgl_storage(0, true),  // input
                bgl_storage(1, true),  // weight
                bgl_storage(2, true),  // bias
                bgl_storage_rw(3),     // running_mean
                bgl_storage_rw(4),     // running_var
                bgl_storage_rw(5),     // output
                bgl_storage_rw(6),     // save
                bgl_uniform(7),
            ],
        });

        // Cross-entropy: logits(read) + targets(read) + grad(rw) + loss_per_b(rw) + uniform
        let ce_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ce_layout"),
            entries: &[
                bgl_storage(0, true),
                bgl_storage(1, true),
                bgl_storage_rw(2),
                bgl_storage_rw(3),
                bgl_uniform(4),
            ],
        });

        // ---- Pipelines ---------------------------------------------------------

        let make_pipeline = |layout: &wgpu::BindGroupLayout,
                             module: &wgpu::ShaderModule,
                             entry: &str,
                             label: &str|
         -> wgpu::ComputePipeline {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[layout],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        // Build all pipelines before moving layouts into Self.
        let add_pipeline = make_pipeline(&binary_layout, &ew_module, "add_kernel", "add");
        let sub_pipeline = make_pipeline(&binary_layout, &ew_module, "sub_kernel", "sub");
        let mul_pipeline = make_pipeline(&binary_layout, &ew_module, "mul_kernel", "mul");
        let relu_bw_pipeline = make_pipeline(&binary_layout, &ew_module, "relu_backward_kernel", "relu_bw");

        let relu_pipeline = make_pipeline(&unary_layout, &unary_module, "relu_kernel", "relu");
        let scale_pipeline = make_pipeline(&unary_layout, &unary_module, "scale_kernel", "scale");
        let im2col_pipeline = make_pipeline(&unary_layout, &conv_module, "im2col_kernel", "im2col");
        let col2im_pipeline = make_pipeline(&unary_layout, &conv_module, "col2im_kernel", "col2im");

        let matmul_pipeline = make_pipeline(&matmul_layout, &mm_module, "matmul_kernel", "matmul");

        let add_bias_pipeline = make_pipeline(&bias_layout, &bias_module, "add_bias_kernel", "add_bias");
        let sum_rows_pipeline = make_pipeline(&bias_layout, &bias_module, "sum_rows_kernel", "sum_rows");
        let add_channel_bias_pipeline = make_pipeline(&bias_layout, &conv_module, "add_channel_bias_kernel", "add_ch_bias");
        let sum_channel_bias_grad_pipeline = make_pipeline(&bias_layout, &conv_module, "sum_channel_bias_grad_kernel", "sum_ch_bias");

        let sgd_pipeline = make_pipeline(&sgd_layout, &optim_module, "sgd_step", "sgd");
        let adam_pipeline = make_pipeline(&adam_layout, &optim_module, "adam_step", "adam");

        let adamw_pipeline = make_pipeline(&adam_layout, &adamw_module, "adamw_step", "adamw");
        let broadcast_add_pipeline = make_pipeline(&binary_layout, &broadcast_module, "broadcast_add_kernel", "bc_add");
        let broadcast_sub_pipeline = make_pipeline(&binary_layout, &broadcast_module, "broadcast_sub_kernel", "bc_sub");
        let broadcast_mul_pipeline = make_pipeline(&binary_layout, &broadcast_module, "broadcast_mul_kernel", "bc_mul");
        let reduce_sum_pipeline = make_pipeline(&unary_layout, &broadcast_module, "reduce_sum_kernel", "reduce_sum");

        let bn_forward_pipeline = make_pipeline(&bn_layout, &bn_module, "batch_norm_forward_kernel", "bn_fwd");
        let bn_backward_pipeline = make_pipeline(&ln_bw_layout, &bn_bw_module, "batch_norm_backward_kernel", "bn_bw");
        let adaptive_pool_fwd_pipeline = make_pipeline(&unary_layout, &ap_module, "adaptive_avg_pool2d_kernel", "ap_fwd");
        let adaptive_pool_bw_pipeline = make_pipeline(&unary_layout, &ap_module, "adaptive_avg_pool2d_backward_kernel", "ap_bw");

        let bmm_pipeline = make_pipeline(&matmul_layout, &bmm_module, "bmm_kernel", "bmm");
        let softmax_forward_pipeline = make_pipeline(&unary_layout, &softmax_module, "softmax_forward_kernel", "sm_fwd");
        let softmax_backward_pipeline = make_pipeline(&binary_layout, &softmax_bw_module, "softmax_backward_kernel", "sm_bw");

        let ln_forward_pipeline = make_pipeline(&ln_layout, &ln_module, "layer_norm_forward_kernel", "ln_fwd");
        let ln_backward_pipeline = make_pipeline(&ln_bw_layout, &ln_bw_module, "layer_norm_backward_kernel", "ln_bw");
        let ln_grad_weight_pipeline = make_pipeline(&ln_bw_layout, &ln_gw_module, "layer_norm_grad_weight_kernel", "ln_gw");
        let embedding_pipeline = make_pipeline(&binary_layout, &embedding_module, "embedding_kernel", "embedding");

        let fused_scale_pipeline = make_pipeline(&unary_layout, &fused_scale_module, "fused_scale_kernel", "fused_scale");

        let broadcast_scale_pipeline = make_pipeline(&binary_layout, &broadcast_scale_module, "broadcast_scale_kernel", "broadcast_scale");
        let ce_forward_pipeline = make_pipeline(&ce_layout, &ce_module, "cross_entropy_forward_kernel", "ce_forward");
        let ce_reduce_pipeline = make_pipeline(&unary_layout, &ce_module, "reduce_loss_kernel", "ce_reduce");

        let contiguous_copy_pipeline = make_pipeline(&unary_layout, &contiguous_module, "contiguous_copy_kernel", "contiguous_copy");
        let dropout_pipeline = make_pipeline(&pool_layout, &dropout_module, "dropout_kernel", "dropout");
        let fused_dropout_pipeline = make_pipeline(&pool_layout, &fused_dropout_module, "fused_dropout_kernel", "fused_dropout");
        let max_pool2d_pipeline = make_pipeline(&pool_layout, &pool_module, "max_pool2d_kernel", "max_pool2d");
        let max_pool2d_bw_pipeline = make_pipeline(&pool_bw_layout, &pool_module, "max_pool2d_backward_kernel", "max_pool2d_bw");
        let reduce_sum_sq_pipeline = make_pipeline(&unary_layout, &reduce_sum_sq_module, "reduce_sum_sq_kernel", "reduce_sum_sq");

        // ---- INT8 Quantization pipelines -----------------------------------

        let quantize_module = make_module(device, "quantize", include_str!("shaders/quantize.wgsl"), DType::F32);
        let dequantize_module = make_module(device, "dequantize", include_str!("shaders/dequantize.wgsl"), DType::F32);
        let matmul_q8_module = make_module(device, "matmul_q8", include_str!("shaders/matmul_q8.wgsl"), DType::F32);

        let quantize_pipeline = make_pipeline(&unary_layout, &quantize_module, "quantize_kernel", "quantize");
        let dequantize_pipeline = make_pipeline(&unary_layout, &dequantize_module, "dequantize_kernel", "dequantize");

        // Q8 matmul layout: A(scalar read) + B(u32 read) + C(scalar rw) + uniform
        let q8_matmul_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("q8_matmul_layout"),
            entries: &[
                bgl_storage(0, true),   // A: activations
                bgl_storage(1, true),   // B: packed Q8 weights
                bgl_storage_rw(2),      // C: output
                bgl_uniform(3),
            ],
        });
        let matmul_q8_pipeline = make_pipeline(&q8_matmul_layout, &matmul_q8_module, "matmul_q8_kernel", "matmul_q8");

        // ---- FlashAttention pipeline -------------------------------------------
        let flash_attn_src = include_str!("shaders/flash_attn.wgsl");
        let flash_attn_module = make_module(device, "flash_attn", flash_attn_src, DType::F32);
        let flash_attn_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("flash_attn_layout"),
            entries: &[
                bgl_storage(0, true),   // Q
                bgl_storage(1, true),   // K
                bgl_storage(2, true),   // V
                bgl_storage_rw(3),      // O
                bgl_uniform(4),
            ],
        });
        let flash_attn_pipeline = make_pipeline(&flash_attn_layout, &flash_attn_module, "flash_attn_kernel", "flash_attn");

        // ---- Cast and F16 pipelines (conditional) ------------------------------

        let (cast_f32_to_f16_pipeline, cast_f16_to_f32_pipeline, f16_pipelines) = if supports_f16 {
            let cast_src = include_str!("shaders/cast.wgsl");
            let cast_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cast"),
                source: wgpu::ShaderSource::Wgsl(cast_src.into()),
            });
            let cast_down = make_pipeline(&unary_layout, &cast_module, "cast_f32_to_f16_kernel", "cast_f32_f16");
            let cast_up = make_pipeline(&unary_layout, &cast_module, "cast_f16_to_f32_kernel", "cast_f16_f32");

            // Compile F16 variants of data-path shaders.
            let f16_ew = make_module(device, "ew_f16", ew_src, DType::F16);
            let f16_unary = make_module(device, "unary_f16", unary_src, DType::F16);
            let f16_mm = make_module(device, "mm_f16", mm_src, DType::F16);
            let f16_bias = make_module(device, "bias_f16", bias_src, DType::F16);
            let f16_broadcast = make_module(device, "bc_f16", broadcast_src, DType::F16);
            let f16_activations = make_module(device, "act_f16", activations_src, DType::F16);
            let f16_softmax = make_module(device, "sm_f16", softmax_src, DType::F16);
            let f16_softmax_bw = make_module(device, "sm_bw_f16", softmax_bw_src, DType::F16);
            let f16_bmm = make_module(device, "bmm_f16", bmm_src, DType::F16);
            let f16_contiguous = make_module(device, "cont_f16", contiguous_src, DType::F16);
            let f16_broadcast_scale = make_module(device, "bsc_f16", broadcast_scale_src, DType::F16);
            let f16_fused_scale = make_module(device, "fs_f16", fused_scale_src, DType::F16);
            let f16_ce = make_module(device, "ce_f16", ce_src, DType::F16);

            let f16 = F16Pipelines {
                add_pipeline: make_pipeline(&binary_layout, &f16_ew, "add_kernel", "f16_add"),
                sub_pipeline: make_pipeline(&binary_layout, &f16_ew, "sub_kernel", "f16_sub"),
                mul_pipeline: make_pipeline(&binary_layout, &f16_ew, "mul_kernel", "f16_mul"),
                relu_bw_pipeline: make_pipeline(&binary_layout, &f16_ew, "relu_backward_kernel", "f16_relu_bw"),
                relu_pipeline: make_pipeline(&unary_layout, &f16_unary, "relu_kernel", "f16_relu"),
                scale_pipeline: make_pipeline(&unary_layout, &f16_unary, "scale_kernel", "f16_scale"),
                matmul_pipeline: make_pipeline(&matmul_layout, &f16_mm, "matmul_kernel", "f16_matmul"),
                add_bias_pipeline: make_pipeline(&bias_layout, &f16_bias, "add_bias_kernel", "f16_add_bias"),
                sum_rows_pipeline: make_pipeline(&bias_layout, &f16_bias, "sum_rows_kernel", "f16_sum_rows"),
                broadcast_add_pipeline: make_pipeline(&binary_layout, &f16_broadcast, "broadcast_add_kernel", "f16_bc_add"),
                broadcast_sub_pipeline: make_pipeline(&binary_layout, &f16_broadcast, "broadcast_sub_kernel", "f16_bc_sub"),
                broadcast_mul_pipeline: make_pipeline(&binary_layout, &f16_broadcast, "broadcast_mul_kernel", "f16_bc_mul"),
                reduce_sum_pipeline: make_pipeline(&unary_layout, &f16_broadcast, "reduce_sum_kernel", "f16_reduce_sum"),
                sigmoid_pipeline: make_pipeline(&unary_layout, &f16_activations, "sigmoid_kernel", "f16_sigmoid"),
                tanh_pipeline: make_pipeline(&unary_layout, &f16_activations, "tanh_kernel", "f16_tanh"),
                gelu_pipeline: make_pipeline(&unary_layout, &f16_activations, "gelu_kernel", "f16_gelu"),
                leaky_relu_pipeline: make_pipeline(&unary_layout, &f16_activations, "leaky_relu_kernel", "f16_lrelu"),
                sigmoid_bw_pipeline: make_pipeline(&binary_layout, &f16_activations, "sigmoid_backward_kernel", "f16_sigmoid_bw"),
                tanh_bw_pipeline: make_pipeline(&binary_layout, &f16_activations, "tanh_backward_kernel", "f16_tanh_bw"),
                gelu_bw_pipeline: make_pipeline(&binary_layout, &f16_activations, "gelu_backward_kernel", "f16_gelu_bw"),
                leaky_relu_bw_pipeline: make_pipeline(&binary_layout, &f16_activations, "leaky_relu_backward_kernel", "f16_lrelu_bw"),
                softmax_forward_pipeline: make_pipeline(&unary_layout, &f16_softmax, "softmax_forward_kernel", "f16_sm_fwd"),
                softmax_backward_pipeline: make_pipeline(&binary_layout, &f16_softmax_bw, "softmax_backward_kernel", "f16_sm_bw"),
                bmm_pipeline: make_pipeline(&matmul_layout, &f16_bmm, "bmm_kernel", "f16_bmm"),
                contiguous_copy_pipeline: make_pipeline(&unary_layout, &f16_contiguous, "contiguous_copy_kernel", "f16_cont"),
                broadcast_scale_pipeline: make_pipeline(&binary_layout, &f16_broadcast_scale, "broadcast_scale_kernel", "f16_bsc"),
                fused_scale_pipeline: make_pipeline(&unary_layout, &f16_fused_scale, "fused_scale_kernel", "f16_fs"),
                ce_forward_pipeline: make_pipeline(&ce_layout, &f16_ce, "cross_entropy_forward_kernel", "f16_ce_fwd"),
                ce_reduce_pipeline: make_pipeline(&unary_layout, &f16_ce, "reduce_loss_kernel", "f16_ce_reduce"),
            };

            (Some(cast_down), Some(cast_up), Some(f16))
        } else {
            (None, None, None)
        };

        Self {
            binary_layout, add_pipeline, sub_pipeline, mul_pipeline, relu_bw_pipeline,
            unary_layout, relu_pipeline, scale_pipeline,
            matmul_layout, matmul_pipeline,
            bias_layout, add_bias_pipeline, sum_rows_pipeline,
            sgd_layout, sgd_pipeline,
            adam_layout, adam_pipeline, adamw_pipeline,
            ce_layout, ce_forward_pipeline, ce_reduce_pipeline,
            broadcast_add_pipeline, broadcast_sub_pipeline, broadcast_mul_pipeline,
            reduce_sum_pipeline,
            bn_layout, bn_forward_pipeline, bn_backward_pipeline,
            adaptive_pool_fwd_pipeline, adaptive_pool_bw_pipeline,
            bmm_pipeline,
            softmax_forward_pipeline, softmax_backward_pipeline,
            ln_layout, ln_forward_pipeline,
            ln_bw_layout, ln_backward_pipeline, ln_grad_weight_pipeline,
            embedding_pipeline,
            fused_scale_pipeline,
            broadcast_scale_pipeline,
            contiguous_copy_pipeline,
            dropout_pipeline, fused_dropout_pipeline,
            pool_layout, max_pool2d_pipeline,
            pool_bw_layout, max_pool2d_bw_pipeline,
            im2col_pipeline, col2im_pipeline,
            add_channel_bias_pipeline, sum_channel_bias_grad_pipeline,
            reduce_sum_sq_pipeline,
            quantize_pipeline,
            dequantize_pipeline,
            q8_matmul_layout,
            matmul_q8_pipeline,
            flash_attn_layout,
            flash_attn_pipeline,
            cast_f32_to_f16_pipeline,
            cast_f16_to_f32_pipeline,
            f16: f16_pipelines,
        }
    }
}

// --- Helper functions for BindGroupLayoutEntry construction ------------------

fn bgl_storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    bgl_storage(binding, false)
}

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// --- GpuContext --------------------------------------------------------------

/// Process-global GPU context holding device, queue, pipeline cache, and
/// buffer pool.
/// Cache for dynamically compiled custom op pipelines.
pub struct CustomOpCache {
    cache: parking_lot::Mutex<std::collections::HashMap<CustomOpKey, std::sync::Arc<CachedCustomPipeline>>>,
}

/// Key for looking up a custom op's compiled pipeline.
#[derive(Hash, PartialEq, Eq)]
pub struct CustomOpKey {
    pub op_name: String,
    pub dtype_tag: u8,
    pub num_inputs: usize,
}

/// A compiled custom op pipeline + its bind group layout.
pub struct CachedCustomPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
}

impl CustomOpCache {
    pub fn new() -> Self {
        Self {
            cache: parking_lot::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Look up or compile a custom op pipeline.
    pub fn get_or_compile(
        &self,
        key: &CustomOpKey,
        device: &wgpu::Device,
        wgsl: &str,
        entry_point: &str,
        num_inputs: usize,
    ) -> std::sync::Arc<CachedCustomPipeline> {
        // Fast path.
        {
            let cache = self.cache.lock();
            if let Some(entry) = cache.get(key) {
                return std::sync::Arc::clone(entry);
            }
        }

        // Slow path: compile.
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("custom_op"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        // Dynamic bind group layout: N read + 1 rw + 1 uniform.
        let total_bindings = num_inputs + 2; // +1 output, +1 uniform
        let mut entries = Vec::with_capacity(total_bindings);
        for i in 0..num_inputs {
            entries.push(bgl_storage(i as u32, true));
        }
        entries.push(bgl_storage_rw(num_inputs as u32));
        entries.push(bgl_uniform((num_inputs + 1) as u32));

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("custom_op_layout"),
            entries: &entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("custom_op_pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("custom_op_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

        let entry = std::sync::Arc::new(CachedCustomPipeline { pipeline, layout });
        self.cache.lock().insert(CustomOpKey {
            op_name: key.op_name.clone(),
            dtype_tag: key.dtype_tag,
            num_inputs: key.num_inputs,
        }, std::sync::Arc::clone(&entry));
        entry
    }
}

pub struct GpuContext {
    pub device: std::sync::Arc<wgpu::Device>,
    pub queue: std::sync::Arc<wgpu::Queue>,
    pub pipelines: PipelineCache,
    pub pool: BufferPool,
    pub supports_f16: bool,
    pub custom_ops: CustomOpCache,
}

/// Prepend a `scalar` type alias to a WGSL shader source.
///
/// - `F32`: `alias scalar = f32;\n`
/// - `F16`: `enable f16;\nalias scalar = f16;\n`
pub fn preprocess_shader(source: &str, dtype: crate::tensor::DType) -> String {
    match dtype {
        crate::tensor::DType::F32 | crate::tensor::DType::Q8 { .. } => {
            format!("alias scalar = f32;\n{}", source)
        }
        crate::tensor::DType::F16 => format!("enable f16;\nalias scalar = f16;\n{}", source),
    }
}

static GPU_CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

impl GpuContext {
    /// Get the global GPU context, initializing it if needed.
    ///
    /// Returns `None` if no compatible GPU adapter is available — never
    /// panics on missing hardware.
    pub fn get() -> Option<&'static GpuContext> {
        GPU_CONTEXT
            .get_or_init(|| {
                let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
                let adapter = pollster::block_on(instance.request_adapter(
                    &wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        ..Default::default()
                    },
                ))?;

                let has_f16 = adapter.features().contains(wgpu::Features::SHADER_F16);
                let required_features = if has_f16 {
                    wgpu::Features::SHADER_F16
                } else {
                    wgpu::Features::empty()
                };

                let (device, queue) = pollster::block_on(
                    adapter.request_device(
                        &wgpu::DeviceDescriptor {
                            label: Some("rumus_device"),
                            required_features,
                            ..Default::default()
                        },
                        None,
                    ),
                )
                .ok()?;
                let pipelines = PipelineCache::new(&device, has_f16);
                let pool = BufferPool::new();
                Some(GpuContext {
                    device: std::sync::Arc::new(device),
                    queue: std::sync::Arc::new(queue),
                    pipelines,
                    pool,
                    supports_f16: has_f16,
                    custom_ops: CustomOpCache::new(),
                })
            })
            .as_ref()
    }

    /// Returns `true` if a compatible GPU is available.
    pub fn is_available() -> bool {
        Self::get().is_some()
    }

    /// Download GPU buffer data to a CPU `Vec<f32>`.
    pub fn download(&self, source: &wgpu::Buffer, len: usize) -> Vec<f32> {
        let byte_size = (len * std::mem::size_of::<f32>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_readback"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, byte_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("map callback not called")
            .expect("buffer map failed");

        let view = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        staging.unmap();
        result
    }

    /// Download raw bytes from a GPU buffer without any dtype conversion.
    ///
    /// Returns exactly `byte_size` bytes.  Used by the ONNX exporter to
    /// preserve F16 weight data without casting to F32.
    pub fn download_raw_bytes(&self, source: &wgpu::Buffer, byte_size: u64) -> Vec<u8> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_raw_readback"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, byte_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("map callback not called")
            .expect("buffer map failed");

        let view = slice.get_mapped_range();
        let result: Vec<u8> = view.to_vec();
        drop(view);
        staging.unmap();
        result
    }

    /// Upload CPU data to a new GPU buffer.
    pub fn upload(&self, data: &[f32]) -> wgpu::Buffer {
        let byte_size = (data.len() * std::mem::size_of::<f32>()) as u64;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_storage"),
            size: byte_size,
            usage: STORAGE_USAGE,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(data));
        buffer
    }
}

// ---------------------------------------------------------------------------
// Multi-GPU context (feature-gated)
// ---------------------------------------------------------------------------

/// Multi-GPU context: holds all available discrete GPUs.
///
/// Each device gets its own `PipelineCache`, `BufferPool`, and `Queue`.
/// `GpuContext::get()` returns `&devices[0]` for backward compatibility.
#[cfg(feature = "multi_gpu")]
pub struct MultiGpuContext {
    pub devices: Vec<GpuContext>,
}

#[cfg(feature = "multi_gpu")]
static MULTI_GPU: OnceLock<Option<MultiGpuContext>> = OnceLock::new();

#[cfg(feature = "multi_gpu")]
impl MultiGpuContext {
    /// Initialize all discrete GPUs.  Returns `None` if no GPUs found.
    pub fn get() -> Option<&'static MultiGpuContext> {
        MULTI_GPU
            .get_or_init(|| {
                let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
                let all_adapters = instance.enumerate_adapters(wgpu::Backends::all());
                let adapters: Vec<wgpu::Adapter> = all_adapters
                    .into_iter()
                    .filter(|a| {
                        let info = a.get_info();
                        info.device_type == wgpu::DeviceType::DiscreteGpu
                            || info.device_type == wgpu::DeviceType::IntegratedGpu
                    })
                    .collect();

                if adapters.is_empty() {
                    return None;
                }

                let mut devices = Vec::with_capacity(adapters.len());
                for adapter in &adapters {
                    let has_f16 = adapter.features().contains(wgpu::Features::SHADER_F16);
                    let required = if has_f16 {
                        wgpu::Features::SHADER_F16
                    } else {
                        wgpu::Features::empty()
                    };

                    let result = pollster::block_on(adapter.request_device(
                        &wgpu::DeviceDescriptor {
                            label: Some("rumus_multi_gpu"),
                            required_features: required,
                            ..Default::default()
                        },
                        None,
                    ));

                    if let Ok((device, queue)) = result {
                        let pipelines = PipelineCache::new(&device, has_f16);
                        let pool = BufferPool::new();
                        devices.push(GpuContext {
                            device: std::sync::Arc::new(device),
                            queue: std::sync::Arc::new(queue),
                            pipelines,
                            pool,
                            supports_f16: has_f16,
                            custom_ops: CustomOpCache::new(),
                        });
                    }
                }

                if devices.is_empty() {
                    None
                } else {
                    Some(MultiGpuContext { devices })
                }
            })
            .as_ref()
    }

    /// Number of available GPUs.
    pub fn num_devices(&self) -> usize {
        self.devices.len()
    }

    /// Get a specific device by index.
    pub fn device(&self, index: usize) -> &GpuContext {
        &self.devices[index]
    }
}
