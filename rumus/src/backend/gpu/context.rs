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
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipelines: PipelineCache,
    pub pool: BufferPool,
    pub supports_f16: bool,
}

/// Prepend a `scalar` type alias to a WGSL shader source.
///
/// - `F32`: `alias scalar = f32;\n`
/// - `F16`: `enable f16;\nalias scalar = f16;\n`
pub fn preprocess_shader(source: &str, dtype: crate::tensor::DType) -> String {
    match dtype {
        crate::tensor::DType::F32 => format!("alias scalar = f32;\n{}", source),
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
                    device,
                    queue,
                    pipelines,
                    pool,
                    supports_f16: has_f16,
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
