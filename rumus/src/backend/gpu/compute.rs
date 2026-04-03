//! GPU compute dispatch functions.
//!
//! Each function retrieves the cached pipeline from [`GpuContext`],
//! creates a fresh uniform buffer (safe for in-flight dispatches),
//! encodes a compute pass, and submits it to the queue.

use super::context::GpuContext;

use wgpu::util::{BufferInitDescriptor, DeviceExt};

// ---------------------------------------------------------------------------
// Uniform parameter structs — must match WGSL layout exactly.
// All padded to 16 bytes per WebGPU uniform buffer size requirement.
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ElementwiseParams {
    numel: u32,
    scalar: f32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulParams {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BiasParams {
    m: u32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SgdParams {
    lr: f32,
    momentum: f32,
    numel: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AdamParams {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    bc1: f32,
    bc2: f32,
    numel: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Im2ColParams {
    c_in: u32,
    h: u32,
    w: u32,
    k: u32,
    stride: u32,
    pad: u32,
    out_h: u32,
    out_w: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ChannelBiasParams {
    channels: u32,
    spatial: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ContiguousParams {
    numel: u32,
    ndim: u32,
    offset: u32,
    _pad: u32,
    shape_lo: [u32; 4],
    shape_hi: [u32; 4],
    strides_lo: [u32; 4],
    strides_hi: [u32; 4],
    suffix_lo: [u32; 4],
    suffix_hi: [u32; 4],
}

// WebGPU requires uniform buffer bindings to be a multiple of 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FusedDropoutParams {
    // Dropout
    numel: u32,
    seed: u32,
    p_threshold: u32,
    scale: f32,
    // Stride
    ndim: u32,
    offset: u32,
    _pad0: u32,
    _pad1: u32,
    shape_lo: [u32; 4],
    shape_hi: [u32; 4],
    strides_lo: [u32; 4],
    strides_hi: [u32; 4],
    suffix_lo: [u32; 4],
    suffix_hi: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CrossEntropyParams {
    batch: u32,
    num_classes: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceParams {
    numel: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BroadcastBinaryParams {
    pub numel: u32,
    pub ndim: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub suffix_lo: [u32; 4],
    pub suffix_hi: [u32; 4],
    pub a_strides_lo: [u32; 4],
    pub a_strides_hi: [u32; 4],
    pub b_strides_lo: [u32; 4],
    pub b_strides_hi: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ReduceSumGpuParams {
    pub out_numel: u32,
    pub ndim: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub in_shape_lo: [u32; 4],
    pub in_shape_hi: [u32; 4],
    pub in_suffix_lo: [u32; 4],
    pub in_suffix_hi: [u32; 4],
    pub out_strides_lo: [u32; 4],
    pub out_strides_hi: [u32; 4],
    pub reduce_extents_lo: [u32; 4],
    pub reduce_extents_hi: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AdamWParams {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    bc1: f32,
    bc2: f32,
    weight_decay: f32,
    numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BroadcastScaleParams {
    numel: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct FusedScaleParams {
    pub numel: u32,
    pub scalar: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub ndim: u32,
    pub offset: u32,
    pub _pad2: u32,
    pub _pad3: u32,
    pub shape_lo: [u32; 4],
    pub shape_hi: [u32; 4],
    pub strides_lo: [u32; 4],
    pub strides_hi: [u32; 4],
    pub suffix_lo: [u32; 4],
    pub suffix_hi: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LayerNormParams {
    pub num_instances: u32,
    pub norm_size: u32,
    pub epsilon: f32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LayerNormBwParams {
    pub num_instances: u32,
    pub norm_size: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct EmbeddingParams {
    pub total_lookups: u32,
    pub embed_dim: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BmmParams {
    pub batch: u32,
    pub m: u32,
    pub k: u32,
    pub n: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SoftmaxParams {
    pub num_rows: u32,
    pub row_size: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

const _: () = assert!(std::mem::size_of::<BmmParams>() == 16);
const _: () = assert!(std::mem::size_of::<SoftmaxParams>() == 16);
const _: () = assert!(std::mem::size_of::<LayerNormParams>() == 16);
const _: () = assert!(std::mem::size_of::<LayerNormBwParams>() == 16);
const _: () = assert!(std::mem::size_of::<EmbeddingParams>() == 16);
const _: () = assert!(std::mem::size_of::<FusedScaleParams>() == 128);
const _: () = assert!(std::mem::size_of::<BroadcastBinaryParams>() == 112);
const _: () = assert!(std::mem::size_of::<ReduceSumGpuParams>() == 144);
const _: () = assert!(std::mem::size_of::<BroadcastScaleParams>() == 16);
const _: () = assert!(std::mem::size_of::<CrossEntropyParams>() == 16);
const _: () = assert!(std::mem::size_of::<ReduceParams>() == 16);
const _: () = assert!(std::mem::size_of::<AdamWParams>() == 32);
const _: () = assert!(std::mem::size_of::<FusedDropoutParams>() == 128);
// 128 = 8 * 16 ✓
const _: () = assert!(std::mem::size_of::<ContiguousParams>() == 112);
// 112 = 7 * 16 ✓
const _: () = assert!(std::mem::size_of::<ElementwiseParams>() == 16);
const _: () = assert!(std::mem::size_of::<MatmulParams>() == 16);
const _: () = assert!(std::mem::size_of::<BiasParams>() == 16);
const _: () = assert!(std::mem::size_of::<SgdParams>() == 16);
const _: () = assert!(std::mem::size_of::<AdamParams>() == 32);
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MaxPool2dParams {
    channels: u32,
    h: u32,
    w: u32,
    k: u32,
    stride: u32,
    out_h: u32,
    out_w: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DropoutParams {
    numel: u32,
    seed: u32,
    p_threshold: u32,
    scale: f32,
}

const _: () = assert!(std::mem::size_of::<Im2ColParams>() == 32);
const _: () = assert!(std::mem::size_of::<ChannelBiasParams>() == 16);
const _: () = assert!(std::mem::size_of::<MaxPool2dParams>() == 32);
const _: () = assert!(std::mem::size_of::<DropoutParams>() == 16);

// ---------------------------------------------------------------------------
// Binary element-wise ops (add, sub, mul, relu_backward)
// ---------------------------------------------------------------------------

/// Helper: dispatch a binary element-wise kernel.
fn dispatch_binary(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    lhs: &wgpu::Buffer,
    rhs: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    numel: u32,
) {
    let params = ElementwiseParams {
        numel,
        scalar: 0.0,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.binary_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: lhs.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: rhs.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dst.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((numel + 63) / 64, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

pub fn add(ctx: &GpuContext, lhs: &wgpu::Buffer, rhs: &wgpu::Buffer, dst: &wgpu::Buffer, numel: u32) {
    dispatch_binary(ctx, &ctx.pipelines.add_pipeline, lhs, rhs, dst, numel);
}

pub fn sub(ctx: &GpuContext, lhs: &wgpu::Buffer, rhs: &wgpu::Buffer, dst: &wgpu::Buffer, numel: u32) {
    dispatch_binary(ctx, &ctx.pipelines.sub_pipeline, lhs, rhs, dst, numel);
}

pub fn mul(ctx: &GpuContext, lhs: &wgpu::Buffer, rhs: &wgpu::Buffer, dst: &wgpu::Buffer, numel: u32) {
    dispatch_binary(ctx, &ctx.pipelines.mul_pipeline, lhs, rhs, dst, numel);
}

pub fn relu_backward(
    ctx: &GpuContext,
    saved_input: &wgpu::Buffer,
    out_grad: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    numel: u32,
) {
    dispatch_binary(ctx, &ctx.pipelines.relu_bw_pipeline, saved_input, out_grad, dst, numel);
}

// ---------------------------------------------------------------------------
// Unary element-wise ops (relu, scale)
// ---------------------------------------------------------------------------

fn dispatch_unary(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    input: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    numel: u32,
    scalar: f32,
) {
    let params = ElementwiseParams {
        numel,
        scalar,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.unary_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((numel + 63) / 64, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

pub fn relu(ctx: &GpuContext, input: &wgpu::Buffer, dst: &wgpu::Buffer, numel: u32) {
    dispatch_unary(ctx, &ctx.pipelines.relu_pipeline, input, dst, numel, 0.0);
}

pub fn scale(ctx: &GpuContext, input: &wgpu::Buffer, dst: &wgpu::Buffer, numel: u32, scalar: f32) {
    dispatch_unary(ctx, &ctx.pipelines.scale_pipeline, input, dst, numel, scalar);
}

// ---------------------------------------------------------------------------
// Matrix multiply
// ---------------------------------------------------------------------------

pub fn matmul(
    ctx: &GpuContext,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    m: u32,
    k: u32,
    n: u32,
) {
    let params = MatmulParams { m, k, n, _pad: 0 };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("matmul_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.matmul_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dst.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.matmul_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // gid.x → columns (n), gid.y → rows (m)
        pass.dispatch_workgroups((n + 15) / 16, (m + 15) / 16, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

// ---------------------------------------------------------------------------
// Bias ops
// ---------------------------------------------------------------------------

pub fn add_bias(
    ctx: &GpuContext,
    matrix: &wgpu::Buffer,
    bias: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    m: u32,
    n: u32,
) {
    let params = BiasParams { m, n, _pad0: 0, _pad1: 0 };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bias_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.bias_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: matrix.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: bias.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dst.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.add_bias_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((m * n + 63) / 64, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

pub fn sum_rows(
    ctx: &GpuContext,
    matrix: &wgpu::Buffer,
    bias_placeholder: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    m: u32,
    n: u32,
) {
    let params = BiasParams { m, n, _pad0: 0, _pad1: 0 };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("sum_rows_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.bias_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: matrix.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: bias_placeholder.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dst.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.sum_rows_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((n + 63) / 64, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

// ---------------------------------------------------------------------------
// Optimizer ops
// ---------------------------------------------------------------------------

/// SGD step: vel = momentum * vel + grad;  param -= lr * vel.
pub fn sgd_step(
    ctx: &GpuContext,
    grad: &wgpu::Buffer,
    vel: &wgpu::Buffer,
    param: &wgpu::Buffer,
    numel: u32,
    lr: f32,
    momentum: f32,
) {
    let params = SgdParams { lr, momentum, numel, _pad: 0 };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("sgd_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.sgd_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: grad.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: vel.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: param.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.sgd_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((numel + 63) / 64, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

/// Adam step: fused m/v update + bias-corrected weight update.
pub fn adam_step(
    ctx: &GpuContext,
    grad: &wgpu::Buffer,
    m: &wgpu::Buffer,
    v: &wgpu::Buffer,
    param: &wgpu::Buffer,
    numel: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    bc1: f32,
    bc2: f32,
) {
    let params = AdamParams { lr, beta1, beta2, epsilon, bc1, bc2, numel, _pad: 0 };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("adam_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.adam_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: grad.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: m.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: v.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: param.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.adam_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((numel + 63) / 64, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

// ---------------------------------------------------------------------------
// Conv ops (im2col, col2im, channel bias)
// ---------------------------------------------------------------------------

/// im2col dispatch: [C_in, H, W] → [col_height, num_patches].
pub fn im2col_dispatch(
    ctx: &GpuContext,
    input: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    c_in: u32, h: u32, w: u32,
    k: u32, stride: u32, pad: u32,
    out_h: u32, out_w: u32,
) {
    let params = Im2ColParams { c_in, h, w, k, stride, pad, out_h, out_w };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("im2col_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.unary_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });

    let total = c_in * k * k * out_h * out_w;
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.im2col_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((total + 63) / 64, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

/// col2im dispatch: [col_height, num_patches] → [C_in, H, W].
pub fn col2im_dispatch(
    ctx: &GpuContext,
    input: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    c_in: u32, h: u32, w: u32,
    k: u32, stride: u32, pad: u32,
    out_h: u32, out_w: u32,
) {
    let params = Im2ColParams { c_in, h, w, k, stride, pad, out_h, out_w };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("col2im_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.unary_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });

    let total = c_in * h * w;
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.col2im_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((total + 63) / 64, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

/// add_channel_bias: [C, spatial] + [C] → [C, spatial].
pub fn add_channel_bias(
    ctx: &GpuContext,
    src: &wgpu::Buffer,
    bias: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    channels: u32,
    spatial: u32,
) {
    let params = ChannelBiasParams { channels, spatial, _pad0: 0, _pad1: 0 };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("channel_bias_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.bias_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: bias.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dst.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });

    let total = channels * spatial;
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.add_channel_bias_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((total + 63) / 64, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

// ---------------------------------------------------------------------------
// Pool ops
// ---------------------------------------------------------------------------

/// max_pool2d forward: [C, H, W] → values [C, out_h, out_w] + indices [C, out_h, out_w].
pub fn max_pool2d_forward(
    ctx: &GpuContext,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    indices: &wgpu::Buffer,
    channels: u32, h: u32, w: u32,
    k: u32, stride: u32,
    out_h: u32, out_w: u32,
) {
    let params = MaxPool2dParams { channels, h, w, k, stride, out_h, out_w, _pad: 0 };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("pool_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.pool_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: indices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });

    let total = channels * out_h * out_w;
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.max_pool2d_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((total + 63) / 64, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

/// max_pool2d backward: scatter out_grad to argmax positions.
pub fn max_pool2d_backward(
    ctx: &GpuContext,
    out_grad: &wgpu::Buffer,
    indices: &wgpu::Buffer,
    grad_input: &wgpu::Buffer,
    channels: u32, h: u32, w: u32,
    out_h: u32, out_w: u32,
) {
    // Reuse MaxPool2dParams — k/stride not needed in backward but must fill struct.
    let params = MaxPool2dParams { channels, h, w, k: 0, stride: 0, out_h, out_w, _pad: 0 };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("pool_bw_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.pool_bw_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: out_grad.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: indices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: grad_input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });

    let total = channels * out_h * out_w;
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.max_pool2d_bw_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((total + 63) / 64, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

// ---------------------------------------------------------------------------
// Dropout
// ---------------------------------------------------------------------------

/// Dropout forward: generate PCG mask and apply to input.
/// Reuses pool_layout: input(read) + output(rw) + mask(rw) + uniform.
pub fn dropout_forward(
    ctx: &GpuContext,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    mask: &wgpu::Buffer,
    numel: u32,
    seed: u32,
    p: f32,
) {
    let scale = 1.0 / (1.0 - p);
    let p_threshold = (p * u32::MAX as f32) as u32;
    let params = DropoutParams { numel, seed, p_threshold, scale };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("dropout_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.pool_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: mask.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.dropout_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((numel + 63) / 64, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

/// Fused stride-aware dropout: reads directly from a potentially
/// non-contiguous source buffer, applies PCG dropout, writes dense
/// output + mask.  Zero intermediate allocation.
pub fn fused_dropout_forward(
    ctx: &GpuContext,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    mask: &wgpu::Buffer,
    numel: u32,
    seed: u32,
    p: f32,
    ndim: u32,
    offset: u32,
    shape: &[usize],
    strides: &[usize],
    suffix: &[usize],
) {
    let scale = 1.0 / (1.0 - p);
    let p_threshold = (p * u32::MAX as f32) as u32;

    let mut params = FusedDropoutParams {
        numel, seed, p_threshold, scale,
        ndim, offset, _pad0: 0, _pad1: 0,
        shape_lo: [0u32; 4], shape_hi: [0u32; 4],
        strides_lo: [0u32; 4], strides_hi: [0u32; 4],
        suffix_lo: [0u32; 4], suffix_hi: [0u32; 4],
    };
    for i in 0..ndim as usize {
        if i < 4 {
            params.shape_lo[i] = shape[i] as u32;
            params.strides_lo[i] = strides[i] as u32;
            params.suffix_lo[i] = suffix[i] as u32;
        } else {
            params.shape_hi[i - 4] = shape[i] as u32;
            params.strides_hi[i - 4] = strides[i] as u32;
            params.suffix_hi[i - 4] = suffix[i] as u32;
        }
    }

    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("fused_dropout_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.pool_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: mask.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.fused_dropout_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((numel + 63) / 64, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

// ---------------------------------------------------------------------------
// Contiguous copy (strided → dense, on-device)
// ---------------------------------------------------------------------------

/// Copy a strided GPU tensor to a dense contiguous GPU buffer.
///
/// Entirely on-device — no D2H transfer.  Uses the `contiguous_copy_kernel`
/// WGSL shader which decomposes each output index into a multi-index via
/// precomputed suffix products and reads from the strided source.
pub fn contiguous_copy(
    ctx: &GpuContext,
    src: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    numel: u32,
    ndim: u32,
    offset: u32,
    shape: &[usize],
    strides: &[usize],
    suffix: &[usize],
) {
    let mut params = ContiguousParams {
        numel,
        ndim,
        offset,
        _pad: 0,
        shape_lo: [0u32; 4],
        shape_hi: [0u32; 4],
        strides_lo: [0u32; 4],
        strides_hi: [0u32; 4],
        suffix_lo: [0u32; 4],
        suffix_hi: [0u32; 4],
    };
    for i in 0..ndim as usize {
        if i < 4 {
            params.shape_lo[i] = shape[i] as u32;
            params.strides_lo[i] = strides[i] as u32;
            params.suffix_lo[i] = suffix[i] as u32;
        } else {
            params.shape_hi[i - 4] = shape[i] as u32;
            params.strides_hi[i - 4] = strides[i] as u32;
            params.suffix_hi[i - 4] = suffix[i] as u32;
        }
    }

    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("contiguous_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.unary_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.contiguous_copy_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((numel + 63) / 64, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

// ---------------------------------------------------------------------------
// Cross-Entropy Loss (forward + gradient fused, then scalar reduction)
// ---------------------------------------------------------------------------

/// Pass 1: compute per-batch loss + full gradient in one dispatch.
/// One workgroup per batch element.
/// Broadcast-scale: dst[i] = src[i] * scalar_buf[0].
///
/// Reads the scalar from a GPU storage buffer (not a host float),
/// keeping the entire operation on-device.  Reuses `binary_layout`.
pub fn broadcast_scale(
    ctx: &GpuContext,
    scalar_buf: &wgpu::Buffer,
    src: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    numel: u32,
) {
    let params = BroadcastScaleParams { numel, _pad0: 0, _pad1: 0, _pad2: 0 };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("broadcast_scale_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.binary_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: scalar_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dst.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.broadcast_scale_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((numel + 63) / 64, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

/// Fused stride-aware scale: reads from a non-contiguous source, multiplies
/// by a scalar, writes to a dense output.  Zero intermediate VRAM allocation.
pub(crate) fn fused_scale(
    ctx: &GpuContext,
    input: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    params: &FusedScaleParams,
) {
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("fused_scale_params"),
        contents: bytemuck::bytes_of(params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.unary_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.fused_scale_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((params.numel + 63) / 64, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

/// Build a `FusedScaleParams` from tensor metadata.
pub(crate) fn make_fused_scale_params(
    numel: usize, scalar: f32,
    ndim: usize, offset: usize,
    shape: &[usize], strides: &[usize], suffix: &[usize],
) -> FusedScaleParams {
    let mut p = FusedScaleParams {
        numel: numel as u32, scalar,
        _pad0: 0, _pad1: 0,
        ndim: ndim as u32, offset: offset as u32,
        _pad2: 0, _pad3: 0,
        shape_lo: [0; 4], shape_hi: [0; 4],
        strides_lo: [0; 4], strides_hi: [0; 4],
        suffix_lo: [0; 4], suffix_hi: [0; 4],
    };
    for i in 0..ndim {
        if i < 4 {
            p.shape_lo[i] = shape[i] as u32;
            p.strides_lo[i] = strides[i] as u32;
            p.suffix_lo[i] = suffix[i] as u32;
        } else {
            p.shape_hi[i - 4] = shape[i] as u32;
            p.strides_hi[i - 4] = strides[i] as u32;
            p.suffix_hi[i - 4] = suffix[i] as u32;
        }
    }
    p
}

pub fn cross_entropy_forward(
    ctx: &GpuContext,
    logits: &wgpu::Buffer,
    targets: &wgpu::Buffer,
    grad: &wgpu::Buffer,
    loss_per_b: &wgpu::Buffer,
    batch: u32,
    num_classes: u32,
) {
    let params = CrossEntropyParams { batch, num_classes, _pad0: 0, _pad1: 0 };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("ce_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.ce_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: logits.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: targets.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: grad.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: loss_per_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.ce_forward_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // One workgroup per batch element.
        pass.dispatch_workgroups(batch, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

/// Pass 2: reduce per-batch losses to a single scalar.
pub fn reduce_loss(
    ctx: &GpuContext,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    numel: u32,
) {
    let params = ReduceParams { numel, _pad0: 0, _pad1: 0, _pad2: 0 };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("reduce_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.unary_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.ce_reduce_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

// ---------------------------------------------------------------------------
// AdamW optimizer
// ---------------------------------------------------------------------------

/// AdamW step: fused m/v update + decoupled weight decay + gradient step.
pub fn adamw_step(
    ctx: &GpuContext,
    grad: &wgpu::Buffer,
    m: &wgpu::Buffer,
    v: &wgpu::Buffer,
    param: &wgpu::Buffer,
    numel: u32,
    lr: f32, beta1: f32, beta2: f32, epsilon: f32,
    bc1: f32, bc2: f32, weight_decay: f32,
) {
    let params = AdamWParams { lr, beta1, beta2, epsilon, bc1, bc2, weight_decay, numel };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("adamw_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.adam_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: grad.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: m.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: v.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: param.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.adamw_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((numel + 63) / 64, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

// ---------------------------------------------------------------------------
// Broadcast binary ops (GPU dispatch)
// ---------------------------------------------------------------------------

fn dispatch_broadcast_binary(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    a: &wgpu::Buffer, b: &wgpu::Buffer, dst: &wgpu::Buffer,
    params: &BroadcastBinaryParams,
) {
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("broadcast_params"),
        contents: bytemuck::bytes_of(params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.binary_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dst.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((params.numel + 63) / 64, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

pub(crate) fn broadcast_add_gpu(ctx: &GpuContext, a: &wgpu::Buffer, b: &wgpu::Buffer, dst: &wgpu::Buffer, params: &BroadcastBinaryParams) {
    dispatch_broadcast_binary(ctx, &ctx.pipelines.broadcast_add_pipeline, a, b, dst, params);
}
pub(crate) fn broadcast_sub_gpu(ctx: &GpuContext, a: &wgpu::Buffer, b: &wgpu::Buffer, dst: &wgpu::Buffer, params: &BroadcastBinaryParams) {
    dispatch_broadcast_binary(ctx, &ctx.pipelines.broadcast_sub_pipeline, a, b, dst, params);
}
pub(crate) fn broadcast_mul_gpu(ctx: &GpuContext, a: &wgpu::Buffer, b: &wgpu::Buffer, dst: &wgpu::Buffer, params: &BroadcastBinaryParams) {
    dispatch_broadcast_binary(ctx, &ctx.pipelines.broadcast_mul_pipeline, a, b, dst, params);
}

pub(crate) fn make_broadcast_params(
    numel: usize, ndim: usize,
    suffix: &[usize], a_strides: &[usize], b_strides: &[usize],
) -> BroadcastBinaryParams {
    let mut p = BroadcastBinaryParams {
        numel: numel as u32, ndim: ndim as u32, _pad0: 0, _pad1: 0,
        suffix_lo: [0; 4], suffix_hi: [0; 4],
        a_strides_lo: [0; 4], a_strides_hi: [0; 4],
        b_strides_lo: [0; 4], b_strides_hi: [0; 4],
    };
    for i in 0..ndim {
        if i < 4 {
            p.suffix_lo[i] = suffix[i] as u32;
            p.a_strides_lo[i] = a_strides[i] as u32;
            p.b_strides_lo[i] = b_strides[i] as u32;
        } else {
            p.suffix_hi[i - 4] = suffix[i] as u32;
            p.a_strides_hi[i - 4] = a_strides[i] as u32;
            p.b_strides_hi[i - 4] = b_strides[i] as u32;
        }
    }
    p
}

// ---------------------------------------------------------------------------
// Reduce sum (broadcast backward, GPU)
// ---------------------------------------------------------------------------

pub(crate) fn reduce_sum_gpu(
    ctx: &GpuContext,
    input: &wgpu::Buffer, output: &wgpu::Buffer,
    params: &ReduceSumGpuParams,
) {
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("reduce_sum_params"),
        contents: bytemuck::bytes_of(params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.unary_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.reduce_sum_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((params.out_numel + 63) / 64, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

pub(crate) fn make_reduce_sum_params(
    input_shape: &[usize], reduced_dims: &[usize], out_numel: usize,
) -> ReduceSumGpuParams {
    let ndim = input_shape.len();
    let suffix = crate::tensor::broadcast::suffix_products(input_shape);
    let mut out_shape = input_shape.to_vec();
    for &d in reduced_dims { out_shape[d] = 1; }
    let mut out_strides = vec![0usize; ndim];
    let mut s = 1usize;
    for d in (0..ndim).rev() {
        if reduced_dims.contains(&d) { out_strides[d] = 0; }
        else { out_strides[d] = s; s *= out_shape[d]; }
    }
    let mut reduce_extents = vec![1usize; ndim];
    for &d in reduced_dims { reduce_extents[d] = input_shape[d]; }

    let mut p = ReduceSumGpuParams {
        out_numel: out_numel as u32, ndim: ndim as u32, _pad0: 0, _pad1: 0,
        in_shape_lo: [0;4], in_shape_hi: [0;4], in_suffix_lo: [0;4], in_suffix_hi: [0;4],
        out_strides_lo: [0;4], out_strides_hi: [0;4],
        reduce_extents_lo: [0;4], reduce_extents_hi: [0;4],
    };
    for i in 0..ndim {
        if i < 4 {
            p.in_shape_lo[i] = input_shape[i] as u32;
            p.in_suffix_lo[i] = suffix[i] as u32;
            p.out_strides_lo[i] = out_strides[i] as u32;
            p.reduce_extents_lo[i] = reduce_extents[i] as u32;
        } else {
            p.in_shape_hi[i-4] = input_shape[i] as u32;
            p.in_suffix_hi[i-4] = suffix[i] as u32;
            p.out_strides_hi[i-4] = out_strides[i] as u32;
            p.reduce_extents_hi[i-4] = reduce_extents[i] as u32;
        }
    }
    p
}

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

pub(crate) fn layer_norm_forward(
    ctx: &GpuContext,
    input: &wgpu::Buffer, weight: &wgpu::Buffer, bias: &wgpu::Buffer,
    output: &wgpu::Buffer, save: &wgpu::Buffer,
    num_instances: u32, norm_size: u32, epsilon: f32,
) {
    let params = LayerNormParams { num_instances, norm_size, epsilon, _pad: 0 };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("ln_params"), contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.ln_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: weight.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bias.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: output.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: save.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.ln_forward_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_instances, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

/// Compute grad_out * x_hat element-wise on GPU (for grad_weight reduction).
pub(crate) fn layer_norm_grad_weight_product(
    ctx: &GpuContext,
    grad_out: &wgpu::Buffer, input: &wgpu::Buffer, weight_placeholder: &wgpu::Buffer,
    save: &wgpu::Buffer, output: &wgpu::Buffer,
    num_instances: u32, norm_size: u32,
) {
    let params = LayerNormBwParams { num_instances, norm_size, _pad0: 0, _pad1: 0 };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("ln_gw_params"), contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.ln_bw_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: grad_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: weight_placeholder.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: save.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: output.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });
    let total = num_instances * norm_size;
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.ln_grad_weight_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((total + 63) / 64, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

pub(crate) fn layer_norm_backward(
    ctx: &GpuContext,
    grad_out: &wgpu::Buffer, input: &wgpu::Buffer, weight: &wgpu::Buffer,
    save: &wgpu::Buffer, grad_in: &wgpu::Buffer,
    num_instances: u32, norm_size: u32,
) {
    let params = LayerNormBwParams { num_instances, norm_size, _pad0: 0, _pad1: 0 };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("lnbw_params"), contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.ln_bw_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: grad_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: weight.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: save.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: grad_in.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.ln_backward_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_instances, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

pub(crate) fn embedding_forward(
    ctx: &GpuContext,
    indices: &wgpu::Buffer, weight: &wgpu::Buffer, output: &wgpu::Buffer,
    total_lookups: u32, embed_dim: u32,
) {
    let params = EmbeddingParams { total_lookups, embed_dim, _pad0: 0, _pad1: 0 };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("emb_params"), contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.binary_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: indices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: weight.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });
    let total = total_lookups * embed_dim;
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.embedding_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((total + 63) / 64, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

// ---------------------------------------------------------------------------
// Batched MatMul
// ---------------------------------------------------------------------------

pub(crate) fn bmm_dispatch(
    ctx: &GpuContext,
    a: &wgpu::Buffer, b: &wgpu::Buffer, out: &wgpu::Buffer,
    batch: u32, m: u32, k: u32, n: u32,
) {
    let params = BmmParams { batch, m, k, n };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bmm_params"), contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.matmul_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.bmm_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((n + 15) / 16, (m + 15) / 16, batch);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

pub(crate) fn softmax_forward_dispatch(
    ctx: &GpuContext,
    input: &wgpu::Buffer, output: &wgpu::Buffer,
    num_rows: u32, row_size: u32,
) {
    let params = SoftmaxParams { num_rows, row_size, _pad0: 0, _pad1: 0 };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("sm_params"), contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.unary_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.softmax_forward_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_rows, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}

pub(crate) fn softmax_backward_dispatch(
    ctx: &GpuContext,
    saved_out: &wgpu::Buffer, grad_out: &wgpu::Buffer, grad_in: &wgpu::Buffer,
    num_rows: u32, row_size: u32,
) {
    let params = SoftmaxParams { num_rows, row_size, _pad0: 0, _pad1: 0 };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("smbw_params"), contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &ctx.pipelines.binary_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: saved_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: grad_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: grad_in.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
        label: None,
    });
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&ctx.pipelines.softmax_backward_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_rows, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
}
