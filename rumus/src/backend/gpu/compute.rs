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

// WebGPU requires uniform buffer bindings to be a multiple of 16 bytes.
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

const _: () = assert!(std::mem::size_of::<Im2ColParams>() == 32);
const _: () = assert!(std::mem::size_of::<ChannelBiasParams>() == 16);
const _: () = assert!(std::mem::size_of::<MaxPool2dParams>() == 32);

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
