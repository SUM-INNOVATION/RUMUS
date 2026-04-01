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

// WebGPU requires uniform buffer bindings to be a multiple of 16 bytes.
const _: () = assert!(std::mem::size_of::<ElementwiseParams>() == 16);
const _: () = assert!(std::mem::size_of::<MatmulParams>() == 16);
const _: () = assert!(std::mem::size_of::<BiasParams>() == 16);
const _: () = assert!(std::mem::size_of::<SgdParams>() == 16);
const _: () = assert!(std::mem::size_of::<AdamParams>() == 32);
// 32 is a multiple of 16 ✓

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
