// SPDX-License-Identifier: Apache-2.0 OR MIT
//! JIT compiler for kernel fusion.
//!
//! Fuses element-wise operations (Add, Relu, Scale, etc.) into a single
//! dynamically generated WGSL kernel.  Autograd tape recording proceeds
//! normally — only the forward GPU dispatch is intercepted.
//!
//! Feature-gated behind `--features jit`.

pub mod cache;
pub mod codegen;
pub mod tracer;

pub use tracer::compile;

use std::sync::OnceLock;

use crate::backend::gpu::context::{GpuContext, STORAGE_USAGE};
use crate::jit::cache::{FusionKey, JitCache};
use crate::jit::tracer::FusionBlock;

use wgpu::util::{BufferInitDescriptor, DeviceExt};

static JIT_CACHE: OnceLock<JitCache> = OnceLock::new();

fn get_cache() -> &'static JitCache {
    JIT_CACHE.get_or_init(JitCache::new)
}

/// Uniform params for fused kernels: just numel.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FusedParams {
    numel: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

/// Flush a fusion block: codegen → cache lookup → dispatch → materialize outputs.
pub(crate) fn flush_block(block: FusionBlock) {
    if block.ops.is_empty() || block.num_outputs == 0 {
        return;
    }

    let ctx = GpuContext::get().expect("GPU required for JIT fusion");
    let cache = get_cache();

    // Build cache key and look up / compile.
    let key = FusionKey::from_block(&block);
    let cached = cache.get_or_compile(&key, &block, &ctx.device);

    // Allocate output GPU buffers.
    let mut output_buffers = Vec::with_capacity(block.num_outputs);
    for _ in 0..block.num_outputs {
        let buf = ctx.pool.acquire(
            &ctx.device,
            block.dtype.gpu_buf_size(block.numel),
            STORAGE_USAGE,
        );
        output_buffers.push(buf);
    }

    // Build uniform buffer.
    let params = FusedParams {
        numel: block.numel as u32,
        _p0: 0, _p1: 0, _p2: 0,
    };
    let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("jit_fused_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Build bind group entries.
    let mut bg_entries: Vec<wgpu::BindGroupEntry<'_>> = Vec::with_capacity(cached.total_bindings);

    // Input bindings.
    let input_guards: Vec<_> = block
        .input_storages
        .iter()
        .map(|s| {
            s.ensure_gpu();
            s.gpu_buffer()
        })
        .collect();
    for (i, guard) in input_guards.iter().enumerate() {
        bg_entries.push(wgpu::BindGroupEntry {
            binding: i as u32,
            resource: guard.as_entire_binding(),
        });
    }

    // Output bindings.
    for (i, buf) in output_buffers.iter().enumerate() {
        let binding = (block.num_inputs + i) as u32;
        bg_entries.push(wgpu::BindGroupEntry {
            binding,
            resource: buf.as_entire_binding(),
        });
    }

    // Uniform binding.
    let uniform_binding = (block.num_inputs + block.num_outputs) as u32;
    bg_entries.push(wgpu::BindGroupEntry {
        binding: uniform_binding,
        resource: params_buf.as_entire_binding(),
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &cached.layout,
        entries: &bg_entries,
        label: Some("jit_fused_bg"),
    });

    // Dispatch.
    let workgroups = (block.numel as u32 + 255) / 256;
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&cached.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));

    // Drop input guards before materializing outputs (release read locks).
    drop(input_guards);

    // Materialize deferred output storages with real GPU buffers.
    for (storage, buffer) in block.output_storages.iter().zip(output_buffers.into_iter()) {
        storage.materialize_gpu(buffer);
    }
}
