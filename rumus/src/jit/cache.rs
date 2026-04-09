// SPDX-License-Identifier: Apache-2.0 OR MIT
//! JIT compilation cache: avoids recompiling WGSL shaders on every forward pass.
//!
//! The cache key is a hash of the operation sequence + dtype + element count.
//! Cache hits are O(1) HashMap lookups.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::jit::tracer::{FusedOpTag, FusionBlock};
use crate::tensor::DType;

// ---------------------------------------------------------------------------
// Cache key
// ---------------------------------------------------------------------------

/// Uniquely identifies a fusion block for caching purposes.
///
/// Two blocks with the same key produce identical WGSL and can share a
/// compiled pipeline.
#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct FusionKey {
    ops: Vec<FusedOpTag>,
    numel: usize,
    dtype_tag: u8, // 0=F32, 1=F16
    num_inputs: usize,
    num_outputs: usize,
}

impl FusionKey {
    pub fn from_block(block: &FusionBlock) -> Self {
        Self {
            ops: block.ops.iter().map(|op| op.tag()).collect(),
            numel: block.numel,
            dtype_tag: match block.dtype {
                DType::F32 => 0,
                DType::F16 => 1,
                DType::Q8 { .. } => panic!("Q8 tensors cannot be JIT-fused"),
            },
            num_inputs: block.num_inputs,
            num_outputs: block.num_outputs,
        }
    }
}

// ---------------------------------------------------------------------------
// Cached pipeline
// ---------------------------------------------------------------------------

pub struct CachedPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
    /// Total number of bindings: inputs + outputs + 1 (uniform).
    pub total_bindings: usize,
}

// ---------------------------------------------------------------------------
// JitCache
// ---------------------------------------------------------------------------

/// Global cache of compiled fused pipelines.
pub struct JitCache {
    cache: Mutex<HashMap<FusionKey, Arc<CachedPipeline>>>,
}

impl JitCache {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Look up a cached pipeline, or compile a new one.
    pub fn get_or_compile(
        &self,
        key: &FusionKey,
        block: &FusionBlock,
        device: &wgpu::Device,
    ) -> Arc<CachedPipeline> {
        // Fast path: cache hit.
        {
            let cache = self.cache.lock();
            if let Some(entry) = cache.get(key) {
                return Arc::clone(entry);
            }
        }

        // Slow path: generate WGSL, compile pipeline.
        let wgsl_body = crate::jit::codegen::generate_wgsl(block);
        let wgsl = crate::backend::gpu::context::preprocess_shader(&wgsl_body, block.dtype);

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jit_fused"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        // Build dynamic bind group layout.
        let total_bindings = block.num_inputs + block.num_outputs + 1; // +1 for uniform
        let mut entries = Vec::with_capacity(total_bindings);

        // Inputs: read-only storage.
        for i in 0..block.num_inputs {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }

        // Outputs: read-write storage.
        for i in 0..block.num_outputs {
            let binding = (block.num_inputs + i) as u32;
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }

        // Uniform.
        let uniform_binding = (block.num_inputs + block.num_outputs) as u32;
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: uniform_binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jit_fused_layout"),
            entries: &entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jit_fused_pipeline_layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jit_fused_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("fused_kernel"),
            compilation_options: Default::default(),
            cache: None,
        });

        let entry = Arc::new(CachedPipeline {
            pipeline,
            layout,
            total_bindings,
        });

        // Insert into cache.
        self.cache.lock().insert(key.clone(), Arc::clone(&entry));

        entry
    }
}
