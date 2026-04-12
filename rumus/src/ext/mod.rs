// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Custom ops extension API (the plugin engine).
//!
//! Users implement [`CustomOp`] to inject custom WGSL kernels and
//! [`CustomBackward`] for autograd derivatives, without modifying
//! the framework source.  Compiled pipelines are cached globally.
//!
//! # Example
//!
//! ```ignore
//! use rumus::ext::{CustomOp, custom_forward};
//!
//! struct MyOp;
//! impl CustomOp for MyOp {
//!     fn op_name(&self) -> &str { "my_op" }
//!     fn wgsl_source(&self) -> &str { "..." }
//!     fn entry_point(&self) -> &str { "my_kernel" }
//!     fn num_inputs(&self) -> usize { 1 }
//!     fn output_shape(&self, shapes: &[&[usize]]) -> Vec<usize> {
//!         shapes[0].to_vec()
//!     }
//! }
//!
//! let y = custom_forward(&MyOp, &[&x]);
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::autograd::{
    context, BackwardOp, CustomBackwardOp, TapeEntry, VersionSnapshot,
};
use crate::tensor::{AutogradState, DType, Layout, StorageHandle, Tensor, TensorMeta};

#[cfg(feature = "gpu")]
use crate::backend::gpu::context::{CustomOpKey, GpuContext, STORAGE_USAGE};
#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

// Re-export the backward trait so users only need `use rumus::ext::*`.
pub use crate::autograd::CustomBackward;

// ---------------------------------------------------------------------------
// CustomOp trait
// ---------------------------------------------------------------------------

/// Trait for user-defined GPU operations.
///
/// Implement this to inject custom WGSL kernels into RUMUS.  The
/// framework handles compilation caching, bind group creation, dispatch,
/// and autograd integration.
pub trait CustomOp: Send + Sync {
    /// Unique, stable identifier for pipeline caching.
    fn op_name(&self) -> &str;

    /// WGSL shader source (without `alias scalar` preamble — the framework
    /// prepends it based on the input dtype).
    ///
    /// Bindings must follow the convention:
    /// - `@binding(0..N-1)`: `var<storage, read>` for inputs
    /// - `@binding(N)`: `var<storage, read_write>` for the output
    /// - `@binding(N+1)`: `var<uniform>` for parameters
    fn wgsl_source(&self) -> &str;

    /// Entry point function name in the WGSL shader.
    fn entry_point(&self) -> &str;

    /// Number of input tensors.
    fn num_inputs(&self) -> usize;

    /// Compute the output shape given input shapes.
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Vec<usize>;

    /// Workgroup size `(x, y, z)`.  Default: `(256, 1, 1)`.
    fn workgroup_size(&self) -> (u32, u32, u32) {
        (256, 1, 1)
    }

    /// Dispatch dimensions given the output element count.
    /// Default: 1D grid covering all elements.
    fn dispatch(&self, output_numel: usize) -> (u32, u32, u32) {
        let wg = self.workgroup_size().0;
        ((output_numel as u32 + wg - 1) / wg, 1, 1)
    }

    /// Uniform data bytes (must be a multiple of 16 bytes).
    /// Default: `[numel as u32, 0, 0, 0]` (16 bytes).
    fn uniform_data(&self, inputs: &[&Tensor]) -> Vec<u8> {
        let numel = inputs.iter().map(|t| t.numel()).max().unwrap_or(0) as u32;
        let mut buf = vec![0u8; 16];
        buf[0..4].copy_from_slice(&numel.to_le_bytes());
        buf
    }

    /// Optional backward handler.  Return `None` for inference-only ops
    /// (the output will be untracked).
    fn backward_handler(&self) -> Option<Arc<dyn CustomBackward>> {
        None
    }

    /// Which tensors to save for backward.
    /// Default: save all inputs.
    fn save_for_backward<'a>(
        &self,
        inputs: &[&'a Tensor],
        _output: &'a Tensor,
    ) -> Vec<&'a Tensor> {
        inputs.iter().copied().collect()
    }
}

// ---------------------------------------------------------------------------
// custom_forward — the forward dispatcher
// ---------------------------------------------------------------------------

/// Execute a custom GPU operation.
///
/// Compiles the WGSL shader (cached after first call), allocates the
/// output buffer, dispatches the kernel, and optionally records a
/// `BackwardOp::Custom` on the autograd tape.
#[cfg(feature = "gpu")]
pub fn custom_forward(op: &dyn CustomOp, inputs: &[&Tensor]) -> Tensor {
    assert_eq!(
        inputs.len(),
        op.num_inputs(),
        "custom_forward: expected {} inputs, got {}",
        op.num_inputs(),
        inputs.len(),
    );
    assert!(!inputs.is_empty(), "custom_forward: need at least one input");

    let dtype = inputs[0].dtype();
    let input_shapes: Vec<&[usize]> = inputs.iter().map(|t| t.shape()).collect();
    let output_shape = op.output_shape(&input_shapes);
    let out_numel: usize = output_shape.iter().product();

    let ctx = GpuContext::get().expect("GPU required for custom_forward");

    // --- Cache lookup / compile ---
    let key = CustomOpKey {
        op_name: op.op_name().to_string(),
        dtype_tag: match dtype {
            DType::F32 => 0,
            DType::F16 => 1,
            DType::Q8 { .. } => panic!("custom_forward: Q8 inputs not supported"),
        },
        num_inputs: op.num_inputs(),
    };

    let wgsl = crate::backend::gpu::context::preprocess_shader(op.wgsl_source(), dtype);
    let cached = ctx.custom_ops.get_or_compile(
        &key,
        &ctx.device,
        &wgsl,
        op.entry_point(),
        op.num_inputs(),
    );

    // --- Allocate output buffer ---
    let out_buf = ctx.pool.acquire(&ctx.device, dtype.gpu_buf_size(out_numel), STORAGE_USAGE);

    // --- Build uniform buffer ---
    let uniform_bytes = op.uniform_data(inputs);
    let uniform_buf = ctx.device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("custom_op_uniform"),
            contents: &uniform_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        },
    );

    // --- Build bind group ---
    let n = op.num_inputs();
    let input_guards: Vec<_> = inputs.iter().map(|t| {
        t.storage.ensure_gpu();
        t.storage.gpu_buffer()
    }).collect();

    let mut bg_entries: Vec<wgpu::BindGroupEntry<'_>> = Vec::with_capacity(n + 2);
    for (i, guard) in input_guards.iter().enumerate() {
        bg_entries.push(wgpu::BindGroupEntry {
            binding: i as u32,
            resource: guard.as_entire_binding(),
        });
    }
    bg_entries.push(wgpu::BindGroupEntry {
        binding: n as u32,
        resource: out_buf.as_entire_binding(),
    });
    bg_entries.push(wgpu::BindGroupEntry {
        binding: (n + 1) as u32,
        resource: uniform_buf.as_entire_binding(),
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &cached.layout,
        entries: &bg_entries,
        label: Some("custom_op_bg"),
    });

    // --- Dispatch ---
    let (dx, dy, dz) = op.dispatch(out_numel);
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&cached.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dx, dy, dz);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));
    drop(input_guards);

    // --- Build output tensor ---
    let out_storage = StorageHandle::new_gpu(out_buf, out_numel);
    let out_layout = Layout::contiguous(output_shape.clone());

    let output = Tensor::from_storage_and_layout(out_storage.clone(), out_layout.clone());

    // --- Autograd recording ---
    let has_bw = op.backward_handler();
    let any_grad = inputs.iter().any(|t| t.requires_grad()) && !context::is_no_grad();

    if let (Some(handler), true) = (has_bw, any_grad) {
        let out_grad_id = context::next_grad_id();

        let mut input_gids = Vec::with_capacity(n);
        let mut input_versions = Vec::with_capacity(n);
        for &inp in inputs {
            let gid = inp.grad_id().unwrap_or_else(context::next_grad_id);
            input_versions.push(VersionSnapshot::new(gid, &inp.storage));
            if let Some(m) = inp.meta() {
                m.total_grads.fetch_add(1, Ordering::Relaxed);
            }
            input_gids.push(gid);
        }

        // Save tensors for backward.
        let to_save = op.save_for_backward(inputs, &output);
        let saved_storages: Vec<StorageHandle> = to_save.iter().map(|t| t.storage.clone()).collect();
        let saved_layouts: Vec<Layout> = to_save.iter().map(|t| t.layout.clone()).collect();
        let saved_shapes: Vec<Vec<usize>> = to_save.iter().map(|t| t.shape().to_vec()).collect();

        let op_id = context::with_tape(|tape| {
            tape.push(TapeEntry {
                op: BackwardOp::Custom(CustomBackwardOp {
                    handler,
                    input_versions,
                    saved_storages,
                    saved_layouts,
                    saved_shapes,
                }),
                inputs: input_gids,
                outputs: vec![out_grad_id],
            })
        });

        Tensor {
            storage: out_storage,
            layout: Layout::contiguous(output_shape),
            state: AutogradState::Tracked(Arc::new(TensorMeta {
                requires_grad: true,
                grad_id: Some(out_grad_id),
                creator: op_id,
                is_leaf: false,
                retains_grad: false,
                total_grads: AtomicUsize::new(0),
            })),
        }
    } else {
        output
    }
}
