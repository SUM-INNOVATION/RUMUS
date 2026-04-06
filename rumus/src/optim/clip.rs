// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Gradient clipping utilities.
//!
//! `clip_grad_norm_` computes the global L2 norm of all parameter gradients
//! and scales them down if the norm exceeds `max_norm`.  Follows a 3-pass
//! strategy to avoid interleaving GPU dispatches and CPU readbacks:
//!
//! - **Pass 1:** Dispatch `reduce_sum_sq` for all GPU-resident gradients.
//! - **Pass 2:** Read back per-parameter squared norms, sum to global norm.
//! - **Pass 3:** If clipping needed, scale all gradients.

use crate::autograd::GradientStore;
use crate::backend::{Backend, CpuBackend};
use crate::nn::Parameter;
use crate::tensor::{GradId, Tensor};

#[cfg(feature = "gpu")]
use crate::tensor::{Layout, StorageHandle};

/// Clip the global L2 norm of parameter gradients.
///
/// If the total norm exceeds `max_norm`, all gradients are uniformly
/// scaled down so their combined norm equals `max_norm`.
///
/// Returns the total (unclipped) norm — useful for logging.
///
/// # Arguments
///
/// - `grads`: mutable gradient store (gradients may be replaced).
/// - `params`: the model's parameters (provides `GradId` mapping).
/// - `max_norm`: maximum allowed global L2 norm.
pub fn clip_grad_norm_(
    grads: &mut GradientStore,
    params: &[Parameter],
    max_norm: f32,
) -> f32 {
    assert!(max_norm > 0.0, "clip_grad_norm_: max_norm must be > 0");

    // Collect (grad_id, numel, is_gpu) for all parameters that have gradients.
    let mut grad_infos: Vec<(GradId, bool)> = Vec::new();
    for param in params {
        let gid = param.grad_id();
        if grads.get(gid).is_some() {
            let is_gpu;
            #[cfg(feature = "gpu")]
            {
                is_gpu = grads.get(gid).unwrap().storage.is_gpu();
            }
            #[cfg(not(feature = "gpu"))]
            {
                is_gpu = false;
            }
            grad_infos.push((gid, is_gpu));
        }
    }

    if grad_infos.is_empty() {
        return 0.0;
    }

    // -----------------------------------------------------------------------
    // Pass 1: Dispatch reduce_sum_sq for all GPU gradients (non-blocking).
    //         Compute squared norms for CPU gradients immediately.
    // -----------------------------------------------------------------------
    let mut cpu_sq_sum: f32 = 0.0;

    #[cfg(feature = "gpu")]
    let mut gpu_norm_tensors: Vec<Tensor> = Vec::new();

    #[cfg(feature = "gpu")]
    {
        use crate::backend::gpu::{
            compute as gpu_compute,
            context::{GpuContext, STORAGE_USAGE},
        };

        for &(gid, is_gpu) in &grad_infos {
            let grad = grads.get(gid).unwrap();
            if is_gpu {
                let ctx = GpuContext::get().expect("GPU required");
                let gc = grad.contiguous();
                let gb = gc.storage.gpu_buffer();
                let numel = grad.numel();
                let dst_buf = ctx.pool.acquire(&ctx.device, 4u64, STORAGE_USAGE);
                gpu_compute::reduce_sum_sq(ctx, &gb, &dst_buf, numel as u32);
                drop(gb);
                let norm_t = Tensor::from_storage_and_layout(
                    StorageHandle::new_gpu(dst_buf, 1),
                    Layout::contiguous(vec![1]),
                );
                gpu_norm_tensors.push(norm_t);
            } else {
                let gc = grad.contiguous();
                let g = gc.storage.data();
                let sq: f32 = g.iter().map(|&x| x * x).sum();
                drop(g);
                cpu_sq_sum += sq;
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        for &(gid, _) in &grad_infos {
            let grad = grads.get(gid).unwrap();
            let gc = grad.contiguous();
            let g = gc.storage.data();
            let sq: f32 = g.iter().map(|&x| x * x).sum();
            drop(g);
            cpu_sq_sum += sq;
        }
    }

    // -----------------------------------------------------------------------
    // Pass 2: Read back GPU squared norms (triggers device sync) and compute
    //         the global norm.
    // -----------------------------------------------------------------------
    #[cfg(feature = "gpu")]
    {
        for norm_t in &gpu_norm_tensors {
            let g = norm_t.storage.data(); // triggers D2H if needed
            cpu_sq_sum += g[0];
            drop(g);
        }
    }

    let total_norm = cpu_sq_sum.sqrt();

    // -----------------------------------------------------------------------
    // Pass 3: If clipping needed, scale all gradients.
    // -----------------------------------------------------------------------
    let clip_coef = max_norm / (total_norm + 1e-6);
    if clip_coef >= 1.0 {
        return total_norm;
    }

    for &(gid, _is_gpu) in &grad_infos {
        let grad = grads.remove(gid).unwrap();

        #[cfg(feature = "gpu")]
        let scaled = if _is_gpu {
            use crate::backend::gpu::{
                compute as gpu_compute,
                context::{GpuContext, STORAGE_USAGE},
            };

            let ctx = GpuContext::get().expect("GPU required");
            let gc = grad.contiguous();
            let numel = gc.numel();
            let gb = gc.storage.gpu_buffer();

            // Write clip_coef to a 1-element GPU buffer for broadcast_scale.
            let scalar_buf = ctx.pool.acquire(&ctx.device, 4u64, STORAGE_USAGE);
            ctx.queue.write_buffer(&scalar_buf, 0, bytemuck::bytes_of(&clip_coef));

            let dst_buf = ctx.pool.acquire(&ctx.device, (numel * 4) as u64, STORAGE_USAGE);
            gpu_compute::broadcast_scale(ctx, &scalar_buf, &gb, &dst_buf, numel as u32);
            drop(gb);
            ctx.pool.release(scalar_buf);

            Tensor::from_storage_and_layout(
                StorageHandle::new_gpu(dst_buf, numel),
                Layout::contiguous(gc.shape().to_vec()),
            )
        } else {
            let gc = grad.contiguous();
            let g = gc.storage.data();
            let numel = gc.numel();
            let mut dst = CpuBackend::zeros(numel);
            CpuBackend::scale(&g, &mut dst, clip_coef);
            drop(g);
            Tensor::new(dst, gc.shape().to_vec())
        };

        #[cfg(not(feature = "gpu"))]
        let scaled = {
            let gc = grad.contiguous();
            let g = gc.storage.data();
            let numel = gc.numel();
            let mut dst = CpuBackend::zeros(numel);
            CpuBackend::scale(&g, &mut dst, clip_coef);
            drop(g);
            Tensor::new(dst, gc.shape().to_vec())
        };

        grads.replace(gid, scaled);
    }

    total_norm
}
