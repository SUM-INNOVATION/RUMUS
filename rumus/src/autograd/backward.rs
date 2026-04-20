// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Kahn's-algorithm backward traversal of the computational graph.

use std::collections::HashMap;

use crate::autograd::backward_ops::BackwardOp;
use crate::autograd::context;
use crate::autograd::{AutogradError, GradientStore};
use crate::backend::{Backend, CpuBackend};
use crate::tensor::{GradId, Tensor};

/// If broadcasting occurred, reduce (sum) the gradient along the broadcast dims
/// to match the operand's original shape.  If no broadcasting, return as-is.
/// Routes to GPU reduce_sum kernel when grad is GPU-resident.
fn reduce_if_broadcast(
    grad: &Tensor,
    info: &Option<crate::tensor::broadcast::BroadcastInfo>,
) -> Tensor {
    match info {
        None => grad.clone(),
        Some(bi) => {
            let out_numel: usize = bi.original_shape.iter().product();

            #[cfg(feature = "gpu")]
            if grad.storage.is_gpu() {
                use crate::backend::gpu::{
                    compute as gpu_compute,
                    context::{GpuContext, STORAGE_USAGE},
                };
                let ctx = GpuContext::get().expect("GPU required");
                let gc = grad.contiguous();
                let gb = gc.storage.gpu_buffer();
                let dst_buf = ctx.pool.acquire(&ctx.device, crate::tensor::DType::F32.gpu_buf_size(out_numel), STORAGE_USAGE);
                // Zero the output buffer before accumulation.
                {
                    let mut enc = ctx.device.create_command_encoder(&Default::default());
                    enc.clear_buffer(&dst_buf, 0, None);
                    ctx.queue.submit(std::iter::once(enc.finish()));
                }
                let params = gpu_compute::make_reduce_sum_params(
                    grad.shape(), &bi.reduced_dims, out_numel,
                );
                gpu_compute::reduce_sum_gpu(ctx, &gb, &dst_buf, &params);
                drop(gb);
                return Tensor::from_storage_and_layout(
                    crate::tensor::StorageHandle::new_gpu(dst_buf, out_numel),
                    crate::tensor::Layout::contiguous(bi.original_shape.clone()),
                );
            }

            // CPU path.
            let gc = grad.contiguous();
            let guard = gc.storage.data();
            let mut dst = CpuBackend::zeros(out_numel);
            crate::tensor::broadcast::cpu_reduce_sum(
                &guard, &mut dst, grad.shape(), &bi.reduced_dims,
            );
            drop(guard);
            Tensor::new(dst, bi.original_shape.clone())
        }
    }
}

/// Negate a tensor on-device (GPU) or host (CPU).
///
/// GPU path: dispatches the fused stride-aware `fused_scale_kernel` with
/// `scalar = -1.0`.  Reads directly from the (potentially non-contiguous)
/// source buffer using strides — zero intermediate VRAM allocation, no
/// `.contiguous()` call.
/// CPU path: makes contiguous then `CpuBackend::scale`.
fn negate_tensor(t: &Tensor) -> Tensor {
    let numel = t.numel();
    let shape = t.shape().to_vec();

    #[cfg(feature = "gpu")]
    if t.storage.is_gpu() {
        use crate::backend::gpu::{
            compute as gpu_compute,
            context::{GpuContext, STORAGE_USAGE},
        };
        let ctx = GpuContext::get().expect("GPU required");
        let tb = t.storage.gpu_buffer();
        let dst_buf = ctx.pool.acquire(&ctx.device, crate::tensor::DType::F32.gpu_buf_size(numel), STORAGE_USAGE);

        let strides = t.strides();
        let offset = t.layout.offset();
        let ndim = t.ndim();
        let suffix = crate::tensor::broadcast::suffix_products(&shape);

        let params = gpu_compute::make_fused_scale_params(
            numel, -1.0, ndim, offset, &shape, strides, &suffix,
        );
        gpu_compute::fused_scale(ctx, &tb, &dst_buf, &params);
        drop(tb);
        return Tensor::from_storage_and_layout(
            crate::tensor::StorageHandle::new_gpu(dst_buf, numel),
            crate::tensor::Layout::contiguous(shape),
        );
    }

    let tc = t.contiguous();
    let guard = tc.storage.data();
    let mut dst = CpuBackend::zeros(numel);
    CpuBackend::scale(&guard, &mut dst, -1.0);
    drop(guard);
    Tensor::new(dst, shape)
}

/// Execute the backward pass from `tensor`, returning accumulated gradients.
///
/// See module-level docs for the full Kahn's algorithm description.
pub fn backward(tensor: &Tensor) -> Result<GradientStore, AutogradError> {
    assert_eq!(
        tensor.numel(),
        1,
        "backward() requires a scalar tensor (numel == 1), got numel == {}",
        tensor.numel(),
    );

    let seed = Tensor::new(vec![1.0f32], tensor.shape().to_vec());
    #[cfg(feature = "gpu")]
    if tensor.storage.is_gpu() {
        seed.to_gpu();
    }

    backward_with_grad(tensor, seed)
}

/// Execute the backward pass with an externally provided gradient seed.
///
/// Like [`backward`], but instead of seeding with `[1.0]`, the caller
/// provides the gradient tensor.  The tensor does NOT need to be scalar.
///
/// Used by pipeline parallelism for cross-stage gradient injection.
pub fn backward_with_grad(
    tensor: &Tensor,
    grad_output: Tensor,
) -> Result<GradientStore, AutogradError> {
    let root_grad_id = tensor.grad_id().ok_or(AutogradError::NoGraph)?;

    let tape = context::take_tape().ok_or(AutogradError::NoGraph)?;
    let entries = tape.into_entries();

    let mut grads = GradientStore::new();
    grads.accumulate(root_grad_id, grad_output)?;

    let mut pending: HashMap<GradId, usize> = HashMap::new();
    for entry in &entries {
        for &input_id in &entry.inputs {
            *pending.entry(input_id).or_insert(0) += 1;
        }
    }

    for entry in entries.into_iter().rev() {
        let out_grad_id = entry.outputs[0];

        let out_pending = pending.get(&out_grad_id).copied().unwrap_or(0);
        if out_pending != 0 {
            continue;
        }

        let out_grad = match grads.get(out_grad_id) {
            Some(g) => g.clone(),
            None => {
                for &input_id in &entry.inputs {
                    if let Some(count) = pending.get_mut(&input_id) {
                        *count -= 1;
                    }
                }
                continue;
            }
        };

        match &entry.op {
            BackwardOp::Add(bw) => {
                bw.lhs_version.check()?;
                bw.rhs_version.check()?;
                grads.accumulate(entry.inputs[0], out_grad.clone())?;
                grads.accumulate(entry.inputs[1], out_grad)?;
            }

            BackwardOp::Sub(bw) => {
                bw.lhs_version.check()?;
                bw.rhs_version.check()?;
                grads.accumulate(entry.inputs[0], out_grad.clone())?;
                let grad_rhs = negate_tensor(&out_grad);
                grads.accumulate(entry.inputs[1], grad_rhs)?;
            }

            BackwardOp::Mul(bw) => {
                bw.lhs_version.check()?;
                bw.rhs_version.check()?;
                let saved_lhs = Tensor::from_storage_and_layout(
                    bw.lhs_storage.clone(),
                    bw.lhs_layout.clone(),
                );
                let saved_rhs = Tensor::from_storage_and_layout(
                    bw.rhs_storage.clone(),
                    bw.rhs_layout.clone(),
                );
                let grad_lhs = out_grad.mul(&saved_rhs);
                let grad_rhs = out_grad.mul(&saved_lhs);
                grads.accumulate(entry.inputs[0], grad_lhs)?;
                grads.accumulate(entry.inputs[1], grad_rhs)?;
            }

            BackwardOp::Matmul(bw) => {
                bw.lhs_version.check()?;
                bw.rhs_version.check()?;
                let saved_a = Tensor::from_storage_and_layout(
                    bw.lhs_storage.clone(),
                    bw.lhs_layout.clone(),
                );
                let saved_b = Tensor::from_storage_and_layout(
                    bw.rhs_storage.clone(),
                    bw.rhs_layout.clone(),
                );
                let b_t = saved_b.transpose(0, 1);
                let grad_lhs = out_grad.matmul(&b_t);
                let a_t = saved_a.transpose(0, 1);
                let grad_rhs = a_t.matmul(&out_grad);
                grads.accumulate(entry.inputs[0], grad_lhs)?;
                grads.accumulate(entry.inputs[1], grad_rhs)?;
            }

            BackwardOp::Relu(bw) => {
                bw.input_version.check()?;
                let saved_input = Tensor::from_storage_and_layout(
                    bw.input_storage.clone(),
                    bw.input_layout.clone(),
                );
                let si_c = saved_input.contiguous();
                let og_c = out_grad.contiguous();
                let si_guard = si_c.storage.data();
                let og_guard = og_c.storage.data();
                let mut dst = CpuBackend::zeros(out_grad.numel());
                CpuBackend::relu_backward(&si_guard, &og_guard, &mut dst);
                drop(si_guard);
                drop(og_guard);
                let grad_input = Tensor::new(dst, out_grad.shape().to_vec());
                grads.accumulate(entry.inputs[0], grad_input)?;
            }

            BackwardOp::MseLoss(bw) => {
                bw.pred_version.check()?;
                bw.target_version.check()?;
                let saved_pred = Tensor::from_storage_and_layout(
                    bw.pred_storage.clone(),
                    bw.pred_layout.clone(),
                );
                let saved_target = Tensor::from_storage_and_layout(
                    bw.target_storage.clone(),
                    bw.target_layout.clone(),
                );
                let pred_c = saved_pred.contiguous();
                let targ_c = saved_target.contiguous();
                let pred_guard = pred_c.storage.data();
                let targ_guard = targ_c.storage.data();
                let og_guard = out_grad.storage.data();
                let og_scalar = og_guard[0];
                drop(og_guard);

                let numel = bw.numel;
                let scale = og_scalar * 2.0 / numel as f32;
                let mut dst = vec![0.0f32; numel];
                for i in 0..numel {
                    dst[i] = scale * (pred_guard[i] - targ_guard[i]);
                }
                drop(pred_guard);
                drop(targ_guard);
                let grad_pred = Tensor::new(dst, saved_pred.shape().to_vec());
                grads.accumulate(entry.inputs[0], grad_pred)?;
            }

            BackwardOp::AddBias(bw) => {
                bw.input_version.check()?;
                bw.bias_version.check()?;
                let (m, n) = (bw.m, bw.n);
                grads.accumulate(entry.inputs[0], out_grad.clone())?;
                let og_c = out_grad.contiguous();
                let og_guard = og_c.storage.data();
                let mut bias_grad = CpuBackend::zeros(n);
                CpuBackend::sum_rows(&og_guard, &mut bias_grad, m, n);
                drop(og_guard);
                let grad_bias = Tensor::new(bias_grad, vec![n]);
                grads.accumulate(entry.inputs[1], grad_bias)?;
            }

            BackwardOp::Im2Col(bw) => {
                bw.input_version.check()?;
                // ∂L/∂input = col2im(∂L/∂output)
                let og_c = out_grad.contiguous();
                let og_guard = og_c.storage.data();
                let input_numel = bw.c_in * bw.h * bw.w;
                let mut dst = CpuBackend::zeros(input_numel);
                CpuBackend::col2im(
                    &og_guard, &mut dst,
                    bw.c_in, bw.h, bw.w,
                    bw.kernel_size, bw.stride, bw.padding,
                    bw.out_h, bw.out_w,
                );
                drop(og_guard);
                let grad_input = Tensor::new(dst, vec![bw.c_in, bw.h, bw.w]);
                grads.accumulate(entry.inputs[0], grad_input)?;
            }

            BackwardOp::Stack(bw) => {
                for v in &bw.versions {
                    v.check()?;
                }
                // ∂L/∂t_i = slice out_grad along axis 0 at index i
                let each_numel: usize = bw.each_shape.iter().product();
                let og_c = out_grad.contiguous();
                let og_guard = og_c.storage.data();
                for i in 0..bw.count {
                    let start = i * each_numel;
                    let slice_data = og_guard[start..start + each_numel].to_vec();
                    let grad_i = Tensor::new(slice_data, bw.each_shape.clone());
                    grads.accumulate(entry.inputs[i], grad_i)?;
                }
                drop(og_guard);
            }

            BackwardOp::SliceBatch(bw) => {
                bw.input_version.check()?;
                // ∂L/∂input is a zero tensor with out_grad placed at batch index.
                let total_numel: usize = bw.original_shape.iter().product();
                let batch_size = bw.original_shape[0];
                let element_numel = total_numel / batch_size;
                let mut dst = vec![0.0f32; total_numel];
                let og_c = out_grad.contiguous();
                let og_guard = og_c.storage.data();
                let start = bw.index * element_numel;
                dst[start..start + element_numel].copy_from_slice(&og_guard);
                drop(og_guard);
                let grad_input = Tensor::new(dst, bw.original_shape.clone());
                grads.accumulate(entry.inputs[0], grad_input)?;
            }

            BackwardOp::AddChannelBias(bw) => {
                bw.input_version.check()?;
                bw.bias_version.check()?;
                grads.accumulate(entry.inputs[0], out_grad.clone())?;
                let og_c = out_grad.contiguous();
                let og_guard = og_c.storage.data();
                let mut bias_grad = CpuBackend::zeros(bw.channels);
                CpuBackend::sum_channel_bias_grad(
                    &og_guard, &mut bias_grad,
                    bw.channels, bw.spatial,
                );
                drop(og_guard);
                let grad_bias = Tensor::new(bias_grad, vec![bw.channels]);
                grads.accumulate(entry.inputs[1], grad_bias)?;
            }

            BackwardOp::MaxPool2d(bw) => {
                bw.input_version.check()?;
                let indices = Tensor::from_storage_and_layout(
                    bw.indices_storage.clone(),
                    bw.indices_layout.clone(),
                );
                let idx_c = indices.contiguous();
                let og_c = out_grad.contiguous();
                let idx_guard = idx_c.storage.data();
                let og_guard = og_c.storage.data();
                let input_numel = bw.channels * bw.h * bw.w;
                let mut dst = CpuBackend::zeros(input_numel);
                CpuBackend::max_pool2d_backward(
                    &og_guard, &idx_guard, &mut dst,
                    bw.channels, bw.h, bw.w,
                    bw.out_h, bw.out_w,
                );
                drop(idx_guard);
                drop(og_guard);
                let grad_input = Tensor::new(dst, vec![bw.channels, bw.h, bw.w]);
                grads.accumulate(entry.inputs[0], grad_input)?;
            }

            BackwardOp::Flatten(bw) => {
                bw.input_version.check()?;
                let grad_input = out_grad.reshape(bw.original_shape.clone());
                grads.accumulate(entry.inputs[0], grad_input)?;
            }

            BackwardOp::Reshape(bw) => {
                bw.input_version.check()?;
                let grad_input = out_grad.reshape(bw.original_shape.clone());
                grads.accumulate(entry.inputs[0], grad_input)?;
            }

            BackwardOp::CrossEntropy(bw) => {
                bw.input_version.check()?;
                // Gradient was pre-computed during forward as (softmax - one_hot) / B.
                // Scale by the incoming out_grad scalar — entirely on-device.
                let saved_grad = Tensor::from_storage_and_layout(
                    bw.grad_storage.clone(),
                    bw.grad_layout.clone(),
                );
                // out_grad is [1], saved_grad is [B, C].
                // Broadcast-scale on GPU; CPU fallback for CPU tensors.
                #[cfg(feature = "gpu")]
                let grad_input = if saved_grad.storage.is_gpu() {
                    use crate::backend::gpu::{
                        compute as gpu_compute,
                        context::{GpuContext, STORAGE_USAGE},
                    };
                    let ctx = GpuContext::get().expect("GPU required");
                    let sg_c = saved_grad.contiguous();
                    let og_c = out_grad.contiguous();
                    let sg_buf = sg_c.storage.gpu_buffer();
                    let og_buf = og_c.storage.gpu_buffer();
                    let numel = saved_grad.numel();
                    let dst_buf = ctx.pool.acquire(
                        &ctx.device, crate::tensor::DType::F32.gpu_buf_size(numel), STORAGE_USAGE,
                    );
                    gpu_compute::broadcast_scale(
                        ctx, &og_buf, &sg_buf, &dst_buf, numel as u32,
                    );
                    drop(sg_buf);
                    drop(og_buf);
                    Tensor::from_storage_and_layout(
                        crate::tensor::StorageHandle::new_gpu(dst_buf, numel),
                        crate::tensor::Layout::contiguous(saved_grad.shape().to_vec()),
                    )
                } else {
                    // CPU path: read scalar, scale saved_grad.
                    let og_guard = out_grad.storage.data();
                    let scalar = og_guard[0];
                    drop(og_guard);
                    let sg_c = saved_grad.contiguous();
                    let sg_guard = sg_c.storage.data();
                    let mut dst = CpuBackend::zeros(saved_grad.numel());
                    CpuBackend::scale(&sg_guard, &mut dst, scalar);
                    drop(sg_guard);
                    Tensor::new(dst, saved_grad.shape().to_vec())
                };
                #[cfg(not(feature = "gpu"))]
                let grad_input = {
                    let og_guard = out_grad.storage.data();
                    let scalar = og_guard[0];
                    drop(og_guard);
                    let sg_c = saved_grad.contiguous();
                    let sg_guard = sg_c.storage.data();
                    let mut dst = CpuBackend::zeros(saved_grad.numel());
                    CpuBackend::scale(&sg_guard, &mut dst, scalar);
                    drop(sg_guard);
                    Tensor::new(dst, saved_grad.shape().to_vec())
                };
                grads.accumulate(entry.inputs[0], grad_input)?;
            }

            BackwardOp::Dropout(bw) => {
                bw.input_version.check()?;
                // ∂L/∂input = ∂L/∂output * saved_mask
                // Reuses existing mul dispatch (auto CPU/GPU).
                let saved_mask = Tensor::from_storage_and_layout(
                    bw.mask_storage.clone(),
                    bw.mask_layout.clone(),
                );
                let grad_input = out_grad.mul(&saved_mask);
                grads.accumulate(entry.inputs[0], grad_input)?;
            }

            BackwardOp::Transpose(bw) => {
                bw.input_version.check()?;
                // Reverse the transpose: transpose(grad, dim0, dim1).
                // Uses untracked transpose (view op) — correct since we're
                // inside the backward engine, not building a new graph.
                let grad_input = out_grad.transpose(bw.dim0, bw.dim1);
                grads.accumulate(entry.inputs[0], grad_input)?;
            }

            BackwardOp::Bmm(bw) => {
                bw.lhs_version.check()?;
                bw.rhs_version.check()?;
                let saved_a = Tensor::from_storage_and_layout(bw.lhs_storage.clone(), bw.lhs_layout.clone());
                let saved_b = Tensor::from_storage_and_layout(bw.rhs_storage.clone(), bw.rhs_layout.clone());
                // grad_A = bmm(grad_C, B^T): [B,M,N] @ [B,N,K] → [B,M,K]
                let b_t = saved_b.batched_transpose();
                let grad_a = out_grad.bmm(&b_t);
                // grad_B = bmm(A^T, grad_C): [B,K,M] @ [B,M,N] → [B,K,N]
                let a_t = saved_a.batched_transpose();
                let grad_b = a_t.bmm(&out_grad);
                grads.accumulate(entry.inputs[0], grad_a)?;
                grads.accumulate(entry.inputs[1], grad_b)?;
            }

            BackwardOp::Softmax(bw) => {
                bw.input_version.check()?;
                let saved_out = Tensor::from_storage_and_layout(bw.output_storage.clone(), bw.output_layout.clone());
                let (num_rows, row_size) = (bw.num_rows, bw.row_size);

                #[cfg(feature = "gpu")]
                if out_grad.storage.is_gpu() {
                    use crate::backend::gpu::{
                        compute as gpu_compute,
                        context::{GpuContext, STORAGE_USAGE},
                    };
                    let ctx = GpuContext::get().expect("GPU required");
                    let so_c = saved_out.contiguous();
                    let og_c = out_grad.contiguous();
                    let so_buf = so_c.storage.gpu_buffer();
                    let og_buf = og_c.storage.gpu_buffer();
                    let gi_buf = ctx.pool.acquire(&ctx.device, crate::tensor::DType::F32.gpu_buf_size(num_rows * row_size), STORAGE_USAGE);
                    gpu_compute::softmax_backward_dispatch(
                        ctx, &so_buf, &og_buf, &gi_buf,
                        num_rows as u32, row_size as u32,
                    );
                    drop(so_buf); drop(og_buf);
                    let gi = Tensor::from_storage_and_layout(
                        crate::tensor::StorageHandle::new_gpu(gi_buf, num_rows * row_size),
                        crate::tensor::Layout::contiguous(out_grad.shape().to_vec()),
                    );
                    grads.accumulate(entry.inputs[0], gi)?;
                } else {
                    let so_c = saved_out.contiguous();
                    let og_c = out_grad.contiguous();
                    let sog = so_c.storage.data();
                    let ogg = og_c.storage.data();
                    let mut gi = CpuBackend::zeros(num_rows * row_size);
                    CpuBackend::softmax_backward(&sog, &ogg, &mut gi, num_rows, row_size);
                    drop(sog); drop(ogg);
                    grads.accumulate(entry.inputs[0], Tensor::new(gi, out_grad.shape().to_vec()))?;
                }

                #[cfg(not(feature = "gpu"))]
                {
                    let so_c = saved_out.contiguous();
                    let og_c = out_grad.contiguous();
                    let sog = so_c.storage.data();
                    let ogg = og_c.storage.data();
                    let mut gi = CpuBackend::zeros(num_rows * row_size);
                    CpuBackend::softmax_backward(&sog, &ogg, &mut gi, num_rows, row_size);
                    drop(sog); drop(ogg);
                    grads.accumulate(entry.inputs[0], Tensor::new(gi, out_grad.shape().to_vec()))?;
                }
            }

            BackwardOp::LayerNorm(bw) => {
                bw.input_version.check()?;
                bw.weight_version.check()?;
                let (n, d) = (bw.num_instances, bw.norm_size);

                let saved_input = Tensor::from_storage_and_layout(bw.input_storage.clone(), bw.input_layout.clone());
                let saved_weight = Tensor::from_storage_and_layout(bw.weight_storage.clone(), bw.weight_layout.clone());
                let saved_save = Tensor::from_storage_and_layout(bw.save_storage.clone(), bw.save_layout.clone());

                #[cfg(feature = "gpu")]
                if out_grad.storage.is_gpu() {
                    use crate::backend::gpu::{
                        compute as gpu_compute,
                        context::{GpuContext, STORAGE_USAGE},
                    };
                    let ctx = GpuContext::get().expect("GPU required");

                    let og_c = out_grad.contiguous();
                    let in_c = saved_input.contiguous();
                    let wt_c = saved_weight.contiguous();
                    let sv_c = saved_save.contiguous();
                    let og_buf = og_c.storage.gpu_buffer();
                    let in_buf = in_c.storage.gpu_buffer();
                    let wt_buf = wt_c.storage.gpu_buffer();
                    let sv_buf = sv_c.storage.gpu_buffer();

                    // 1. grad_input via per-instance WGSL kernel
                    let gi_buf = ctx.pool.acquire(&ctx.device, crate::tensor::DType::F32.gpu_buf_size(n * d), STORAGE_USAGE);
                    gpu_compute::layer_norm_backward(
                        ctx, &og_buf, &in_buf, &wt_buf, &sv_buf, &gi_buf,
                        n as u32, d as u32,
                    );

                    // 2. grad_weight = reduce_sum(grad_out * x_hat, dim=0)
                    //    First compute grad_out * x_hat [N, D] on GPU.
                    let gw_product_buf = ctx.pool.acquire(&ctx.device, crate::tensor::DType::F32.gpu_buf_size(n * d), STORAGE_USAGE);
                    gpu_compute::layer_norm_grad_weight_product(
                        ctx, &og_buf, &in_buf, &wt_buf, &sv_buf, &gw_product_buf,
                        n as u32, d as u32,
                    );
                    //    Then reduce over dim 0: [N, D] → [D]
                    let gw_buf = ctx.pool.acquire(&ctx.device, crate::tensor::DType::F32.gpu_buf_size(d), STORAGE_USAGE);
                    {
                        let mut enc = ctx.device.create_command_encoder(&Default::default());
                        enc.clear_buffer(&gw_buf, 0, None);
                        ctx.queue.submit(std::iter::once(enc.finish()));
                    }
                    let gw_params = gpu_compute::make_reduce_sum_params(&[n, d], &[0], d);
                    gpu_compute::reduce_sum_gpu(ctx, &gw_product_buf, &gw_buf, &gw_params);

                    // 3. grad_bias = reduce_sum(grad_out, dim=0): [N, D] → [D]
                    let gb_buf = ctx.pool.acquire(&ctx.device, crate::tensor::DType::F32.gpu_buf_size(d), STORAGE_USAGE);
                    {
                        let mut enc = ctx.device.create_command_encoder(&Default::default());
                        enc.clear_buffer(&gb_buf, 0, None);
                        ctx.queue.submit(std::iter::once(enc.finish()));
                    }
                    let gb_params = gpu_compute::make_reduce_sum_params(&[n, d], &[0], d);
                    gpu_compute::reduce_sum_gpu(ctx, &og_buf, &gb_buf, &gb_params);

                    drop(og_buf); drop(in_buf); drop(wt_buf); drop(sv_buf);
                    ctx.pool.release(gw_product_buf);

                    let gi_t = Tensor::from_storage_and_layout(
                        crate::tensor::StorageHandle::new_gpu(gi_buf, n * d),
                        crate::tensor::Layout::contiguous(vec![n, d]),
                    );
                    let gw_t = Tensor::from_storage_and_layout(
                        crate::tensor::StorageHandle::new_gpu(gw_buf, d),
                        crate::tensor::Layout::contiguous(vec![d]),
                    );
                    let gb_t = Tensor::from_storage_and_layout(
                        crate::tensor::StorageHandle::new_gpu(gb_buf, d),
                        crate::tensor::Layout::contiguous(vec![d]),
                    );

                    grads.accumulate(entry.inputs[0], gi_t)?;
                    grads.accumulate(entry.inputs[1], gw_t)?;
                    grads.accumulate(entry.inputs[2], gb_t)?;

                    // Skip CPU path below.
                } else {
                    // CPU path
                    let og_c = out_grad.contiguous();
                    let in_c = saved_input.contiguous();
                    let wt_c = saved_weight.contiguous();
                    let sv_c = saved_save.contiguous();
                    let ogg = og_c.storage.data();
                    let ing = in_c.storage.data();
                    let wtg = wt_c.storage.data();
                    let svg = sv_c.storage.data();

                    let mut grad_in = CpuBackend::zeros(n * d);
                    CpuBackend::layer_norm_backward(&ogg, &ing, &wtg, &svg, &mut grad_in, n, d);

                    let mut gw = CpuBackend::zeros(d);
                    let mut gb = CpuBackend::zeros(d);
                    for i in 0..n {
                        let mean = svg[i * 2];
                        let invstd = svg[i * 2 + 1];
                        for j in 0..d {
                            let x_hat = (ing[i * d + j] - mean) * invstd;
                            gw[j] += ogg[i * d + j] * x_hat;
                            gb[j] += ogg[i * d + j];
                        }
                    }
                    drop(ogg); drop(ing); drop(wtg); drop(svg);

                    grads.accumulate(entry.inputs[0], Tensor::new(grad_in, vec![n, d]))?;
                    grads.accumulate(entry.inputs[1], Tensor::new(gw, vec![d]))?;
                    grads.accumulate(entry.inputs[2], Tensor::new(gb, vec![d]))?;
                }

                #[cfg(not(feature = "gpu"))]
                {
                    let og_c = out_grad.contiguous();
                    let in_c = saved_input.contiguous();
                    let wt_c = saved_weight.contiguous();
                    let sv_c = saved_save.contiguous();
                    let ogg = og_c.storage.data();
                    let ing = in_c.storage.data();
                    let wtg = wt_c.storage.data();
                    let svg = sv_c.storage.data();

                    let mut grad_in = CpuBackend::zeros(n * d);
                    CpuBackend::layer_norm_backward(&ogg, &ing, &wtg, &svg, &mut grad_in, n, d);

                    let mut gw = CpuBackend::zeros(d);
                    let mut gb = CpuBackend::zeros(d);
                    for i in 0..n {
                        let mean = svg[i * 2];
                        let invstd = svg[i * 2 + 1];
                        for j in 0..d {
                            let x_hat = (ing[i * d + j] - mean) * invstd;
                            gw[j] += ogg[i * d + j] * x_hat;
                            gb[j] += ogg[i * d + j];
                        }
                    }
                    drop(ogg); drop(ing); drop(wtg); drop(svg);

                    grads.accumulate(entry.inputs[0], Tensor::new(grad_in, vec![n, d]))?;
                    grads.accumulate(entry.inputs[1], Tensor::new(gw, vec![d]))?;
                    grads.accumulate(entry.inputs[2], Tensor::new(gb, vec![d]))?;
                }
            }

            BackwardOp::Embedding(bw) => {
                bw.input_version.check()?;
                let saved_idx = Tensor::from_storage_and_layout(
                    bw.indices_storage.clone(), bw.indices_layout.clone(),
                );
                // CPU sparse scatter (no f32 atomics in WGSL).
                let og_c = out_grad.contiguous();
                let ogg = og_c.storage.data();
                let idx_c = saved_idx.contiguous();
                let idxg = idx_c.storage.data();
                let mut gw = CpuBackend::zeros(bw.vocab_size * bw.embed_dim);
                CpuBackend::embedding_backward(&ogg, &idxg, &mut gw, bw.total_lookups, bw.embed_dim);
                drop(ogg); drop(idxg);
                grads.accumulate(entry.inputs[0], Tensor::new(gw, vec![bw.vocab_size, bw.embed_dim]))?;
            }

            BackwardOp::Sigmoid(bw) => {
                bw.input_version.check()?;
                let saved_out = Tensor::from_storage_and_layout(
                    bw.output_storage.clone(), bw.output_layout.clone(),
                );
                let sc = saved_out.contiguous();
                let og = out_grad.contiguous();
                let sg = sc.storage.data();
                let ogg = og.storage.data();
                let mut dst = CpuBackend::zeros(out_grad.numel());
                CpuBackend::sigmoid_backward(&sg, &ogg, &mut dst);
                drop(sg); drop(ogg);
                grads.accumulate(entry.inputs[0], Tensor::new(dst, out_grad.shape().to_vec()))?;
            }

            BackwardOp::Tanh(bw) => {
                bw.input_version.check()?;
                let saved_out = Tensor::from_storage_and_layout(
                    bw.output_storage.clone(), bw.output_layout.clone(),
                );
                let sc = saved_out.contiguous();
                let og = out_grad.contiguous();
                let sg = sc.storage.data();
                let ogg = og.storage.data();
                let mut dst = CpuBackend::zeros(out_grad.numel());
                CpuBackend::tanh_backward(&sg, &ogg, &mut dst);
                drop(sg); drop(ogg);
                grads.accumulate(entry.inputs[0], Tensor::new(dst, out_grad.shape().to_vec()))?;
            }

            BackwardOp::Gelu(bw) => {
                bw.input_version.check()?;
                let saved_in = Tensor::from_storage_and_layout(
                    bw.input_storage.clone(), bw.input_layout.clone(),
                );
                let sc = saved_in.contiguous();
                let og = out_grad.contiguous();
                let sg = sc.storage.data();
                let ogg = og.storage.data();
                let mut dst = CpuBackend::zeros(out_grad.numel());
                CpuBackend::gelu_backward(&sg, &ogg, &mut dst);
                drop(sg); drop(ogg);
                grads.accumulate(entry.inputs[0], Tensor::new(dst, out_grad.shape().to_vec()))?;
            }

            BackwardOp::LeakyRelu(bw) => {
                bw.input_version.check()?;
                let saved_in = Tensor::from_storage_and_layout(
                    bw.input_storage.clone(), bw.input_layout.clone(),
                );
                let sc = saved_in.contiguous();
                let og = out_grad.contiguous();
                let sg = sc.storage.data();
                let ogg = og.storage.data();
                let mut dst = CpuBackend::zeros(out_grad.numel());
                CpuBackend::leaky_relu_backward(&sg, &ogg, &mut dst, bw.alpha);
                drop(sg); drop(ogg);
                grads.accumulate(entry.inputs[0], Tensor::new(dst, out_grad.shape().to_vec()))?;
            }

            BackwardOp::BroadcastAdd(bw) => {
                bw.lhs_version.check()?;
                bw.rhs_version.check()?;
                let grad_lhs = reduce_if_broadcast(&out_grad, &bw.lhs_broadcast);
                let grad_rhs = reduce_if_broadcast(&out_grad, &bw.rhs_broadcast);
                grads.accumulate(entry.inputs[0], grad_lhs)?;
                grads.accumulate(entry.inputs[1], grad_rhs)?;
            }

            BackwardOp::BroadcastSub(bw) => {
                bw.lhs_version.check()?;
                bw.rhs_version.check()?;
                let grad_lhs = reduce_if_broadcast(&out_grad, &bw.lhs_broadcast);
                // Negate out_grad for the RHS: ∂(A-B)/∂B = -1.
                // GPU path: dispatch scale kernel with scalar -1.0 (uniform,
                // never touches host memory as tensor data).
                // CPU path: CpuBackend::scale.
                let neg_og = negate_tensor(&out_grad);
                let grad_rhs = reduce_if_broadcast(&neg_og, &bw.rhs_broadcast);
                grads.accumulate(entry.inputs[0], grad_lhs)?;
                grads.accumulate(entry.inputs[1], grad_rhs)?;
            }

            BackwardOp::BroadcastMul(bw) => {
                bw.lhs_version.check()?;
                bw.rhs_version.check()?;
                let saved_lhs = Tensor::from_storage_and_layout(
                    bw.lhs_storage.clone(), bw.lhs_layout.clone(),
                );
                let saved_rhs = Tensor::from_storage_and_layout(
                    bw.rhs_storage.clone(), bw.rhs_layout.clone(),
                );
                let gl = out_grad.broadcast_mul(&saved_rhs);
                let gr = out_grad.broadcast_mul(&saved_lhs);
                let grad_lhs = reduce_if_broadcast(&gl, &bw.lhs_broadcast);
                let grad_rhs = reduce_if_broadcast(&gr, &bw.rhs_broadcast);
                grads.accumulate(entry.inputs[0], grad_lhs)?;
                grads.accumulate(entry.inputs[1], grad_rhs)?;
            }

            BackwardOp::BatchNorm2d(bw) => {
                bw.input_version.check()?;
                bw.weight_version.check()?;
                let (b, c, h, w) = (bw.batch, bw.channels, bw.height, bw.width);

                let saved_input = Tensor::from_storage_and_layout(bw.input_storage.clone(), bw.input_layout.clone());
                let saved_weight = Tensor::from_storage_and_layout(bw.weight_storage.clone(), bw.weight_layout.clone());
                let saved_save = Tensor::from_storage_and_layout(bw.save_storage.clone(), bw.save_layout.clone());

                #[cfg(feature = "gpu")]
                if out_grad.storage.is_gpu() {
                    use crate::backend::gpu::{
                        compute as gpu_compute,
                        context::{GpuContext, STORAGE_USAGE},
                    };
                    let ctx = GpuContext::get().expect("GPU required");

                    let og_c = out_grad.contiguous();
                    let in_c = saved_input.contiguous();
                    let wt_c = saved_weight.contiguous();
                    let sv_c = saved_save.contiguous();
                    let og_buf = og_c.storage.gpu_buffer();
                    let in_buf = in_c.storage.gpu_buffer();
                    let wt_buf = wt_c.storage.gpu_buffer();
                    let sv_buf = sv_c.storage.gpu_buffer();

                    let gi_buf = ctx.pool.acquire(&ctx.device, crate::tensor::DType::F32.gpu_buf_size(b * c * h * w), STORAGE_USAGE);
                    gpu_compute::batch_norm_backward(
                        ctx, &og_buf, &in_buf, &wt_buf, &sv_buf, &gi_buf,
                        b as u32, c as u32, h as u32, w as u32,
                    );

                    // grad_weight/grad_bias: CPU reduction (D2H via .data()).
                    let og_cpu = og_c.storage.data();
                    let in_cpu = in_c.storage.data();
                    let sv_cpu = sv_c.storage.data();
                    let spatial = h * w;
                    let mut gw = CpuBackend::zeros(c);
                    let mut gb = CpuBackend::zeros(c);
                    for ch in 0..c {
                        let mean = sv_cpu[ch * 2];
                        let invstd = sv_cpu[ch * 2 + 1];
                        for bi in 0..b {
                            for s in 0..spatial {
                                let idx = bi * c * spatial + ch * spatial + s;
                                let x_hat = (in_cpu[idx] - mean) * invstd;
                                gw[ch] += og_cpu[idx] * x_hat;
                                gb[ch] += og_cpu[idx];
                            }
                        }
                    }
                    drop(og_buf); drop(in_buf); drop(wt_buf); drop(sv_buf);
                    drop(og_cpu); drop(in_cpu); drop(sv_cpu);

                    let gi_t = Tensor::from_storage_and_layout(
                        crate::tensor::StorageHandle::new_gpu(gi_buf, b * c * h * w),
                        crate::tensor::Layout::contiguous(vec![b, c, h, w]),
                    );
                    grads.accumulate(entry.inputs[0], gi_t)?;
                    grads.accumulate(entry.inputs[1], Tensor::new(gw, vec![c]))?;
                    grads.accumulate(entry.inputs[2], Tensor::new(gb, vec![c]))?;
                } else {
                    let og_c = out_grad.contiguous();
                    let in_c = saved_input.contiguous();
                    let wt_c = saved_weight.contiguous();
                    let sv_c = saved_save.contiguous();
                    let ogg = og_c.storage.data();
                    let ing = in_c.storage.data();
                    let wtg = wt_c.storage.data();
                    let svg = sv_c.storage.data();

                    let mut grad_in = CpuBackend::zeros(b * c * h * w);
                    CpuBackend::batch_norm_backward(&ogg, &ing, &wtg, &svg, &mut grad_in, b, c, h, w);

                    let spatial = h * w;
                    let mut gw = CpuBackend::zeros(c);
                    let mut gb = CpuBackend::zeros(c);
                    for ch in 0..c {
                        let mean = svg[ch * 2];
                        let invstd = svg[ch * 2 + 1];
                        for bi in 0..b {
                            for s in 0..spatial {
                                let idx = bi * c * spatial + ch * spatial + s;
                                let x_hat = (ing[idx] - mean) * invstd;
                                gw[ch] += ogg[idx] * x_hat;
                                gb[ch] += ogg[idx];
                            }
                        }
                    }
                    drop(ogg); drop(ing); drop(wtg); drop(svg);

                    grads.accumulate(entry.inputs[0], Tensor::new(grad_in, vec![b, c, h, w]))?;
                    grads.accumulate(entry.inputs[1], Tensor::new(gw, vec![c]))?;
                    grads.accumulate(entry.inputs[2], Tensor::new(gb, vec![c]))?;
                }

                #[cfg(not(feature = "gpu"))]
                {
                    let og_c = out_grad.contiguous();
                    let in_c = saved_input.contiguous();
                    let wt_c = saved_weight.contiguous();
                    let sv_c = saved_save.contiguous();
                    let ogg = og_c.storage.data();
                    let ing = in_c.storage.data();
                    let wtg = wt_c.storage.data();
                    let svg = sv_c.storage.data();

                    let mut grad_in = CpuBackend::zeros(b * c * h * w);
                    CpuBackend::batch_norm_backward(&ogg, &ing, &wtg, &svg, &mut grad_in, b, c, h, w);

                    let spatial = h * w;
                    let mut gw = CpuBackend::zeros(c);
                    let mut gb = CpuBackend::zeros(c);
                    for ch in 0..c {
                        let mean = svg[ch * 2];
                        let invstd = svg[ch * 2 + 1];
                        for bi in 0..b {
                            for s in 0..spatial {
                                let idx = bi * c * spatial + ch * spatial + s;
                                let x_hat = (ing[idx] - mean) * invstd;
                                gw[ch] += ogg[idx] * x_hat;
                                gb[ch] += ogg[idx];
                            }
                        }
                    }
                    drop(ogg); drop(ing); drop(wtg); drop(svg);

                    grads.accumulate(entry.inputs[0], Tensor::new(grad_in, vec![b, c, h, w]))?;
                    grads.accumulate(entry.inputs[1], Tensor::new(gw, vec![c]))?;
                    grads.accumulate(entry.inputs[2], Tensor::new(gb, vec![c]))?;
                }
            }

            BackwardOp::AdaptiveAvgPool2d(bw) => {
                bw.input_version.check()?;
                let (b, c, h_in, w_in, h_out, w_out) =
                    (bw.batch, bw.channels, bw.h_in, bw.w_in, bw.h_out, bw.w_out);
                let in_numel = b * c * h_in * w_in;

                #[cfg(feature = "gpu")]
                if out_grad.storage.is_gpu() {
                    use crate::backend::gpu::{
                        compute as gpu_compute,
                        context::{GpuContext, STORAGE_USAGE},
                    };
                    let ctx = GpuContext::get().expect("GPU required");
                    let og_c = out_grad.contiguous();
                    let og_buf = og_c.storage.gpu_buffer();
                    let gi_buf = ctx.pool.acquire(&ctx.device, crate::tensor::DType::F32.gpu_buf_size(in_numel), STORAGE_USAGE);
                    {
                        let mut enc = ctx.device.create_command_encoder(&Default::default());
                        enc.clear_buffer(&gi_buf, 0, None);
                        ctx.queue.submit(std::iter::once(enc.finish()));
                    }
                    gpu_compute::adaptive_avg_pool2d_backward(
                        ctx, &og_buf, &gi_buf,
                        b as u32, c as u32, h_in as u32, w_in as u32, h_out as u32, w_out as u32,
                    );
                    drop(og_buf);
                    let gi_t = Tensor::from_storage_and_layout(
                        crate::tensor::StorageHandle::new_gpu(gi_buf, in_numel),
                        crate::tensor::Layout::contiguous(vec![b, c, h_in, w_in]),
                    );
                    grads.accumulate(entry.inputs[0], gi_t)?;
                } else {
                    let og_c = out_grad.contiguous();
                    let ogg = og_c.storage.data();
                    let mut gi = CpuBackend::zeros(in_numel);
                    CpuBackend::adaptive_avg_pool2d_backward(
                        &ogg, &mut gi, b, c, h_in, w_in, h_out, w_out,
                    );
                    drop(ogg);
                    grads.accumulate(entry.inputs[0], Tensor::new(gi, vec![b, c, h_in, w_in]))?;
                }

                #[cfg(not(feature = "gpu"))]
                {
                    let og_c = out_grad.contiguous();
                    let ogg = og_c.storage.data();
                    let mut gi = CpuBackend::zeros(in_numel);
                    CpuBackend::adaptive_avg_pool2d_backward(
                        &ogg, &mut gi, b, c, h_in, w_in, h_out, w_out,
                    );
                    drop(ogg);
                    grads.accumulate(entry.inputs[0], Tensor::new(gi, vec![b, c, h_in, w_in]))?;
                }
            }

            BackwardOp::Cast(bw) => {
                bw.input_version.check()?;
                let grad_input = out_grad.to_dtype(bw.source_dtype);
                grads.accumulate(entry.inputs[0], grad_input)?;
            }

            BackwardOp::SliceRange(bw) => {
                bw.input_version.check()?;
                // Scatter grad into a zero tensor at the slice position.
                let total_numel: usize = bw.original_shape.iter().product();
                let mut dst = vec![0.0f32; total_numel];
                let og_c = out_grad.contiguous();
                let og_guard = og_c.storage.data();

                // Compute strides for the original shape.
                let ndim = bw.original_shape.len();
                let mut strides = vec![1usize; ndim];
                for i in (0..ndim - 1).rev() {
                    strides[i] = strides[i + 1] * bw.original_shape[i + 1];
                }

                // Copy grad into the slice region.
                let slice_numel: usize = og_c.numel();
                for flat in 0..slice_numel {
                    // Decompose flat index into coordinates of the output grad.
                    let mut rem = flat;
                    let mut src_coords = vec![0usize; ndim];
                    for d in 0..ndim {
                        let dim_size = if d == bw.dim {
                            bw.end - bw.start
                        } else {
                            bw.original_shape[d]
                        };
                        let s = slice_numel / {
                            let mut p = 1;
                            for dd in 0..=d { p *= if dd == bw.dim { bw.end - bw.start } else { bw.original_shape[dd] }; }
                            p
                        };
                        // Actually simpler: use contiguous layout math.
                        src_coords[d] = rem / (slice_numel / {
                            let mut p = 1;
                            for dd in (d..ndim) {
                                p *= if dd == bw.dim { bw.end - bw.start } else { bw.original_shape[dd] };
                            }
                            p
                        });
                        rem %= slice_numel / {
                            let mut p = 1;
                            for dd in (d..ndim) {
                                p *= if dd == bw.dim { bw.end - bw.start } else { bw.original_shape[dd] };
                            }
                            p
                        };
                    }
                    // Offset the dim coordinate.
                    src_coords[bw.dim] += bw.start;
                    let dst_flat: usize = src_coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum();
                    dst[dst_flat] = og_guard[flat];
                }
                drop(og_guard);
                grads.accumulate(entry.inputs[0], Tensor::new(dst, bw.original_shape.clone()))?;
            }

            BackwardOp::Cat(bw) => {
                for v in &bw.versions {
                    v.check()?;
                }
                // Split the grad along the cat dim.
                let og_c = out_grad.contiguous();
                let og_guard = og_c.storage.data();
                let shape = out_grad.shape();
                let ndim = shape.len();

                let mut offset = 0usize;
                for (i, &split_size) in bw.splits.iter().enumerate() {
                    // Build the slice shape.
                    let mut slice_shape: Vec<usize> = shape.to_vec();
                    slice_shape[bw.dim] = split_size;
                    let slice_numel: usize = slice_shape.iter().product();

                    // Compute strides for the output grad.
                    let mut out_strides = vec![1usize; ndim];
                    for d in (0..ndim - 1).rev() {
                        out_strides[d] = out_strides[d + 1] * shape[d + 1];
                    }

                    // Extract the slice.
                    let mut slice_data = vec![0.0f32; slice_numel];
                    let mut slice_strides = vec![1usize; ndim];
                    for d in (0..ndim - 1).rev() {
                        slice_strides[d] = slice_strides[d + 1] * slice_shape[d + 1];
                    }

                    for flat in 0..slice_numel {
                        let mut rem = flat;
                        let mut out_flat = 0usize;
                        for d in 0..ndim {
                            let coord = rem / slice_strides[d];
                            rem %= slice_strides[d];
                            let out_coord = if d == bw.dim { coord + offset } else { coord };
                            out_flat += out_coord * out_strides[d];
                        }
                        slice_data[flat] = og_guard[out_flat];
                    }

                    grads.accumulate(entry.inputs[i], Tensor::new(slice_data, slice_shape))?;
                    offset += split_size;
                }
                drop(og_guard);
            }

            #[cfg(feature = "multi_gpu")]
            BackwardOp::FsdpLinear(bw) => {
                bw.input_version.check()?;

                let d_in = bw.full_weight_shape[1];
                let d_out = bw.full_weight_shape[0];
                let n = bw.world_size as f32;

                // 1. Re-gather the full weight from shard storages (CPU staging).
                let full_w = {
                    let mut all_data: Vec<f32> = Vec::new();
                    for shard_storage in &bw.weight_shard_storages {
                        let guard = shard_storage.data();
                        all_data.extend_from_slice(&guard);
                        drop(guard);
                    }
                    let full_numel: usize = bw.full_weight_shape.iter().product();
                    all_data.truncate(full_numel);
                    Tensor::new(all_data, bw.full_weight_shape.clone())
                };

                // 2. grad_X = grad_Y @ W
                let w_t = full_w.transpose(0, 1);
                let grad_x = out_grad.matmul(&w_t);
                grads.accumulate(entry.inputs[0], grad_x)?;
                drop(full_w);

                // 3. Compute LOCAL grad_W_full = grad_Y^T @ X
                let saved_input = Tensor::from_storage_and_layout(
                    bw.input_storage.clone(), bw.input_layout.clone(),
                );
                let og_t = out_grad.transpose(0, 1);
                let grad_w_local = og_t.matmul(&saved_input);
                let gw_c = grad_w_local.contiguous();
                let gw_guard = gw_c.storage.data();
                let local_gw: Vec<f32> = gw_guard.to_vec();
                drop(gw_guard);
                drop(grad_w_local);

                // 4. Compute LOCAL grad_b_full (if bias).
                let local_gb = if bw.has_bias {
                    let og_c = out_grad.contiguous();
                    let ogg = og_c.storage.data();
                    let batch = out_grad.shape()[0];
                    let mut gb = vec![0.0f32; d_out];
                    for b_idx in 0..batch {
                        for j in 0..d_out {
                            gb[j] += ogg[b_idx * d_out + j];
                        }
                    }
                    drop(ogg);
                    gb
                } else {
                    vec![]
                };

                // 5. BARRIER: push local grads, wait for all ranks, reduce.
                let (reduced_gw, reduced_gb) = {
                    let sync = &bw.sync;
                    let mut state = sync.state.lock().unwrap();

                    // Push this rank's local gradients.
                    state.weight_grads.push(local_gw);
                    if bw.has_bias {
                        state.bias_grads.push(local_gb);
                    }

                    // Am I the last to arrive?
                    if state.weight_grads.len() == sync.world_size {
                        // Reduce: sum all ranks' gradients element-wise, then average.
                        let numel_w = d_out * d_in;
                        let mut summed_w = vec![0.0f32; numel_w];
                        for gw in &state.weight_grads {
                            for (s, &v) in summed_w.iter_mut().zip(gw.iter()) {
                                *s += v;
                            }
                        }
                        for v in &mut summed_w {
                            *v /= n;
                        }

                        let summed_b = if bw.has_bias {
                            let mut sb = vec![0.0f32; d_out];
                            for gb in &state.bias_grads {
                                for (s, &v) in sb.iter_mut().zip(gb.iter()) {
                                    *s += v;
                                }
                            }
                            for v in &mut sb {
                                *v /= n;
                            }
                            sb
                        } else {
                            vec![]
                        };

                        // Store the result for other threads to read.
                        state.weight_result = Some(summed_w);
                        state.bias_result = Some(summed_b);
                        state.read_count = 0;

                        // Wake all waiting threads.
                        sync.cvar.notify_all();
                    } else {
                        // Wait for the last rank to finish reducing.
                        state = sync.cvar.wait_while(state, |s| {
                            s.weight_result.is_none()
                        }).unwrap();
                    }

                    // Read the reduced result.
                    let rw = state.weight_result.as_ref().unwrap().clone();
                    let rb = state.bias_result.as_ref().cloned().unwrap_or_default();
                    state.read_count += 1;

                    // Last reader clears the state for the next iteration.
                    if state.read_count == sync.world_size {
                        state.weight_grads.clear();
                        state.bias_grads.clear();
                        state.weight_result = None;
                        state.bias_result = None;
                        state.read_count = 0;
                    }

                    (rw, rb)
                };

                // 6. Scatter: slice this rank's shard from the reduced gradient.
                let shard_start = bw.weight_shard_offset * d_in;
                let shard_end = shard_start + bw.shard_size * d_in;
                let shard_data = reduced_gw[shard_start..shard_end].to_vec();
                let shard_grad = Tensor::new(shard_data, vec![bw.shard_size, d_in]);
                grads.accumulate(entry.inputs[1], shard_grad)?;

                // 7. Bias shard gradient.
                if bw.has_bias {
                    let bs_start = bw.bias_shard_offset;
                    let bs_end = bs_start + bw.bias_shard_size;
                    let bs_data = reduced_gb[bs_start..bs_end].to_vec();
                    let bs_grad = Tensor::new(bs_data, vec![bw.bias_shard_size]);
                    grads.accumulate(entry.inputs[2], bs_grad)?;
                }
            }

            BackwardOp::Custom(custom) => {
                for vs in &custom.input_versions {
                    vs.check()?;
                }
                // Reconstruct saved tensors from their StorageHandle + Layout.
                let saved: Vec<Tensor> = custom.saved_storages.iter()
                    .zip(custom.saved_layouts.iter())
                    .map(|(s, l)| Tensor::from_storage_and_layout(s.clone(), l.clone()))
                    .collect();
                // Invoke the user's backward math.
                let input_grads = custom.handler.backward(&out_grad, &saved);
                // Accumulate each gradient against the corresponding input GradId.
                for (i, grad) in input_grads.into_iter().enumerate() {
                    if i < entry.inputs.len() {
                        grads.accumulate(entry.inputs[i], grad)?;
                    }
                }
            }
        }

        for &input_id in &entry.inputs {
            if let Some(count) = pending.get_mut(&input_id) {
                *count -= 1;
            }
        }
    }

    Ok(grads)
}
