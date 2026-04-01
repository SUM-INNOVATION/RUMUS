//! Kahn's-algorithm backward traversal of the computational graph.

use std::collections::HashMap;

use crate::autograd::backward_ops::BackwardOp;
use crate::autograd::context;
use crate::autograd::{AutogradError, GradientStore};
use crate::backend::{Backend, CpuBackend};
use crate::tensor::{GradId, Tensor};

/// Execute the backward pass from `tensor`, returning accumulated gradients.
///
/// See module-level docs for the full Kahn's algorithm description.
pub fn backward(tensor: &Tensor) -> Result<GradientStore, AutogradError> {
    let root_grad_id = tensor.grad_id().ok_or(AutogradError::NoGraph)?;

    assert_eq!(
        tensor.numel(),
        1,
        "backward() requires a scalar tensor (numel == 1), got numel == {}",
        tensor.numel(),
    );

    let tape = context::take_tape().ok_or(AutogradError::NoGraph)?;
    let entries = tape.into_entries();

    let mut grads = GradientStore::new();
    let seed = Tensor::new(vec![1.0f32], tensor.shape().to_vec());
    // If the root tensor is GPU-resident, push the seed to the GPU so
    // the entire backward pass stays on-device — all tensor ops check
    // is_gpu() and dispatch WGSL kernels automatically.
    #[cfg(feature = "gpu")]
    if tensor.storage.is_gpu() {
        seed.to_gpu();
    }
    grads.accumulate(root_grad_id, seed)?;

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
                // ∂L/∂a = ∂L/∂c
                grads.accumulate(entry.inputs[0], out_grad.clone())?;
                // ∂L/∂b = -∂L/∂c
                let og_guard = out_grad.storage.data();
                let mut neg = CpuBackend::zeros(out_grad.numel());
                CpuBackend::scale(&og_guard, &mut neg, -1.0);
                drop(og_guard);
                let grad_rhs = Tensor::new(neg, out_grad.shape().to_vec());
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
                // ∂L/∂src = ∂L/∂out (identity)
                grads.accumulate(entry.inputs[0], out_grad.clone())?;
                // ∂L/∂bias = sum over spatial per channel
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
        }

        for &input_id in &entry.inputs {
            if let Some(count) = pending.get_mut(&input_id) {
                *count -= 1;
            }
        }
    }

    Ok(grads)
}
