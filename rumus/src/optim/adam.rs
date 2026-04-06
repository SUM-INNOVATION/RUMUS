// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Adam optimizer (Kingma & Ba, 2014).
//!
//! When parameters are GPU-resident, dispatches a fused WGSL kernel that
//! updates moments and weights in a single pass — zero D2H transfers.

use std::collections::HashMap;

use crate::autograd::{AutogradError, GradientStore};
use crate::nn::Parameter;
use crate::optim::Optimizer;
use crate::tensor::{ParamId, Tensor};

pub struct Adam {
    params: Vec<Parameter>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    step_count: usize,
    first_moment: HashMap<ParamId, Tensor>,
    second_moment: HashMap<ParamId, Tensor>,
}

impl Adam {
    pub fn new(params: Vec<Parameter>, lr: f32) -> Self {
        Self::with_betas(params, lr, 0.9, 0.999, 1e-8)
    }

    pub fn with_betas(
        params: Vec<Parameter>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> Self {
        Self {
            params,
            lr,
            beta1,
            beta2,
            epsilon,
            step_count: 0,
            first_moment: HashMap::new(),
            second_moment: HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, grads: &mut GradientStore) -> Result<(), AutogradError> {
        self.step_count += 1;
        let t = self.step_count;
        let bc1 = 1.0 - self.beta1.powi(t as i32);
        let bc2 = 1.0 - self.beta2.powi(t as i32);

        for param in &self.params {
            let raw_grad = grads
                .remove(param.grad_id())
                .ok_or(AutogradError::MissingGrad {
                    grad_id: param.grad_id(),
                })?;
            let grad = if raw_grad.dtype() != crate::tensor::DType::F32 {
                raw_grad.to_dtype(crate::tensor::DType::F32)
            } else {
                raw_grad
            };
            let numel = param.tensor.numel();
            let shape = param.tensor.shape().to_vec();

            let m = self
                .first_moment
                .entry(param.id)
                .or_insert_with(|| Tensor::new(vec![0.0; numel], shape.clone()));
            let v = self
                .second_moment
                .entry(param.id)
                .or_insert_with(|| Tensor::new(vec![0.0; numel], shape));

            // ---- GPU path ----
            #[cfg(feature = "gpu")]
            if param.tensor.storage.is_gpu() {
                use crate::backend::gpu::{
                    compute as gpu_compute,
                    context::GpuContext,
                };

                let ctx = GpuContext::get().expect("GPU required for GPU param");

                // Ensure grad and moment buffers are on GPU.
                grad.to_gpu();
                m.to_gpu();
                v.to_gpu();

                let grad_buf = grad.storage.gpu_buffer();
                let m_buf = m.storage.gpu_buffer();
                let v_buf = v.storage.gpu_buffer();
                let param_buf = param.tensor.storage.gpu_buffer();

                gpu_compute::adam_step(
                    ctx,
                    &grad_buf,
                    &m_buf,
                    &v_buf,
                    &param_buf,
                    numel as u32,
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.epsilon,
                    bc1,
                    bc2,
                );

                drop(grad_buf);
                drop(m_buf);
                drop(v_buf);
                drop(param_buf);

                // GPU wrote to m, v, and param — mark CPU copies stale.
                m.storage.mark_gpu_dirty();
                v.storage.mark_gpu_dirty();
                param.tensor.storage.mark_gpu_dirty();
                param.tensor.storage.bump_version();
                continue;
            }

            // ---- CPU path (unchanged) ----
            let grad_guard = grad.storage.data();
            {
                let mut m_data = m.storage.data_write();
                let mut v_data = v.storage.data_write();
                let mut param_data = param.tensor.storage.data_write();

                let beta1 = self.beta1;
                let beta2 = self.beta2;
                let lr = self.lr;
                let eps = self.epsilon;

                for i in 0..numel {
                    let g = grad_guard[i];
                    m_data[i] = beta1 * m_data[i] + (1.0 - beta1) * g;
                    v_data[i] = beta2 * v_data[i] + (1.0 - beta2) * g * g;
                    let m_hat = m_data[i] / bc1;
                    let v_hat = v_data[i] / bc2;
                    param_data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                }
            }
            drop(grad_guard);
            param.tensor.storage.bump_version();
        }

        Ok(())
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }
}
