//! Stochastic Gradient Descent with optional momentum.
//!
//! When parameters are GPU-resident, dispatches a WGSL kernel —
//! zero D2H transfers.

use std::collections::HashMap;

use crate::autograd::{AutogradError, GradientStore};
use crate::nn::Parameter;
use crate::optim::Optimizer;
use crate::tensor::{ParamId, Tensor};

pub struct SGD {
    params: Vec<Parameter>,
    lr: f32,
    momentum: f32,
    velocity: HashMap<ParamId, Tensor>,
}

impl SGD {
    pub fn new(params: Vec<Parameter>, lr: f32, momentum: f32) -> Self {
        Self {
            params,
            lr,
            momentum,
            velocity: HashMap::new(),
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, grads: &mut GradientStore) -> Result<(), AutogradError> {
        for param in &self.params {
            let grad = grads
                .remove(param.grad_id())
                .ok_or(AutogradError::MissingGrad {
                    grad_id: param.grad_id(),
                })?;
            let numel = param.tensor.numel();

            // Ensure velocity buffer exists (needed for both GPU and CPU
            // when momentum > 0; also used as dummy for GPU path when
            // momentum == 0).
            let shape = param.tensor.shape().to_vec();
            let vel = self
                .velocity
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

                grad.to_gpu();
                vel.to_gpu();

                let grad_buf = grad.storage.gpu_buffer();
                let vel_buf = vel.storage.gpu_buffer();
                let param_buf = param.tensor.storage.gpu_buffer();

                gpu_compute::sgd_step(
                    ctx,
                    &grad_buf,
                    &vel_buf,
                    &param_buf,
                    numel as u32,
                    self.lr,
                    self.momentum,
                );

                drop(grad_buf);
                drop(vel_buf);
                drop(param_buf);

                vel.storage.mark_gpu_dirty();
                param.tensor.storage.mark_gpu_dirty();
                param.tensor.storage.bump_version();
                continue;
            }

            // ---- CPU path ----
            let grad_guard = grad.storage.data();

            if self.momentum > 0.0 {
                {
                    let mut vel_data = vel.storage.data_write();
                    let mut param_data = param.tensor.storage.data_write();

                    for i in 0..numel {
                        vel_data[i] = self.momentum * vel_data[i] + grad_guard[i];
                        param_data[i] -= self.lr * vel_data[i];
                    }
                }
            } else {
                {
                    let mut param_data = param.tensor.storage.data_write();
                    for i in 0..numel {
                        param_data[i] -= self.lr * grad_guard[i];
                    }
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
