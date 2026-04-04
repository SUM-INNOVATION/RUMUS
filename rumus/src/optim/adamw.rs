//! AdamW optimizer (decoupled weight decay).
//!
//! Weight decay is applied directly to weights before the gradient step,
//! not through the gradient (unlike Adam with L2 regularization).
//! Moment buffers are initialized on the GPU when parameters are GPU-resident.

use std::collections::HashMap;

use crate::autograd::{AutogradError, GradientStore};
use crate::nn::Parameter;
use crate::optim::Optimizer;
use crate::tensor::{ParamId, Tensor};

pub struct AdamW {
    params: Vec<Parameter>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    step_count: usize,
    first_moment: HashMap<ParamId, Tensor>,
    second_moment: HashMap<ParamId, Tensor>,
}

impl AdamW {
    /// Create an AdamW optimizer with default hyperparameters.
    ///
    /// Defaults: `beta1 = 0.9`, `beta2 = 0.999`, `epsilon = 1e-8`,
    /// `weight_decay = 0.01`.
    pub fn new(params: Vec<Parameter>, lr: f32) -> Self {
        Self::with_params(params, lr, 0.9, 0.999, 1e-8, 0.01)
    }

    pub fn with_params(
        params: Vec<Parameter>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Self {
        Self {
            params,
            lr,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            step_count: 0,
            first_moment: HashMap::new(),
            second_moment: HashMap::new(),
        }
    }
}

impl Optimizer for AdamW {
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

            // Get or init moment buffers.
            let m = self.first_moment.entry(param.id).or_insert_with(|| {
                #[cfg(feature = "gpu")]
                if param.tensor.storage.is_gpu() {
                    return gpu_zeros(numel, shape.clone());
                }
                Tensor::new(vec![0.0; numel], shape.clone())
            });
            let v = self.second_moment.entry(param.id).or_insert_with(|| {
                #[cfg(feature = "gpu")]
                if param.tensor.storage.is_gpu() {
                    return gpu_zeros(numel, shape.clone());
                }
                Tensor::new(vec![0.0; numel], shape)
            });

            // ---- GPU path ----
            #[cfg(feature = "gpu")]
            if param.tensor.storage.is_gpu() {
                use crate::backend::gpu::{
                    compute as gpu_compute,
                    context::GpuContext,
                };

                let ctx = GpuContext::get().expect("GPU required");
                grad.to_gpu();
                m.to_gpu();
                v.to_gpu();

                let grad_buf = grad.storage.gpu_buffer();
                let m_buf = m.storage.gpu_buffer();
                let v_buf = v.storage.gpu_buffer();
                let param_buf = param.tensor.storage.gpu_buffer();

                gpu_compute::adamw_step(
                    ctx,
                    &grad_buf, &m_buf, &v_buf, &param_buf,
                    numel as u32,
                    self.lr, self.beta1, self.beta2, self.epsilon,
                    bc1, bc2, self.weight_decay,
                );

                drop(grad_buf);
                drop(m_buf);
                drop(v_buf);
                drop(param_buf);

                m.storage.mark_gpu_dirty();
                v.storage.mark_gpu_dirty();
                param.tensor.storage.mark_gpu_dirty();
                param.tensor.storage.bump_version();
                continue;
            }

            // ---- CPU path ----
            let grad_guard = grad.storage.data();
            {
                let mut m_data = m.storage.data_write();
                let mut v_data = v.storage.data_write();
                let mut param_data = param.tensor.storage.data_write();

                let beta1 = self.beta1;
                let beta2 = self.beta2;
                let lr = self.lr;
                let eps = self.epsilon;
                let wd = self.weight_decay;

                for i in 0..numel {
                    let g = grad_guard[i];
                    m_data[i] = beta1 * m_data[i] + (1.0 - beta1) * g;
                    v_data[i] = beta2 * v_data[i] + (1.0 - beta2) * g * g;
                    let m_hat = m_data[i] / bc1;
                    let v_hat = v_data[i] / bc2;
                    // Decoupled weight decay.
                    param_data[i] -= lr * wd * param_data[i];
                    // Adaptive gradient step.
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

/// Allocate a GPU-native zeroed tensor — strictly on-device.
///
/// Uses `encoder.clear_buffer` (a GPU memset) instead of allocating a
/// CPU `Vec<u8>` and `queue.write_buffer`.  Zero host-side allocation.
#[cfg(feature = "gpu")]
fn gpu_zeros(numel: usize, shape: Vec<usize>) -> Tensor {
    use crate::backend::gpu::context::{GpuContext, STORAGE_USAGE};
    use crate::tensor::{Layout, StorageHandle};

    let ctx = GpuContext::get().expect("GPU required");
    let buf = ctx.pool.acquire(&ctx.device, (numel * 4) as u64, STORAGE_USAGE);
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    encoder.clear_buffer(&buf, 0, None);
    ctx.queue.submit(std::iter::once(encoder.finish()));
    Tensor::from_storage_and_layout(
        StorageHandle::new_gpu(buf, numel),
        Layout::contiguous(shape),
    )
}
