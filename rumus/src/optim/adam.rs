//! Adam optimizer (Kingma & Ba, 2014).
//!
//! Maintains per-parameter first-moment (m) and second-moment (v) estimates
//! keyed by [`ParamId`].  All state tensors are created with
//! `AutogradState::None` — optimizer state is never tracked by autograd.

use std::collections::HashMap;

use crate::autograd::{AutogradError, GradientStore};
use crate::nn::Parameter;
use crate::optim::Optimizer;
use crate::tensor::{ParamId, Tensor};

/// Adam optimizer.
///
/// Update rule (per parameter):
///
/// ```text
/// m_t = beta1 * m_{t-1} + (1 - beta1) * grad
/// v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
/// m_hat = m_t / (1 - beta1^t)
/// v_hat = v_t / (1 - beta2^t)
/// param -= lr * m_hat / (sqrt(v_hat) + epsilon)
/// ```
///
/// ## Deadlock analysis
///
/// `step()` acquires write guards on three separate `Arc<StorageInner>`
/// allocations per parameter (first_moment, second_moment, param).  Since
/// each parameter has its own independent storage allocations, no two
/// lock targets overlap and deadlock is impossible.
pub struct Adam {
    params: Vec<Parameter>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    /// Global step counter for bias correction.
    step_count: usize,
    /// First moment estimates (m), keyed by [`ParamId`].
    first_moment: HashMap<ParamId, Tensor>,
    /// Second moment estimates (v), keyed by [`ParamId`].
    second_moment: HashMap<ParamId, Tensor>,
}

impl Adam {
    /// Create an Adam optimizer with default hyperparameters.
    ///
    /// Defaults: `beta1 = 0.9`, `beta2 = 0.999`, `epsilon = 1e-8`.
    pub fn new(params: Vec<Parameter>, lr: f32) -> Self {
        Self::with_betas(params, lr, 0.9, 0.999, 1e-8)
    }

    /// Create an Adam optimizer with custom hyperparameters.
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

        // Bias correction factors (scalars, no locks needed).
        let bc1 = 1.0 - self.beta1.powi(t as i32);
        let bc2 = 1.0 - self.beta2.powi(t as i32);

        for param in &self.params {
            // ---- Phase 1: Extract gradient (move out of store) ----
            let grad = grads
                .remove(param.grad_id())
                .ok_or(AutogradError::MissingGrad {
                    grad_id: param.grad_id(),
                })?;
            let grad_guard = grad.storage.data();
            let numel = param.tensor.numel();

            // ---- Phase 2: Get or initialize moment buffers ----
            let shape = param.tensor.shape().to_vec();
            let m = self
                .first_moment
                .entry(param.id)
                .or_insert_with(|| Tensor::new(vec![0.0; numel], shape.clone()));
            let v = self
                .second_moment
                .entry(param.id)
                .or_insert_with(|| Tensor::new(vec![0.0; numel], shape));

            // ---- Phase 3: Update moments + apply weight update ----
            // Lock order: m → v → param.  All three are separate
            // Arc<StorageInner> allocations — no deadlock possible.
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

                    // Update moments.
                    m_data[i] = beta1 * m_data[i] + (1.0 - beta1) * g;
                    v_data[i] = beta2 * v_data[i] + (1.0 - beta2) * g * g;

                    // Bias-corrected estimates.
                    let m_hat = m_data[i] / bc1;
                    let v_hat = v_data[i] / bc2;

                    // Apply update.
                    param_data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                }
            }
            // All write guards dropped here.

            drop(grad_guard);
            param.tensor.storage.bump_version();
        }

        Ok(())
    }
}
