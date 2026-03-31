//! Stochastic Gradient Descent with optional momentum.

use std::collections::HashMap;

use crate::autograd::{AutogradError, GradientStore};
use crate::nn::Parameter;
use crate::optim::Optimizer;
use crate::tensor::{ParamId, Tensor};

/// SGD optimizer with optional momentum.
///
/// When `momentum > 0`, maintains a velocity buffer per parameter:
///
/// ```text
/// v_t = momentum * v_{t-1} + grad
/// param -= lr * v_t
/// ```
///
/// Velocity tensors are created with `AutogradState::None` — optimizer
/// state is never tracked by the autograd engine.
pub struct SGD {
    params: Vec<Parameter>,
    lr: f32,
    momentum: f32,
    /// Momentum buffers, keyed by [`ParamId`].  Lazily initialized on
    /// the first step.
    velocity: HashMap<ParamId, Tensor>,
}

impl SGD {
    /// Create a new SGD optimizer.
    ///
    /// - `params`: cloned from a module's `parameters()`.
    /// - `lr`: learning rate.
    /// - `momentum`: momentum factor (0.0 for vanilla SGD).
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
            let grad_guard = grad.storage.data();
            let numel = param.tensor.numel();

            if self.momentum > 0.0 {
                let shape = param.tensor.shape().to_vec();
                let vel = self
                    .velocity
                    .entry(param.id)
                    .or_insert_with(|| Tensor::new(vec![0.0; numel], shape));

                // Write locks on velocity and param — separate Arc allocations,
                // no deadlock possible.  Block scope makes lock lifetimes
                // visually explicit.
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
            // All write guards dropped by block scope; grad read guard
            // dropped explicitly before version bump.
            drop(grad_guard);
            param.tensor.storage.bump_version();
        }

        Ok(())
    }
}
