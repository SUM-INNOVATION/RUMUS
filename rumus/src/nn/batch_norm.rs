//! Batch normalization for 2D spatial data (BatchNorm2d).

use std::cell::RefCell;
use std::collections::HashMap;

use crate::autograd::AutogradError;
use crate::nn::{Module, Parameter};
use crate::tensor::Tensor;

/// Batch normalization over `[B, C, H, W]` inputs.
///
/// During training, normalizes each channel using per-batch statistics
/// and updates exponential moving averages of mean and variance.
/// During eval, uses the stored running statistics.
///
/// Weight (γ) initialized to ones, bias (β) to zeros.
/// `running_mean` / `running_var` are non-parameter state (not optimized),
/// stored in `RefCell` for interior mutability during `&self` forward.
pub struct BatchNorm2d {
    pub weight: Parameter,
    pub bias: Parameter,
    running_mean: RefCell<Vec<f32>>,
    running_var: RefCell<Vec<f32>>,
    pub num_features: usize,
    pub epsilon: f32,
    pub momentum: f32,
    is_training: bool,
}

impl BatchNorm2d {
    pub fn new(num_features: usize) -> Self {
        Self::with_config(num_features, 1e-5, 0.1)
    }

    pub fn with_config(num_features: usize, epsilon: f32, momentum: f32) -> Self {
        let weight = Parameter::new(Tensor::new(vec![1.0; num_features], vec![num_features]));
        let bias = Parameter::new(Tensor::new(vec![0.0; num_features], vec![num_features]));
        Self {
            weight,
            bias,
            running_mean: RefCell::new(vec![0.0; num_features]),
            running_var: RefCell::new(vec![1.0; num_features]),
            num_features,
            epsilon,
            momentum,
            is_training: true,
        }
    }

    /// Forward pass.
    ///
    /// `input` shape: `[B, C, H, W]`.
    /// Output shape: `[B, C, H, W]` (same).
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let mut rm = self.running_mean.borrow().clone();
        let mut rv = self.running_var.borrow().clone();

        let out = input.batch_norm_2d(
            &self.weight.tensor,
            &self.bias.tensor,
            &mut rm,
            &mut rv,
            self.epsilon,
            self.momentum,
            self.is_training,
        );

        if self.is_training {
            *self.running_mean.borrow_mut() = rm;
            *self.running_var.borrow_mut() = rv;
        }

        out
    }
}

impl Module for BatchNorm2d {
    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn train(&mut self) {
        self.is_training = true;
    }

    fn eval(&mut self) {
        self.is_training = false;
    }

    fn state_dict(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut dict = self.weight.state_dict(&format!("{}weight.", prefix));
        dict.extend(self.bias.state_dict(&format!("{}bias.", prefix)));
        dict.insert(
            format!("{}running_mean", prefix),
            Tensor::new(self.running_mean.borrow().clone(), vec![self.num_features]),
        );
        dict.insert(
            format!("{}running_var", prefix),
            Tensor::new(self.running_var.borrow().clone(), vec![self.num_features]),
        );
        dict
    }

    fn load_state_dict(
        &mut self,
        dict: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), AutogradError> {
        self.weight.load_state_dict(dict, &format!("{}weight.", prefix))?;
        self.bias.load_state_dict(dict, &format!("{}bias.", prefix))?;
        if let Some(rm) = dict.get(&format!("{}running_mean", prefix)) {
            let g = rm.contiguous();
            let d = g.storage.data();
            *self.running_mean.borrow_mut() = d.to_vec();
        }
        if let Some(rv) = dict.get(&format!("{}running_var", prefix)) {
            let g = rv.contiguous();
            let d = g.storage.data();
            *self.running_var.borrow_mut() = d.to_vec();
        }
        Ok(())
    }
}
