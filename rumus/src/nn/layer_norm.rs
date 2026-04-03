//! Layer normalization.

use std::collections::HashMap;

use crate::autograd::AutogradError;
use crate::nn::{Module, Parameter};
use crate::tensor::Tensor;

/// Layer normalization over the last dimension.
///
/// `y = γ * (x - mean) / sqrt(var + ε) + β`
///
/// Initialized with `weight = ones`, `bias = zeros` (standard PyTorch default).
pub struct LayerNorm {
    pub weight: Parameter,  // γ, shape [norm_size]
    pub bias: Parameter,    // β, shape [norm_size]
    pub norm_size: usize,
    pub epsilon: f32,
}

impl LayerNorm {
    pub fn new(norm_size: usize, epsilon: f32) -> Self {
        let weight = Parameter::new(Tensor::new(vec![1.0; norm_size], vec![norm_size]));
        let bias = Parameter::new(Tensor::new(vec![0.0; norm_size], vec![norm_size]));
        Self { weight, bias, norm_size, epsilon }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        input.layer_norm(&self.weight.tensor, &self.bias.tensor, self.epsilon)
    }
}

impl Module for LayerNorm {
    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn state_dict(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut dict = self.weight.state_dict(&format!("{}weight.", prefix));
        dict.extend(self.bias.state_dict(&format!("{}bias.", prefix)));
        dict
    }

    fn load_state_dict(
        &mut self,
        dict: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), AutogradError> {
        self.weight.load_state_dict(dict, &format!("{}weight.", prefix))?;
        self.bias.load_state_dict(dict, &format!("{}bias.", prefix))?;
        Ok(())
    }
}
