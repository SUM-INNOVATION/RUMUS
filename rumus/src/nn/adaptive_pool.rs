//! Adaptive average pooling layer.

use std::collections::HashMap;

use crate::autograd::AutogradError;
use crate::nn::{Module, Parameter};
use crate::tensor::Tensor;

/// Adaptive average pooling: reduces spatial dims to a fixed output size.
///
/// No learnable parameters.  Dynamically computes bin boundaries so any
/// input spatial size maps to `(output_h, output_w)`.
///
/// Commonly used before the classifier head to handle variable input sizes.
pub struct AdaptiveAvgPool2d {
    pub output_h: usize,
    pub output_w: usize,
}

impl AdaptiveAvgPool2d {
    /// Create a new AdaptiveAvgPool2d with the given output size.
    ///
    /// `output_size` is `(H_out, W_out)`.  Use `(1, 1)` for global
    /// average pooling.
    pub fn new(output_h: usize, output_w: usize) -> Self {
        Self { output_h, output_w }
    }

    /// Forward pass.
    ///
    /// `input` shape: `[B, C, H, W]`.
    /// Output shape: `[B, C, output_h, output_w]`.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        input.adaptive_avg_pool2d(self.output_h, self.output_w)
    }
}

impl Module for AdaptiveAvgPool2d {
    fn parameters(&self) -> Vec<Parameter> { vec![] }
    fn state_dict(&self, _prefix: &str) -> HashMap<String, Tensor> { HashMap::new() }
    fn load_state_dict(
        &mut self,
        _dict: &HashMap<String, Tensor>,
        _prefix: &str,
    ) -> Result<(), AutogradError> {
        Ok(())
    }
}
