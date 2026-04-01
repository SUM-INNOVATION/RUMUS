//! Max pooling layer.

use std::collections::HashMap;

use crate::autograd::AutogradError;
use crate::nn::{Module, Parameter};
use crate::tensor::Tensor;

/// 2D max pooling layer.
///
/// No learnable parameters.  Slides a `kernel_size × kernel_size` window
/// with the given `stride` and extracts the maximum value in each patch.
///
/// Requires `stride >= kernel_size` (non-overlapping windows).
pub struct MaxPool2d {
    pub kernel_size: usize,
    pub stride: usize,
}

impl MaxPool2d {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self { kernel_size, stride }
    }

    /// Forward pass.
    ///
    /// `input` shape: `[C, H, W]` (single batch element, called per-batch
    /// inside a CNN forward).
    /// Output shape: `[C, out_h, out_w]`.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        input.max_pool2d(self.kernel_size, self.stride)
    }
}

impl Module for MaxPool2d {
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
