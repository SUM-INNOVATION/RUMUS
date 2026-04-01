//! Flatten layer — reshapes `[batch, ...]` to `[batch, numel]`.

use std::collections::HashMap;

use crate::autograd::AutogradError;
use crate::nn::{Module, Parameter};
use crate::tensor::Tensor;

/// Flatten layer.  Zero-copy tracked reshape.
///
/// No learnable parameters.  Collapses all spatial dimensions into one.
pub struct Flatten;

impl Flatten {
    pub fn new() -> Self { Self }

    /// Forward pass.
    ///
    /// `input` shape: `[batch, d1, d2, ...]`.
    /// Output shape: `[batch, d1*d2*...]`.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        input.flatten()
    }
}

impl Module for Flatten {
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
