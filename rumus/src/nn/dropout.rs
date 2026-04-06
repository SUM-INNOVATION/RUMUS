// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Dropout regularization layer.

use std::collections::HashMap;

use crate::autograd::AutogradError;
use crate::nn::{Module, Parameter};
use crate::tensor::Tensor;

/// Dropout layer.
///
/// During training, randomly zeroes elements with probability `p` and
/// scales survivors by `1 / (1 - p)` (inverted dropout).  During
/// evaluation, forward is a no-op (returns a clone of the input).
///
/// The `#[derive(Module)]` macro calls `train()` / `eval()` recursively
/// on all fields, so `Dropout` automatically toggles when the user
/// calls `model.train()` or `model.eval()`.
pub struct Dropout {
    pub p: f32,
    is_training: bool,
}

impl Dropout {
    /// Create a new Dropout layer.
    ///
    /// `p` is the probability of dropping each element (0.0 = no dropout,
    /// 0.5 = 50% dropout).  Must be in `[0, 1)`.
    pub fn new(p: f32) -> Self {
        assert!(p >= 0.0 && p < 1.0, "Dropout: p must be in [0, 1)");
        Self {
            p,
            is_training: true,
        }
    }

    /// Forward pass.
    ///
    /// Returns `input.clone()` during eval (zero overhead).
    /// Applies tracked dropout during training.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        if !self.is_training || self.p == 0.0 {
            return input.clone();
        }
        input.dropout(self.p)
    }
}

impl Module for Dropout {
    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }

    fn train(&mut self) {
        self.is_training = true;
    }

    fn eval(&mut self) {
        self.is_training = false;
    }

    fn state_dict(&self, _prefix: &str) -> HashMap<String, Tensor> {
        HashMap::new()
    }

    fn load_state_dict(
        &mut self,
        _dict: &HashMap<String, Tensor>,
        _prefix: &str,
    ) -> Result<(), AutogradError> {
        Ok(())
    }
}
