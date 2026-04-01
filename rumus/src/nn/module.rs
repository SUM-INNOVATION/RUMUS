//! The core `Module` trait for neural network layers.
//!
//! `Module` handles **state management** only: parameter collection,
//! train/eval mode toggling, and serialization via state dictionaries.
//! `forward()` is deliberately **not** in this trait because different
//! layers have different forward signatures.

use std::collections::HashMap;

use crate::autograd::AutogradError;
use crate::nn::Parameter;
use crate::tensor::Tensor;

/// Core trait for neural network state management.
///
/// The `#[derive(Module)]` macro auto-generates this impl by delegating
/// to each field's `Module` implementation.
pub trait Module {
    /// Return clones of all learnable parameters.
    fn parameters(&self) -> Vec<Parameter>;

    /// Switch to training mode.
    fn train(&mut self) {}

    /// Switch to evaluation mode.
    fn eval(&mut self) {}

    /// Serialize all parameters into a flat `name → Tensor` map.
    ///
    /// `prefix` is the dot-path accumulated by the parent module.
    /// For the root call, pass `""`.
    ///
    /// # Example output
    ///
    /// For an MLP with two linear layers:
    /// ```text
    /// "linear1.weight" → Tensor [2, 8]
    /// "linear1.bias"   → Tensor [8]
    /// "linear2.weight" → Tensor [8, 1]
    /// "linear2.bias"   → Tensor [1]
    /// ```
    fn state_dict(&self, prefix: &str) -> HashMap<String, Tensor>;

    /// Load parameters from a flat `name → Tensor` map.
    ///
    /// For each key matching this module's prefix, the parameter's
    /// storage is overwritten via the `RwLock` write guard.
    ///
    /// Missing keys are silently skipped to support partial loading
    /// (e.g., fine-tuning only the head of a pretrained model).
    /// Shape mismatches return [`AutogradError::StateError`].
    fn load_state_dict(
        &mut self,
        dict: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), AutogradError>;
}

/// Blanket extension trait for moving module parameters to the GPU.
///
/// Automatically implemented for any type that implements [`Module`].
/// Calling `to_gpu()` pushes all parameter tensors to the GPU via H2D
/// transfer.
#[cfg(feature = "gpu")]
pub trait ModuleToGpu: Module {
    /// Push all parameters to the GPU.
    ///
    /// Triggers H2D transfers for any CPU-only parameters.
    /// No-op for parameters already on the GPU.
    fn to_gpu(&self) {
        for param in self.parameters() {
            param.tensor.to_gpu();
        }
    }
}

#[cfg(feature = "gpu")]
impl<T: Module> ModuleToGpu for T {}
