//! The core `Module` trait for neural network layers.

use crate::nn::Parameter;
use crate::tensor::Tensor;

/// Core trait for neural network layers.
///
/// Every layer that contains learnable parameters implements `Module`.
/// The trait is intentionally minimal — methods are added as needed
/// rather than front-loading a kitchen-sink interface.
pub trait Module {
    /// Run the forward pass.
    ///
    /// Takes `&self` because the forward pass is logically read-only on
    /// the weights.  Layers that need mutable state for forward (e.g.,
    /// BatchNorm running stats) use interior mutability (`Cell`/`RefCell`).
    fn forward(&self, input: &Tensor) -> Tensor;

    /// Return clones of all learnable parameters.
    ///
    /// The returned `Parameter`s share storage with the module's internal
    /// state (`Arc` refcount bump), so the optimizer can mutate weights
    /// through the `RwLock` write guard.
    fn parameters(&self) -> Vec<Parameter>;

    /// Switch to training mode (enables Dropout, BatchNorm stats, etc.).
    ///
    /// Default is a no-op — layers without mode-dependent behavior don't
    /// need to override.
    fn train(&mut self) {}

    /// Switch to evaluation mode.
    fn eval(&mut self) {}
}
