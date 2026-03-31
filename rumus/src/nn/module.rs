//! The core `Module` trait for neural network layers.
//!
//! `Module` handles **state management** only: parameter collection and
//! train/eval mode toggling.  `forward()` is deliberately **not** in
//! this trait because different layers have different forward signatures
//! (Linear takes 1 tensor, Attention takes 3, Loss takes 2, etc.).
//! Users write `forward()` as an inherent method on their struct.

use crate::nn::Parameter;

/// Core trait for neural network state management.
///
/// The `#[derive(Module)]` macro auto-generates this impl by delegating
/// to each field's `Module` implementation.
pub trait Module {
    /// Return clones of all learnable parameters.
    ///
    /// The returned `Parameter`s share storage with the module's internal
    /// state (`Arc` refcount bump), so the optimizer can mutate weights
    /// through the `RwLock` write guard.
    fn parameters(&self) -> Vec<Parameter>;

    /// Switch to training mode.
    fn train(&mut self) {}

    /// Switch to evaluation mode.
    fn eval(&mut self) {}
}
