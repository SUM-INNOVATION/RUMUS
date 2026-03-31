//! Optimizers for updating learnable parameters.
//!
//! Per Tenet #4 ("Borrow-Safe Optimizers"): [`Optimizer::step`] takes
//! `&mut GradientStore` and selectively drains only the gradients belonging
//! to its registered parameters.  Multiple optimizers can each drain their
//! subset from the same store (e.g., GAN generator vs. discriminator).

mod adam;
mod sgd;

pub use adam::Adam;
pub use sgd::SGD;

use crate::autograd::{AutogradError, GradientStore};

/// Trait for all optimizers.
pub trait Optimizer {
    /// Apply one optimization step using gradients from the store.
    ///
    /// The optimizer iterates over its registered parameters, calls
    /// `grads.remove(param.grad_id())` to drain each gradient, and applies
    /// the weight update in-place via the `RwLock` write guard on the
    /// parameter's storage.
    ///
    /// # Errors
    ///
    /// Returns [`AutogradError::MissingGrad`] if a required gradient is
    /// absent from the store.
    fn step(&mut self, grads: &mut GradientStore) -> Result<(), AutogradError>;

    /// Reset optimizer state between epochs if needed.
    ///
    /// For SGD without momentum this is a no-op.  For Adam it can
    /// optionally reset the moment estimates (rarely used in practice).
    fn zero_grad(&mut self) {}
}
