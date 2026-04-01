//! Lightweight training loop orchestrator.
//!
//! The [`Trainer`] struct wraps an [`Optimizer`] and provides a
//! `train_step()` method that executes one forward-backward-update cycle.
//! The forward pass is supplied as a closure, so the trainer is agnostic
//! to the model architecture and loss function.

use crate::autograd::{self, AutogradError};
use crate::optim::Optimizer;
use crate::tensor::Tensor;

/// Lightweight training loop orchestrator.
///
/// Does not own the model — the user retains full control.  Generic over
/// any [`Optimizer`] (Adam, AdamW, SGD).
///
/// # Example
///
/// ```ignore
/// let mut trainer = Trainer::new(AdamW::new(model.parameters(), 0.01));
///
/// for epoch in 0..100 {
///     trainer.reset_epoch();
///     let loss = trainer.train_step(|| {
///         let logits = model.forward(&batch_x);
///         nn::cross_entropy_loss(&logits, &batch_y)
///     }).unwrap();
///     println!("epoch {} loss: {:.4}", epoch, trainer.epoch_avg_loss());
/// }
/// ```
pub struct Trainer<O: Optimizer> {
    optimizer: O,
    epoch_loss: f32,
    epoch_steps: usize,
}

impl<O: Optimizer> Trainer<O> {
    /// Create a new trainer wrapping the given optimizer.
    pub fn new(optimizer: O) -> Self {
        Self {
            optimizer,
            epoch_loss: 0.0,
            epoch_steps: 0,
        }
    }

    /// Execute one training step:
    ///
    /// 1. Call `forward_fn()` — must return a **scalar** loss tensor.
    /// 2. Read the loss value (4-byte D2H for logging).
    /// 3. `backward()` — builds gradients via Kahn's algorithm.
    /// 4. `optimizer.step()` — applies weight updates (drains the store).
    ///
    /// Returns the step loss as `f32`.
    pub fn train_step<F>(&mut self, forward_fn: F) -> Result<f32, AutogradError>
    where
        F: FnOnce() -> Tensor,
    {
        let loss = forward_fn();

        // 4-byte D2H: read the scalar loss for logging.
        let loss_val = {
            let g = loss.data();
            g[0]
        };

        let mut grads = autograd::backward(&loss)?;
        self.optimizer.step(&mut grads)?;

        self.epoch_loss += loss_val;
        self.epoch_steps += 1;

        Ok(loss_val)
    }

    /// Average loss accumulated since the last `reset_epoch()`.
    pub fn epoch_avg_loss(&self) -> f32 {
        if self.epoch_steps == 0 {
            0.0
        } else {
            self.epoch_loss / self.epoch_steps as f32
        }
    }

    /// Reset epoch-level counters.  Call at the start of each epoch.
    pub fn reset_epoch(&mut self) {
        self.epoch_loss = 0.0;
        self.epoch_steps = 0;
    }

    /// Mutable reference to the inner optimizer (for hyperparameter tuning).
    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }
}
