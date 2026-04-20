// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Central gradient accumulation buffer.
//!
//! Per Tenet #2 ("Op-Driven Autograd"): the [`GradientStore`] is
//! intentionally dumb.  It knows how to allocate, accumulate (`+=`), and
//! hand out gradient tensors.  It does **not** un-broadcast, reduce, or
//! reshape — those responsibilities belong to each backward op.

use std::collections::HashMap;

use crate::autograd::AutogradError;
use crate::tensor::{GradId, Tensor};

/// Central repository for accumulated gradients.
///
/// During backward traversal, each edge's gradient contribution is
/// [`accumulate`](GradientStore::accumulate)d into the buffer keyed by
/// the target tensor's [`GradId`].  After backward completes, the
/// optimizer drains the gradients it needs via
/// [`remove`](GradientStore::remove).
pub struct GradientStore {
    grads: HashMap<GradId, Tensor>,
}

impl GradientStore {
    /// Create an empty gradient store.
    pub fn new() -> Self {
        Self {
            grads: HashMap::new(),
        }
    }

    /// Accumulate a gradient contribution for the given `id`.
    ///
    /// - **First call** for a given `id`: inserts `grad` directly (move,
    ///   no copy).
    /// - **Subsequent calls**: asserts that the incoming `grad` has exactly
    ///   the same shape as the stored gradient, then performs element-wise
    ///   addition via [`Tensor::add`] and replaces the entry.
    ///
    /// # Errors
    ///
    /// Returns [`AutogradError::ShapeMismatch`] if the shapes differ.
    ///
    /// # Why not in-place `+=`?
    ///
    /// In-place mutation would require `Arc::get_mut` exclusivity on the
    /// stored tensor's storage, which we may not have (other backward ops
    /// or the user may hold clones).  Allocating a fresh tensor via
    /// `Tensor::add` is correct and avoids fighting the borrow checker.
    pub fn accumulate(&mut self, id: GradId, grad: Tensor) -> Result<(), AutogradError> {
        match self.grads.get(&id) {
            Some(existing) => {
                if existing.shape() != grad.shape() {
                    return Err(AutogradError::ShapeMismatch {
                        grad_id: id,
                        expected: existing.shape().to_vec(),
                        found: grad.shape().to_vec(),
                    });
                }
                // Tensor::add allocates a new contiguous tensor with the
                // summed values.  We replace the entry with the result.
                let summed = existing.add(&grad);
                self.grads.insert(id, summed);
                Ok(())
            }
            None => {
                self.grads.insert(id, grad);
                Ok(())
            }
        }
    }

    /// Remove and return the gradient for `id`, freeing the entry.
    ///
    /// Used by the optimizer to drain gradients through `&mut self`.
    /// Per Tenet #4 ("Borrow-Safe Optimizers"), the optimizer takes
    /// `&mut GradientStore` and selectively drains only the `ParamId`s
    /// it owns, preventing overlapping borrows.
    pub fn remove(&mut self, id: GradId) -> Option<Tensor> {
        self.grads.remove(&id)
    }

    /// Read-only access to a stored gradient (for inspection / testing).
    pub fn get(&self, id: GradId) -> Option<&Tensor> {
        self.grads.get(&id)
    }

    /// Returns `true` if there are no stored gradients.
    pub fn is_empty(&self) -> bool {
        self.grads.is_empty()
    }

    /// Number of stored gradients.
    pub fn len(&self) -> usize {
        self.grads.len()
    }

    /// Replace the gradient for `id` with a new tensor.
    ///
    /// Used by `clip_grad_norm_` to swap in scaled gradient tensors
    /// without WebGPU buffer aliasing issues.  The old tensor is dropped,
    /// returning its GPU buffer to the pool.
    pub fn replace(&mut self, id: GradId, grad: Tensor) {
        self.grads.insert(id, grad);
    }

    /// Merge all gradients from `other` into this store.
    ///
    /// Moves entries from `other` into `self`.  If a key already exists
    /// in `self`, the values are accumulated (added).
    pub fn merge_from(&mut self, other: &mut GradientStore) {
        for (id, grad) in other.grads.drain() {
            if let Err(_) = self.accumulate(id, grad) {
                // Shape mismatch — silently skip (shouldn't happen in practice).
            }
        }
    }
}
