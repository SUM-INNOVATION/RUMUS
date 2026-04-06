// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Loss functions.

use crate::tensor::Tensor;

/// Mean squared error loss: `sum((pred - target)^2) / N`.
///
/// Returns a scalar tensor (shape `[1]`).  Only `pred` receives a
/// gradient; `target` is treated as a constant.
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    pred.mse_loss(target)
}

/// Cross-entropy loss with Log-Sum-Exp numerical stability.
///
/// `logits`: `[B, C]` — raw unnormalized scores.
/// `targets`: `[B]` — integer class indices (stored as f32).
///
/// Returns a scalar tensor (shape `[1]`).  The gradient (softmax - one_hot,
/// scaled by 1/B) is pre-computed during the forward pass for efficiency.
pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
    logits.cross_entropy_loss(targets)
}
