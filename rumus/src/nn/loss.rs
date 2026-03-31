//! Loss functions.

use crate::tensor::Tensor;

/// Mean squared error loss: `sum((pred - target)^2) / N`.
///
/// Returns a scalar tensor (shape `[1]`).  Only `pred` receives a
/// gradient; `target` is treated as a constant.
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    pred.mse_loss(target)
}
