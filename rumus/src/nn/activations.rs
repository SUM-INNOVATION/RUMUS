//! Activation functions.

use crate::tensor::Tensor;

/// Element-wise ReLU activation: `max(0, x)`.
///
/// This is a free function, not a `Module`, because ReLU has no learnable
/// parameters and no state.
pub fn relu(input: &Tensor) -> Tensor {
    input.relu()
}
