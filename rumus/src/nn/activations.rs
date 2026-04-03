//! Activation functions.

use crate::tensor::Tensor;

/// Element-wise ReLU: `max(0, x)`.
pub fn relu(input: &Tensor) -> Tensor {
    input.relu()
}

/// Element-wise Sigmoid: `1 / (1 + exp(-x))`.
pub fn sigmoid(input: &Tensor) -> Tensor {
    input.sigmoid()
}

/// Element-wise Tanh: `tanh(x)`.
pub fn tanh(input: &Tensor) -> Tensor {
    input.tanh_act()
}

/// Element-wise GELU (tanh approximation):
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
pub fn gelu(input: &Tensor) -> Tensor {
    input.gelu()
}

/// Element-wise Leaky ReLU: `max(alpha * x, x)`.
pub fn leaky_relu(input: &Tensor, alpha: f32) -> Tensor {
    input.leaky_relu(alpha)
}
