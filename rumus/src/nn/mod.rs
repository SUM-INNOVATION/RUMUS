//! Neural network building blocks: parameters, modules, layers, losses, and IO.

mod activations;
mod adaptive_pool;
pub mod attention;
mod batch_norm;
mod conv;
mod conv_transpose;
mod dropout;
mod embedding;
mod flatten;
mod io;
mod layer_norm;
mod linear;
mod loss;
mod module;
mod parameter;
mod pool;
pub mod transformer;

pub use activations::{gelu, leaky_relu, relu, sigmoid, tanh};
pub use adaptive_pool::AdaptiveAvgPool2d;
pub use attention::{scaled_dot_product_attention, MultiheadAttention};
pub use batch_norm::BatchNorm2d;
pub use transformer::TransformerBlock;
pub use conv::Conv2d;
pub use conv_transpose::ConvTranspose2d;
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use flatten::Flatten;
pub use layer_norm::LayerNorm;
pub use io::{load_safetensors, save_safetensors};
pub use linear::Linear;
pub use loss::{cross_entropy_loss, mse_loss};
pub use pool::MaxPool2d;
pub use module::Module;
#[cfg(feature = "gpu")]
pub use module::ModuleToGpu;
pub use parameter::Parameter;

// Re-export the derive macro so users can write `use rumus::nn::Module;`
// and `#[derive(Module)]` in the same scope.
pub use rumus_macros::Module;
