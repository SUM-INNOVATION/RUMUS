//! Neural network building blocks: parameters, modules, layers, losses, and IO.

mod activations;
mod io;
mod linear;
mod loss;
mod module;
mod parameter;

pub use activations::relu;
pub use io::{load_safetensors, save_safetensors};
pub use linear::Linear;
pub use loss::mse_loss;
pub use module::Module;
#[cfg(feature = "gpu")]
pub use module::ModuleToGpu;
pub use parameter::Parameter;

// Re-export the derive macro so users can write `use rumus::nn::Module;`
// and `#[derive(Module)]` in the same scope.
pub use rumus_macros::Module;
