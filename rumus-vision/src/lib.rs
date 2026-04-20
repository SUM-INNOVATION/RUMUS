// SPDX-License-Identifier: Apache-2.0 OR MIT
//! `rumus-vision` — Spatial CNN engine for RUMUS.
//!
//! Direct sliding-window convolution and max-pooling via WebGPU.
//! Zero im2col VRAM overhead.  Fully differentiable through the
//! M19 Custom Ops Plugin API.
//!
//! # Example
//!
//! ```ignore
//! use rumus_vision::ops;
//! use rumus::tensor::Tensor;
//!
//! let input = Tensor::new(vec![0.0; 1 * 3 * 32 * 32], vec![1, 3, 32, 32]);
//! let weight = Tensor::new(vec![0.01; 16 * 3 * 3 * 3], vec![16, 3, 3, 3]);
//! let output = ops::conv2d(&input, &weight, None, (1, 1), (1, 1), (1, 1));
//! // output shape: [1, 16, 32, 32]
//! ```

pub mod ops;
pub mod quant;

pub use ops::{conv2d, max_pool2d, ConvParams, PoolParams};
pub use quant::{QLinear, QuantizedTensor};
