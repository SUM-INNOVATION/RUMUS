// SPDX-License-Identifier: Apache-2.0 OR MIT
//! ONNX model export via graph tracing.
//!
//! Feature-gated behind `--features onnx`.

pub mod export;
pub mod proto;
pub mod tracer;

pub use export::{export_onnx, export_onnx_with_opset, OnnxError};
pub use tracer::{TracedAttribute, TracedGraph, TracedNode, ValueInfo};
