// SPDX-License-Identifier: Apache-2.0 OR MIT
//! `rumus-graph` — Sparse graph engine for GNNs.
//!
//! Provides a WebGPU-native fused Sparse-Dense Matrix Multiplication (SpMM)
//! kernel for differentiable message passing on graphs.
//!
//! # Example
//!
//! ```ignore
//! use rumus_graph::Graph;
//! use rumus::tensor::Tensor;
//!
//! let graph = Graph::new(&src_nodes, &dst_nodes, None, num_nodes);
//! let features = Tensor::new(feat_data, vec![num_nodes, hidden_dim]);
//! let output = graph.spmm(&features);  // differentiable!
//! ```

pub mod graph;
pub(crate) mod ops;

pub use graph::{Graph, SparseTensor};
