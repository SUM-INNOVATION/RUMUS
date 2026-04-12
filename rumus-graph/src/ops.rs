// SPDX-License-Identifier: Apache-2.0 OR MIT
//! SpMM operations using the RUMUS plugin API (CustomOp / CustomBackward).

use std::sync::Arc;

use rumus::autograd::CustomBackward;
use rumus::ext::{self, CustomOp};
use rumus::tensor::Tensor;

use crate::graph::Graph;

/// WGSL source for the SpMM kernel (embedded at compile time).
const SPMM_WGSL: &str = include_str!("spmm.wgsl");

// ---------------------------------------------------------------------------
// SpMMOp — CustomOp for forward message passing
// ---------------------------------------------------------------------------

pub(crate) struct SpMMOp {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub hidden_dim: usize,
    pub has_values: bool,
}

impl CustomOp for SpMMOp {
    fn op_name(&self) -> &str {
        "rumus_graph_spmm"
    }

    fn wgsl_source(&self) -> &str {
        SPMM_WGSL
    }

    fn entry_point(&self) -> &str {
        "spmm_forward_kernel"
    }

    fn num_inputs(&self) -> usize {
        4 // row_ptr, col_indices, values, features
    }

    fn output_shape(&self, input_shapes: &[&[usize]]) -> Vec<usize> {
        // features is input[3] with shape [N, D].
        // Output is [N, D].
        vec![self.num_nodes, self.hidden_dim]
    }

    fn dispatch(&self, _output_numel: usize) -> (u32, u32, u32) {
        // 1 thread per node.
        ((self.num_nodes as u32 + 255) / 256, 1, 1)
    }

    fn uniform_data(&self, _inputs: &[&Tensor]) -> Vec<u8> {
        let mut buf = vec![0u8; 16];
        buf[0..4].copy_from_slice(&(self.num_nodes as u32).to_le_bytes());
        buf[4..8].copy_from_slice(&(self.num_edges as u32).to_le_bytes());
        buf[8..12].copy_from_slice(&(self.hidden_dim as u32).to_le_bytes());
        buf[12..16].copy_from_slice(&(if self.has_values { 1u32 } else { 0u32 }).to_le_bytes());
        buf
    }

    fn backward_handler(&self) -> Option<Arc<dyn CustomBackward>> {
        // The backward handler is set per-call in spmm_forward, not here.
        // This default returns None; the actual handler is injected via
        // a wrapper that holds the transposed graph.
        None
    }

    fn save_for_backward<'a>(
        &self,
        inputs: &[&'a Tensor],
        _output: &'a Tensor,
    ) -> Vec<&'a Tensor> {
        // Save the features tensor for potential future use (grad_values).
        vec![inputs[3]]
    }
}

// ---------------------------------------------------------------------------
// SpMMOpWithBackward — wraps SpMMOp and attaches the backward handler
// ---------------------------------------------------------------------------

/// SpMMOp variant that carries the backward handler referencing the
/// transposed graph.
pub(crate) struct SpMMOpWithBackward {
    pub inner: SpMMOp,
    pub backward: Arc<SpMMBackward>,
}

impl CustomOp for SpMMOpWithBackward {
    fn op_name(&self) -> &str { self.inner.op_name() }
    fn wgsl_source(&self) -> &str { self.inner.wgsl_source() }
    fn entry_point(&self) -> &str { self.inner.entry_point() }
    fn num_inputs(&self) -> usize { self.inner.num_inputs() }
    fn output_shape(&self, shapes: &[&[usize]]) -> Vec<usize> { self.inner.output_shape(shapes) }
    fn dispatch(&self, n: usize) -> (u32, u32, u32) { self.inner.dispatch(n) }
    fn uniform_data(&self, inputs: &[&Tensor]) -> Vec<u8> { self.inner.uniform_data(inputs) }

    fn backward_handler(&self) -> Option<Arc<dyn CustomBackward>> {
        Some(self.backward.clone() as Arc<dyn CustomBackward>)
    }

    fn save_for_backward<'a>(
        &self,
        inputs: &[&'a Tensor],
        output: &'a Tensor,
    ) -> Vec<&'a Tensor> {
        self.inner.save_for_backward(inputs, output)
    }
}

// ---------------------------------------------------------------------------
// SpMMBackward — gradient routing via transposed SpMM
// ---------------------------------------------------------------------------

/// Backward for SpMM: grad_features = SpMM(A^T, grad_output).
///
/// Uses the pre-computed transposed CSR graph for gradient routing.
/// No f32 atomics needed — pure SpMM on A^T.
#[derive(Debug)]
pub(crate) struct SpMMBackward {
    pub bwd_row_ptr: Tensor,
    pub bwd_col_idx: Tensor,
    pub bwd_values: Tensor,
    pub num_nodes: usize,
    pub num_edges: usize,
    pub hidden_dim: usize,
    pub has_values: bool,
}

impl CustomBackward for SpMMBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Vec<Tensor> {
        // grad_features = SpMM(A^T, grad_output)
        let op = SpMMOp {
            num_nodes: self.num_nodes,
            num_edges: self.num_edges,
            hidden_dim: self.hidden_dim,
            has_values: self.has_values,
        };

        let grad_features = ext::custom_forward(
            &op,
            &[
                &self.bwd_row_ptr,
                &self.bwd_col_idx,
                &self.bwd_values,
                grad_output,
            ],
        );

        // Return 4 gradients matching the 4 inputs:
        //   [0] row_ptr: no grad (structural)
        //   [1] col_indices: no grad (structural)
        //   [2] values: no grad (M21 scope)
        //   [3] features: grad_features
        let zero = Tensor::new(vec![0.0], vec![1]);
        vec![zero.clone(), zero.clone(), zero, grad_features]
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Execute differentiable SpMM: output[i] = Σ_{j ∈ N(i)} A[i,j] * features[j].
pub(crate) fn spmm_forward(graph: &Graph, features: &Tensor, hidden_dim: usize) -> Tensor {
    let backward = Arc::new(SpMMBackward {
        bwd_row_ptr: graph.bwd_row_ptr.clone(),
        bwd_col_idx: graph.bwd_col_idx.clone(),
        bwd_values: graph.bwd_values.clone(),
        num_nodes: graph.num_nodes,
        num_edges: graph.num_edges,
        hidden_dim,
        has_values: graph.has_values,
    });

    let op = SpMMOpWithBackward {
        inner: SpMMOp {
            num_nodes: graph.num_nodes,
            num_edges: graph.num_edges,
            hidden_dim,
            has_values: graph.has_values,
        },
        backward,
    };

    ext::custom_forward(
        &op,
        &[
            &graph.fwd_row_ptr,
            &graph.fwd_col_idx,
            &graph.fwd_values,
            features,
        ],
    )
}
