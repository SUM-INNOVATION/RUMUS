// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Tensor Parallelism: ColumnParallelLinear and RowParallelLinear.

use std::sync::Arc;

use rumus::nn::Parameter;
use rumus::tensor::Tensor;

use crate::collective::CollectiveBarrier;

// ---------------------------------------------------------------------------
// ColumnParallelLinear
// ---------------------------------------------------------------------------

/// Linear layer with weight sharded along columns (N dimension).
///
/// Forward: `Y_t = X @ W_t` (no collective).
/// Backward: `grad_X = AllReduce(grad_Y_t @ W_t^T)` — handled by normal
///           autograd + external AllReduce call.
pub struct ColumnParallelLinear {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
    pub rank: usize,
    pub world_size: usize,
    pub barrier: Arc<CollectiveBarrier>,
}

impl ColumnParallelLinear {
    /// Forward: Y_t = X @ W_t (+ bias).  No collective in forward.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let y = x.matmul(&self.weight.tensor);
        match &self.bias {
            Some(b) => y.add_bias(&b.tensor),
            None => y,
        }
    }

    /// AllReduce grad_X after backward (called explicitly by the user/executor).
    pub fn allreduce_grad_x(&self, grad_x: &Tensor) -> Tensor {
        let data = {
            let g = grad_x.data();
            g.to_vec()
        };
        let reduced = self.barrier.reduce(data);
        let t = Tensor::new(reduced, grad_x.shape().to_vec());
        t.to_gpu();
        t
    }
}

// ---------------------------------------------------------------------------
// RowParallelLinear
// ---------------------------------------------------------------------------

/// Linear layer with weight sharded along rows (K dimension).
///
/// Forward: `Y_t = X_t @ W_t` (partial sum), then `Y = AllReduce(Y_t)`.
/// Backward: `grad_X_t = grad_Y @ W_t^T` (no collective).
pub struct RowParallelLinear {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
    pub rank: usize,
    pub world_size: usize,
    pub barrier: Arc<CollectiveBarrier>,
}

impl RowParallelLinear {
    /// Forward: Y_t = X_t @ W_t, then AllReduce → Y.
    pub fn forward(&self, x_t: &Tensor) -> Tensor {
        let y_partial = x_t.matmul(&self.weight.tensor);

        // AllReduce the partial sums via CPU staging.
        let data = {
            let g = y_partial.data();
            g.to_vec()
        };
        let reduced = self.barrier.reduce(data);
        let y = Tensor::new(reduced, y_partial.shape().to_vec());
        y.to_gpu();

        match &self.bias {
            Some(b) if self.rank == 0 => y.add_bias(&b.tensor),
            _ => y,
        }
    }
}
