//! Backward operation structs and the version-checking snapshot.
//!
//! Each struct captures the minimal data needed to compute gradients for
//! its corresponding forward op.  No opaque closures — every backward op
//! is a concrete, inspectable type that is `Send + Sync` by construction.

use crate::autograd::AutogradError;
use crate::tensor::{GradId, Layout, StorageHandle, WeakStorageHandle};

// ---------------------------------------------------------------------------
// VersionSnapshot — weak-reference version checker
// ---------------------------------------------------------------------------

/// Snapshot of a [`StorageHandle`]'s version counter at tape-record time.
///
/// Holds a [`WeakStorageHandle`] so recording does **not** keep intermediate
/// tensor memory alive.
///
/// - **Upgrade succeeds:** compare live version vs recorded.  Mismatch →
///   [`AutogradError::VersionMismatch`].
/// - **Upgrade fails:** dead tensor → provably unmutated → `Ok(())`.
#[derive(Debug, Clone)]
pub struct VersionSnapshot {
    pub grad_id: GradId,
    pub weak_storage: WeakStorageHandle,
    pub recorded_version: usize,
}

impl VersionSnapshot {
    pub fn new(grad_id: GradId, storage: &StorageHandle) -> Self {
        Self {
            grad_id,
            recorded_version: storage.version(),
            weak_storage: storage.downgrade(),
        }
    }

    pub fn check(&self) -> Result<(), AutogradError> {
        match self.weak_storage.upgrade() {
            Some(strong) => {
                let current = strong.version();
                if current != self.recorded_version {
                    Err(AutogradError::VersionMismatch {
                        grad_id: self.grad_id,
                        expected: self.recorded_version,
                        found: current,
                    })
                } else {
                    Ok(())
                }
            }
            None => Ok(()),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-op backward structs
// ---------------------------------------------------------------------------

/// Backward for `c = a + b`.
///
/// `∂L/∂a = ∂L/∂c`,  `∂L/∂b = ∂L/∂c`  (identity).
#[derive(Debug)]
pub struct AddBackward {
    pub lhs_version: VersionSnapshot,
    pub rhs_version: VersionSnapshot,
}

/// Backward for `c = a - b`.
///
/// `∂L/∂a = ∂L/∂c`,  `∂L/∂b = -∂L/∂c`.
#[derive(Debug)]
pub struct SubBackward {
    pub lhs_version: VersionSnapshot,
    pub rhs_version: VersionSnapshot,
}

/// Backward for `c = a * b` (element-wise).
///
/// `∂L/∂a = ∂L/∂c ⊙ b`,  `∂L/∂b = ∂L/∂c ⊙ a`.
#[derive(Debug)]
pub struct MulBackward {
    pub lhs_storage: StorageHandle,
    pub lhs_layout: Layout,
    pub lhs_version: VersionSnapshot,
    pub rhs_storage: StorageHandle,
    pub rhs_layout: Layout,
    pub rhs_version: VersionSnapshot,
}

/// Backward for `C = A @ B`.
///
/// `∂L/∂A = ∂L/∂C @ Bᵀ`,  `∂L/∂B = Aᵀ @ ∂L/∂C`.
#[derive(Debug)]
pub struct MatmulBackward {
    pub lhs_storage: StorageHandle,
    pub lhs_layout: Layout,
    pub lhs_version: VersionSnapshot,
    pub rhs_storage: StorageHandle,
    pub rhs_layout: Layout,
    pub rhs_version: VersionSnapshot,
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

/// Backward for `y = relu(x)`.
///
/// `∂L/∂x[i] = ∂L/∂y[i]  if x[i] > 0,  else 0`.
#[derive(Debug)]
pub struct ReluBackward {
    pub input_storage: StorageHandle,
    pub input_layout: Layout,
    pub input_version: VersionSnapshot,
}

/// Backward for `loss = mse_loss(pred, target)` (fused).
///
/// `∂L/∂pred[i] = out_grad_scalar * 2 * (pred[i] - target[i]) / N`.
///
/// Only `pred` receives a gradient; `target` is treated as a constant.
#[derive(Debug)]
pub struct MseLossBackward {
    pub pred_storage: StorageHandle,
    pub pred_layout: Layout,
    pub pred_version: VersionSnapshot,
    pub target_storage: StorageHandle,
    pub target_layout: Layout,
    pub target_version: VersionSnapshot,
    pub numel: usize,
}

/// Backward for `y = add_bias(matrix, bias)`.
///
/// `∂L/∂matrix = ∂L/∂y`  (identity, same shape `[m,n]`).
/// `∂L/∂bias = sum_rows(∂L/∂y)`  (reduce `[m,n]` → `[n]`).
#[derive(Debug)]
pub struct AddBiasBackward {
    pub input_version: VersionSnapshot,
    pub bias_version: VersionSnapshot,
    pub m: usize,
    pub n: usize,
}

// ---------------------------------------------------------------------------
// BackwardOp enum
// ---------------------------------------------------------------------------

/// Discriminated union of all backward operation types.
///
/// No closures, no trait objects — `Send + Sync` and inspectable.
#[derive(Debug)]
pub enum BackwardOp {
    Add(AddBackward),
    Sub(SubBackward),
    Mul(MulBackward),
    Matmul(MatmulBackward),
    Relu(ReluBackward),
    MseLoss(MseLossBackward),
    AddBias(AddBiasBackward),
}

const _: () = {
    fn _assert_send<T: Send>() {}
    fn _assert_sync<T: Sync>() {}
    fn _assertions() {
        _assert_send::<BackwardOp>();
        _assert_sync::<BackwardOp>();
    }
};
