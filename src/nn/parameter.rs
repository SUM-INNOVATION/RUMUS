//! Learnable parameter with globally unique identity.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::tensor::{GradId, ParamId, Tensor};

/// Global monotonically increasing `ParamId` counter.
///
/// Thread-safe — multiple modules can be constructed concurrently.
/// `Relaxed` ordering is sufficient: uniqueness is guaranteed by the
/// atomic RMW itself; no cross-thread data publication depends on the
/// ordering of this counter.
static NEXT_PARAM_ID: AtomicUsize = AtomicUsize::new(0);

fn alloc_param_id() -> ParamId {
    ParamId(NEXT_PARAM_ID.fetch_add(1, Ordering::Relaxed))
}

/// A learnable parameter: a [`Tensor`] with persistent identity.
///
/// On construction, the tensor is automatically set to `requires_grad = true`
/// and assigned a fresh [`GradId`].  The [`ParamId`] is globally unique and
/// stable across forward passes — optimizers key their per-parameter state
/// (momentum, EMA) on it.
///
/// Cloning a `Parameter` is cheap: it clones the `Arc`-backed storage
/// (refcount bump) and the `Arc<TensorMeta>` (shared metadata).  The
/// optimizer and the module both see the same underlying data.
#[derive(Clone, Debug)]
pub struct Parameter {
    /// Globally unique identifier for this parameter.
    pub id: ParamId,
    /// The underlying tensor holding the parameter's value.
    pub tensor: Tensor,
}

impl Parameter {
    /// Create a new parameter from a tensor.
    ///
    /// Automatically:
    /// - Allocates a globally unique [`ParamId`].
    /// - Sets `requires_grad = true` (which allocates a [`GradId`]).
    pub fn new(mut tensor: Tensor) -> Self {
        tensor.set_requires_grad(true);
        Self {
            id: alloc_param_id(),
            tensor,
        }
    }

    /// The [`GradId`] of this parameter's tensor.
    ///
    /// Always `Some(...)` because the constructor enforces `requires_grad`.
    pub fn grad_id(&self) -> GradId {
        self.tensor
            .grad_id()
            .expect("Parameter tensor must have a GradId")
    }
}
