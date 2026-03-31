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
/// Holds a [`WeakStorageHandle`] (not a strong one) so that recording an
/// operation does **not** artificially keep intermediate tensor memory
/// alive.  This is critical for ops like [`AddBackward`] that only need
/// to verify the version counter, not read the data.
///
/// # Check semantics
///
/// During backward, [`check`](VersionSnapshot::check) attempts to upgrade
/// the weak reference:
///
/// - **Upgrade succeeds:** compare the live version counter against the
///   recorded snapshot.  A mismatch means the user mutated the tensor
///   in-place after it was recorded — return
///   [`AutogradError::VersionMismatch`].
///
/// - **Upgrade fails (storage was dropped):** the tensor is dead.  A dead
///   tensor cannot have been mutated in-place since the recording, so the
///   version is trivially valid.  Return `Ok(())`.
#[derive(Debug, Clone)]
pub struct VersionSnapshot {
    /// Which tensor this snapshot belongs to (for error reporting).
    pub grad_id: GradId,
    /// Weak reference to the storage — does not keep the data alive.
    pub weak_storage: WeakStorageHandle,
    /// The version counter's value at the moment the op was recorded.
    pub recorded_version: usize,
}

impl VersionSnapshot {
    /// Create a snapshot from a live [`StorageHandle`].
    ///
    /// Reads the current version with `Acquire` ordering and immediately
    /// downgrades the strong reference to a weak one.
    pub fn new(grad_id: GradId, storage: &StorageHandle) -> Self {
        Self {
            grad_id,
            recorded_version: storage.version(),
            weak_storage: storage.downgrade(),
        }
    }

    /// Verify that the storage has not been mutated since recording.
    ///
    /// See the struct-level docs for the upgrade-or-dead semantics.
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
            // Storage has been dropped — it cannot have been mutated.
            None => Ok(()),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-op backward structs
// ---------------------------------------------------------------------------

/// Backward for element-wise addition: `c = a + b`.
///
/// ```text
/// ∂L/∂a = ∂L/∂c   (identity — gradient passes through)
/// ∂L/∂b = ∂L/∂c   (identity — gradient passes through)
/// ```
///
/// No saved tensor data is needed — the gradient is simply forwarded to
/// both inputs.  Only version snapshots are stored to detect illegal
/// in-place mutation.
///
/// When broadcasting is added, this struct will carry
/// `lhs_broadcast: Option<BroadcastInfo>` and
/// `rhs_broadcast: Option<BroadcastInfo>` indicating which axes were
/// expanded, so the backward pass can sum-reduce `grad_c` along those
/// axes before propagating upstream.
#[derive(Debug)]
pub struct AddBackward {
    /// Version check for input `a`.
    pub lhs_version: VersionSnapshot,
    /// Version check for input `b`.
    pub rhs_version: VersionSnapshot,
}

/// Backward for element-wise multiplication: `c = a * b`.
///
/// ```text
/// ∂L/∂a = ∂L/∂c ⊙ b   (element-wise)
/// ∂L/∂b = ∂L/∂c ⊙ a   (element-wise)
/// ```
///
/// Both input values are needed to compute the other's gradient, so we
/// save strong [`StorageHandle`]s and [`Layout`]s for both.  These strong
/// references intentionally keep the data alive through backward — unlike
/// [`AddBackward`], `MulBackward` genuinely needs to read the data.
///
/// We save `StorageHandle` + `Layout` rather than a full `Tensor` to
/// avoid dragging `AutogradState` metadata into the saved context, which
/// would create confusing ownership of graph tracking state.
#[derive(Debug)]
pub struct MulBackward {
    /// Saved value of input `a` — storage for data access.
    pub lhs_storage: StorageHandle,
    /// Layout of input `a` at record time.
    pub lhs_layout: Layout,
    /// Version check for input `a`.
    pub lhs_version: VersionSnapshot,

    /// Saved value of input `b` — storage for data access.
    pub rhs_storage: StorageHandle,
    /// Layout of input `b` at record time.
    pub rhs_layout: Layout,
    /// Version check for input `b`.
    pub rhs_version: VersionSnapshot,
}

/// Backward for matrix multiplication: `C = A @ B`.
///
/// For `A: (m × k)` and `B: (k × n)`:
///
/// ```text
/// ∂L/∂A = ∂L/∂C @ Bᵀ     shape: (m×n) @ (n×k) → (m×k)
/// ∂L/∂B = Aᵀ @ ∂L/∂C     shape: (k×m) @ (m×n) → (k×n)
/// ```
///
/// Both input values are saved (strong references).  The transpose
/// operations in the backward pass are zero-copy view ops on the saved
/// storages — `Tensor::transpose` just swaps strides.
#[derive(Debug)]
pub struct MatmulBackward {
    /// Saved value of input `A` — storage for data access.
    pub lhs_storage: StorageHandle,
    /// Layout of input `A` at record time.
    pub lhs_layout: Layout,
    /// Version check for input `A`.
    pub lhs_version: VersionSnapshot,

    /// Saved value of input `B` — storage for data access.
    pub rhs_storage: StorageHandle,
    /// Layout of input `B` at record time.
    pub rhs_layout: Layout,
    /// Version check for input `B`.
    pub rhs_version: VersionSnapshot,

    /// Row count of A (and C).
    pub m: usize,
    /// Shared (inner) dimension.
    pub k: usize,
    /// Column count of B (and C).
    pub n: usize,
}

// ---------------------------------------------------------------------------
// BackwardOp enum
// ---------------------------------------------------------------------------

/// Discriminated union of all backward operation types.
///
/// Replaces opaque `Box<dyn Fn(...)>` closures, giving us:
///
/// - **Strict `Send + Sync` safety** — all fields are concrete types, no
///   trait objects with hidden captured state.
/// - **Inspectability** — `match` on the variant to see exactly which op
///   produced a graph node (essential for `tape.dump_graph()`).
/// - **Deterministic serialisation** — no function pointers to serialise.
#[derive(Debug)]
pub enum BackwardOp {
    Add(AddBackward),
    Mul(MulBackward),
    Matmul(MatmulBackward),
}

// All backward op structs are Send + Sync by construction (they contain
// only Arc-backed handles, atomics, Vec<usize>, and usize).  Assert this
// at compile time so a future field addition that breaks the invariant is
// caught immediately.
const _: () = {
    fn _assert_send<T: Send>() {}
    fn _assert_sync<T: Sync>() {}
    fn _assertions() {
        _assert_send::<VersionSnapshot>();
        _assert_sync::<VersionSnapshot>();
        _assert_send::<AddBackward>();
        _assert_sync::<AddBackward>();
        _assert_send::<MulBackward>();
        _assert_sync::<MulBackward>();
        _assert_send::<MatmulBackward>();
        _assert_sync::<MatmulBackward>();
        _assert_send::<BackwardOp>();
        _assert_sync::<BackwardOp>();
    }
};
