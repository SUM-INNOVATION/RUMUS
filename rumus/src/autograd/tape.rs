// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Append-only Wengert list (tape) for recording the computational graph.
//!
//! Operations are recorded in forward-execution order via [`Tape::push`].
//! During backward, the tape is traversed using Kahn's algorithm (built
//! in a subsequent chunk) with the edge counts stored in
//! [`TensorMeta::total_grads`](crate::tensor::TensorMeta::total_grads).

use crate::autograd::BackwardOp;
use crate::tensor::{GradId, OpId};

// ---------------------------------------------------------------------------
// TapeEntry
// ---------------------------------------------------------------------------

/// A single node in the computational graph.
///
/// Records which backward operation produced which outputs from which
/// inputs, along with the [`BackwardOp`] that knows how to compute
/// gradients.
///
/// # Edge semantics
///
/// - `inputs` lists the [`GradId`]s of the tensors this op **consumed**.
///   During backward, computed gradients are propagated **to** these ids.
///
/// - `outputs` lists the [`GradId`]s of the tensors this op **produced**.
///   During backward, the incoming gradient is read **from** these ids.
///
/// Per Tenet #3 ("Strict Graph Edge Counting"): if a tensor is used twice
/// as an input to a single op, its `GradId` appears twice in `inputs`,
/// and its `TensorMeta::total_grads` was incremented by two at record
/// time.
#[derive(Debug)]
pub struct TapeEntry {
    /// The backward computation to run during backward traversal.
    pub op: BackwardOp,

    /// GradIds of this op's inputs (gradients flow **to** these).
    pub inputs: Vec<GradId>,

    /// GradIds of this op's outputs (gradients are read **from** these).
    pub outputs: Vec<GradId>,
}

// ---------------------------------------------------------------------------
// Tape
// ---------------------------------------------------------------------------

/// Append-only Wengert list of recorded operations.
///
/// The tape is the backbone of the autograd engine.  During the forward
/// pass, every differentiable operation appends a [`TapeEntry`].  During
/// backward, the engine reads entries (by [`OpId`] index) to determine
/// what gradient math to run and where to propagate the results.
///
/// The tape is intentionally **not** thread-safe — it is owned by a single
/// forward-pass context.  If we need parallel forward passes in the future,
/// each will own its own tape.
#[derive(Debug)]
pub struct Tape {
    entries: Vec<TapeEntry>,
}

impl Tape {
    /// Create an empty tape.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Record an operation on the tape.
    ///
    /// Returns the [`OpId`] (index) of the new entry, which is stored in
    /// the output tensor's [`TensorMeta::creator`] field so the backward
    /// engine can look up the entry later.
    pub fn push(&mut self, entry: TapeEntry) -> OpId {
        let id = OpId(self.entries.len());
        self.entries.push(entry);
        id
    }

    /// Number of recorded operations.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no operations have been recorded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Read-only access to an entry by [`OpId`].
    ///
    /// Returns `None` if the id is out of range.
    pub fn get(&self, id: OpId) -> Option<&TapeEntry> {
        self.entries.get(id.0)
    }

    /// Consume the tape and return the entries as a `Vec`.
    ///
    /// Used by the backward engine: it takes ownership of the tape (via
    /// [`context::take_tape`](crate::autograd::context::take_tape)), then
    /// consumes it into entries to iterate in reverse.  This prevents
    /// accidental double-backward — the tape is gone after this call.
    pub fn into_entries(self) -> Vec<TapeEntry> {
        self.entries
    }
}
