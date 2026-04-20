// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Automatic differentiation engine for RUMUS.
//!
//! This module provides:
//!
//! - **[`Tape`]** — append-only Wengert list recording the computational graph.
//! - **[`GradientStore`]** — dumb accumulate-only gradient buffer.
//! - **[`BackwardOp`]** — concrete enum of backward operations (no closures).
//! - **[`backward`]** — Kahn's algorithm backward traversal.
//! - **[`context`]** — thread-local tape, GradId generator, and `no_grad` guard.
//!
//! # Design tenets enforced here
//!
//! - **No opaque closures:** backward ops are concrete enum variants,
//!   not `Box<dyn Fn>`.  `Send + Sync` and inspectability for free.
//! - **Dumb GradientStore:** accumulate-only; no un-broadcasting or
//!   layout logic.  Ops own their reduction math.
//! - **Errors over panics:** user-triggerable faults produce
//!   [`AutogradError`], not panics.

pub mod context;

mod backward;
mod backward_ops;
mod gradient_store;
mod tape;

pub use backward::{backward, backward_with_grad};
pub use backward_ops::*;
pub use gradient_store::*;
pub use tape::*;

use std::fmt;

use crate::tensor::GradId;

// ---------------------------------------------------------------------------
// AutogradError
// ---------------------------------------------------------------------------

/// Structured errors for user-triggerable autograd faults.
///
/// Per Tenet #8 ("Errors over Panics"): user-facing violations return these
/// instead of panicking.  Panics are strictly reserved for internal framework
/// invariant violations that indicate bugs in RUMUS itself.
#[derive(Debug, Clone)]
pub enum AutogradError {
    /// A tensor's storage was mutated in-place after it was recorded on the
    /// tape.  The version counter at record time does not match the current
    /// version.
    VersionMismatch {
        grad_id: GradId,
        expected: usize,
        found: usize,
    },

    /// A gradient's shape does not match the tensor it is being accumulated
    /// into.
    ShapeMismatch {
        grad_id: GradId,
        expected: Vec<usize>,
        found: Vec<usize>,
    },

    /// `backward()` was called on a tensor that has no autograd graph
    /// (`AutogradState::None` or no `creator`).
    NoGraph,

    /// A `GradId` referenced during backward was not found in the
    /// [`GradientStore`].
    MissingGrad { grad_id: GradId },

    /// Error during state dict load/save (shape mismatch, missing key, IO).
    StateError { key: String, message: String },
}

impl fmt::Display for AutogradError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AutogradError::VersionMismatch {
                grad_id,
                expected,
                found,
            } => write!(
                f,
                "autograd: tensor (GradId({})) was mutated in-place after being \
                 recorded on the tape (expected version {}, found {})",
                grad_id.0, expected, found,
            ),
            AutogradError::ShapeMismatch {
                grad_id,
                expected,
                found,
            } => write!(
                f,
                "autograd: gradient shape mismatch for GradId({}): \
                 expected {:?}, got {:?}",
                grad_id.0, expected, found,
            ),
            AutogradError::NoGraph => write!(
                f,
                "autograd: backward() called on a tensor with no computational graph",
            ),
            AutogradError::MissingGrad { grad_id } => write!(
                f,
                "autograd: gradient for GradId({}) not found in GradientStore",
                grad_id.0,
            ),
            AutogradError::StateError { key, message } => write!(
                f,
                "state_dict: key \"{}\": {}",
                key, message,
            ),
        }
    }
}

impl std::error::Error for AutogradError {}
