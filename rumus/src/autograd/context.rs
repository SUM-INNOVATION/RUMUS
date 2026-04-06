// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Thread-local autograd state: tape, GradId generator, and `no_grad` guard.
//!
//! Each thread owns its own [`Tape`] and id counter — no global locks.
//! This mirrors PyTorch's thread-local dispatch model.

use std::cell::RefCell;

use crate::autograd::Tape;
use crate::tensor::GradId;

// ---------------------------------------------------------------------------
// Thread-local context
// ---------------------------------------------------------------------------

/// Per-thread autograd bookkeeping.
struct AutogradContext {
    /// The active tape.  `None` before the first tracked op or after
    /// `take_tape()` consumes it for backward.
    tape: Option<Tape>,
    /// Monotonically increasing GradId generator.
    next_grad_id: usize,
    /// Nesting depth of `no_grad` guards.  Recording is suppressed while
    /// this is > 0.
    no_grad_depth: usize,
}

thread_local! {
    static CONTEXT: RefCell<AutogradContext> = RefCell::new(AutogradContext {
        tape: None,
        next_grad_id: 0,
        no_grad_depth: 0,
    });
}

// ---------------------------------------------------------------------------
// Public API (crate-internal)
// ---------------------------------------------------------------------------

/// Generate a fresh [`GradId`], unique within this thread.
///
/// The counter is monotonically increasing and never resets.
pub(crate) fn next_grad_id() -> GradId {
    CONTEXT.with(|c| {
        let mut ctx = c.borrow_mut();
        let id = GradId(ctx.next_grad_id);
        ctx.next_grad_id += 1;
        id
    })
}

/// Execute `f` with a mutable reference to the active [`Tape`].
///
/// If no tape exists yet, one is lazily created.  Returns `None` if
/// `no_grad` is active (recording is suppressed).
pub(crate) fn with_tape<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&mut Tape) -> R,
{
    CONTEXT.with(|c| {
        let mut ctx = c.borrow_mut();
        if ctx.no_grad_depth > 0 {
            return None;
        }
        // Lazily create the tape on the first tracked operation.
        if ctx.tape.is_none() {
            ctx.tape = Some(Tape::new());
        }
        Some(f(ctx.tape.as_mut().unwrap()))
    })
}

/// Take the tape out of the thread-local context, consuming it.
///
/// Returns `None` if no tape exists (no tracked ops were recorded).
/// After this call, the next tracked op will create a fresh tape.
pub(crate) fn take_tape() -> Option<Tape> {
    CONTEXT.with(|c| c.borrow_mut().tape.take())
}

/// Returns `true` if tape recording is currently suppressed by a
/// [`NoGradGuard`].
pub(crate) fn is_no_grad() -> bool {
    CONTEXT.with(|c| c.borrow().no_grad_depth > 0)
}

// ---------------------------------------------------------------------------
// NoGradGuard — RAII suppression of tape recording
// ---------------------------------------------------------------------------

/// RAII guard that suppresses tape recording for the duration of its
/// lifetime.
///
/// Supports nesting: creating a second `NoGradGuard` while one is already
/// active increments the depth counter.  Recording resumes only when all
/// guards have been dropped (depth returns to 0).
///
/// # Example
///
/// ```ignore
/// let _guard = no_grad();
/// let y = x.add(&x);  // not recorded on the tape
/// // _guard drops here → recording resumes
/// ```
pub struct NoGradGuard;

/// Create a [`NoGradGuard`] that suppresses autograd tape recording.
///
/// The guard is active until it is dropped.  Nesting is supported.
pub fn no_grad() -> NoGradGuard {
    CONTEXT.with(|c| c.borrow_mut().no_grad_depth += 1);
    NoGradGuard
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        CONTEXT.with(|c| {
            let mut ctx = c.borrow_mut();
            ctx.no_grad_depth = ctx.no_grad_depth.saturating_sub(1);
        });
    }
}
