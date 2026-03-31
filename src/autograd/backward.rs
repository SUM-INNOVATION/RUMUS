//! Kahn's-algorithm backward traversal of the computational graph.
//!
//! The [`backward`] function consumes the thread-local tape and walks it
//! in reverse, computing gradients for each operation and accumulating
//! them in a [`GradientStore`].

use std::collections::HashMap;

use crate::autograd::backward_ops::BackwardOp;
use crate::autograd::context;
use crate::autograd::{AutogradError, GradientStore};
use crate::tensor::{GradId, Tensor};

/// Execute the backward pass from `tensor`, returning accumulated gradients.
///
/// # Algorithm: Kahn's in reverse tape order
///
/// 1. **Validate:** assert the tensor is tracked and scalar (numel == 1).
/// 2. **Take the tape** from the thread-local context (consuming it —
///    prevents double-backward).
/// 3. **Seed** the [`GradientStore`] with a `1.0` tensor for the root's
///    `GradId`.
/// 4. **Build a `pending` map** by counting how many times each `GradId`
///    appears as an input across all tape entries.  This mirrors the
///    `total_grads` edge count.
/// 5. **Walk the tape in reverse.** For each entry:
///    - **Kahn's gate:** check `pending[out_grad_id] != 0`.  If not zero,
///      the node is not ready — skip it (unreachable from the root).
///    - **Dead branch handling:** if the node *is* ready (`pending == 0`)
///      but has no accumulated gradient in the `GradientStore`, it is a
///      dead branch — no gradient flows through it.  We still **must
///      decrement** the pending counts of its inputs before skipping.
///      This simulates the dead branch passing a "zero gradient" upstream,
///      freeing parent nodes whose other (live) branches have already
///      contributed.  Without this, a parent shared between a live and dead
///      branch would never reach `pending == 0` and would be incorrectly
///      skipped.
///    - Otherwise: extract the output gradient, run the `BackwardOp` math,
///      accumulate input gradients, and decrement pending counts for the
///      inputs.
///
/// # Errors
///
/// Returns [`AutogradError`] on:
/// - No computational graph (`NoGraph`).
/// - Version counter mismatch (illegal in-place mutation).
/// - Shape mismatch during gradient accumulation.
pub fn backward(tensor: &Tensor) -> Result<GradientStore, AutogradError> {
    // ---- Step 1: Validate ----
    let root_grad_id = tensor.grad_id().ok_or(AutogradError::NoGraph)?;

    assert_eq!(
        tensor.numel(),
        1,
        "backward() requires a scalar tensor (numel == 1), got numel == {}",
        tensor.numel(),
    );

    // ---- Step 2: Take the tape ----
    let tape = context::take_tape().ok_or(AutogradError::NoGraph)?;
    let entries = tape.into_entries();

    // ---- Step 3: Seed the GradientStore ----
    let mut grads = GradientStore::new();
    let seed = Tensor::new(vec![1.0f32], tensor.shape().to_vec());
    grads.accumulate(root_grad_id, seed)?;

    // ---- Step 4: Build the pending map ----
    // Count how many times each GradId appears as an *output* that other
    // entries consume.  We do this by counting input appearances: each
    // time a GradId appears in an entry's `inputs`, it means one
    // downstream consumer will push a gradient to it.
    //
    // For the output GradIds of each entry, we track how many downstream
    // entries consume that output.  We build this by scanning *all*
    // entries' output GradIds and counting how many times each output
    // appears as an input in *other* entries.
    //
    // Simpler equivalent: for each entry's output, the pending count
    // equals the number of times that output GradId appears across all
    // entries' input lists.  But since an entry's output is consumed by
    // entries that come *after* it in the tape, and we walk backwards,
    // by the time we reach the producer, all consumers have already run.
    //
    // We track pending per GradId as "how many gradient contributions
    // are still outstanding."
    let mut pending: HashMap<GradId, usize> = HashMap::new();
    for entry in &entries {
        for &input_id in &entry.inputs {
            *pending.entry(input_id).or_insert(0) += 1;
        }
    }
    // The root's output has no downstream consumers in the tape — its
    // pending count is 0 (or absent), which is correct: it's ready to
    // be processed immediately.

    // ---- Step 5: Walk tape in reverse (Kahn's) ----
    for entry in entries.into_iter().rev() {
        // Single-output ops for now (all our ops produce one tensor).
        let out_grad_id = entry.outputs[0];

        // Kahn's gate (strict zero-ready invariant): this node is
        // executable only if ALL downstream consumers have contributed
        // their gradients, i.e., pending == 0.  If pending != 0, the
        // node is unreachable from the backward root — skip without
        // decrementing parents (they are unreachable too).
        let out_pending = pending.get(&out_grad_id).copied().unwrap_or(0);
        if out_pending != 0 {
            continue;
        }

        // The node is ready (pending == 0).  Check whether any gradient
        // actually arrived.
        let out_grad = match grads.get(out_grad_id) {
            Some(g) => g.clone(),
            None => {
                // Dead branch: the node is topologically ready but no
                // gradient was seeded or propagated to it.  This happens
                // when backward() is called on a root that doesn't
                // transitively depend on this node's output.
                //
                // We MUST still decrement pending counts for the inputs.
                // Without this, a parent tensor shared between this dead
                // branch and a live branch would never reach pending == 0
                // and would be incorrectly skipped.  Conceptually, the
                // dead branch contributes a "zero gradient" upstream.
                for &input_id in &entry.inputs {
                    if let Some(count) = pending.get_mut(&input_id) {
                        *count -= 1;
                    }
                }
                continue;
            }
        };

        // Dispatch backward math and accumulate input gradients.
        match &entry.op {
            BackwardOp::Add(add_bw) => {
                add_bw.lhs_version.check()?;
                add_bw.rhs_version.check()?;

                // ∂L/∂a = ∂L/∂c  (identity — clone the gradient)
                // ∂L/∂b = ∂L/∂c  (identity)
                let grad_lhs = out_grad.clone();
                let grad_rhs = out_grad;

                grads.accumulate(entry.inputs[0], grad_lhs)?;
                grads.accumulate(entry.inputs[1], grad_rhs)?;
            }

            BackwardOp::Mul(mul_bw) => {
                mul_bw.lhs_version.check()?;
                mul_bw.rhs_version.check()?;

                // Reconstruct saved tensors (no autograd state).
                let saved_lhs = Tensor::from_storage_and_layout(
                    mul_bw.lhs_storage.clone(),
                    mul_bw.lhs_layout.clone(),
                );
                let saved_rhs = Tensor::from_storage_and_layout(
                    mul_bw.rhs_storage.clone(),
                    mul_bw.rhs_layout.clone(),
                );

                // ∂L/∂a = ∂L/∂c ⊙ b
                let grad_lhs = out_grad.mul(&saved_rhs);
                // ∂L/∂b = ∂L/∂c ⊙ a
                let grad_rhs = out_grad.mul(&saved_lhs);

                grads.accumulate(entry.inputs[0], grad_lhs)?;
                grads.accumulate(entry.inputs[1], grad_rhs)?;
            }

            BackwardOp::Matmul(mm_bw) => {
                mm_bw.lhs_version.check()?;
                mm_bw.rhs_version.check()?;

                let saved_a = Tensor::from_storage_and_layout(
                    mm_bw.lhs_storage.clone(),
                    mm_bw.lhs_layout.clone(),
                );
                let saved_b = Tensor::from_storage_and_layout(
                    mm_bw.rhs_storage.clone(),
                    mm_bw.rhs_layout.clone(),
                );

                // ∂L/∂A = ∂L/∂C @ Bᵀ
                //   (m×n) @ (n×k) → (m×k)
                let b_t = saved_b.transpose(0, 1);
                let grad_lhs = out_grad.matmul(&b_t);

                // ∂L/∂B = Aᵀ @ ∂L/∂C
                //   (k×m) @ (m×n) → (k×n)
                let a_t = saved_a.transpose(0, 1);
                let grad_rhs = a_t.matmul(&out_grad);

                grads.accumulate(entry.inputs[0], grad_lhs)?;
                grads.accumulate(entry.inputs[1], grad_rhs)?;
            }
        }

        // Decrement pending counts for inputs (live branch).
        for &input_id in &entry.inputs {
            if let Some(count) = pending.get_mut(&input_id) {
                *count -= 1;
            }
        }
    }

    Ok(grads)
}
