//! Scaled Dot-Product Attention.
//!
//! Implements `Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V`
//! as a free function composing tracked ops.  The autograd engine handles
//! the full backward pass automatically via Kahn's traversal.

use crate::tensor::Tensor;

/// Scaled dot-product attention.
///
/// `q`, `k`, `v` shapes: `[B, S, D]` (batch × sequence × head_dim).
/// Output shape: `[B, S, D]`.
///
/// Computes:
/// ```text
/// scores      = Q @ K^T / sqrt(d_k)     [B, S, S]
/// attn_weights = softmax(scores)         [B, S, S]
/// output       = attn_weights @ V        [B, S, D]
/// ```
///
/// All operations are individually tape-recorded — no custom backward needed.
pub fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
    assert_eq!(q.ndim(), 3, "sdpa: Q must be 3-D [B, S, D]");
    assert_eq!(k.ndim(), 3, "sdpa: K must be 3-D [B, S, D]");
    assert_eq!(v.ndim(), 3, "sdpa: V must be 3-D [B, S, D]");

    let d_k = q.shape()[2] as f32;

    // Q @ K^T: [B, S, D] @ [B, D, S] → [B, S, S]
    let k_t = k.batched_transpose();
    let scores = q.bmm(&k_t);

    // Scale by 1/sqrt(d_k) using broadcast_mul with a scalar tensor.
    let scale = Tensor::new(vec![1.0 / d_k.sqrt()], vec![1]);
    let scaled = scores.broadcast_mul(&scale);

    // Softmax over the last dimension (sequence length).
    let attn_weights = scaled.softmax();

    // attn_weights @ V: [B, S, S] @ [B, S, D] → [B, S, D]
    attn_weights.bmm(v)
}
