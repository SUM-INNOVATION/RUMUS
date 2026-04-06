// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Scaled Dot-Product Attention and Multi-Head Attention.

use std::collections::HashMap;

use crate::autograd::AutogradError;
use crate::nn::{Linear, Module, Parameter};
use crate::tensor::Tensor;

/// Scaled dot-product attention with optional causal mask.
///
/// `q`, `k`, `v` shapes: `[B, S, D]`.
/// `mask`: optional `[S, S]` or broadcastable — added to scores before softmax.
///         For causal masking, use upper-triangular `-1e9`.
///
/// Output: `[B, S, D]`.
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> Tensor {
    assert_eq!(q.ndim(), 3, "sdpa: Q must be 3-D [B, S, D]");
    assert_eq!(k.ndim(), 3, "sdpa: K must be 3-D [B, S, D]");
    assert_eq!(v.ndim(), 3, "sdpa: V must be 3-D [B, S, D]");

    let d_k = q.shape()[2] as f32;

    let k_t = k.batched_transpose();
    let scores = q.bmm(&k_t); // [B, S, S]

    // Scale by 1/sqrt(d_k). Create a [1,1,1]-shaped scalar for 3D broadcast.
    let inv_sqrt = 1.0 / d_k.sqrt();
    let scale = Tensor::new(vec![inv_sqrt], vec![1, 1, 1]);
    let scaled = scores.broadcast_mul(&scale);

    // Apply mask before softmax (e.g., causal: upper triangle = -1e9).
    let masked = match mask {
        Some(m) => scaled.broadcast_add(m),
        None => scaled,
    };

    let attn_weights = masked.softmax();
    attn_weights.bmm(v)
}

// ---------------------------------------------------------------------------
// MultiheadAttention
// ---------------------------------------------------------------------------

/// Multi-head attention with separate Q/K/V/O linear projections.
///
/// Splits `[B, S, D]` into `num_heads` heads of `head_dim = D / num_heads`,
/// applies SDPA per head, concatenates, and projects back to `D`.
pub struct MultiheadAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
    pub d_model: usize,
}

impl MultiheadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert_eq!(d_model % num_heads, 0, "d_model must be divisible by num_heads");
        let head_dim = d_model / num_heads;
        Self {
            q_proj: Linear::new(d_model, d_model, true),
            k_proj: Linear::new(d_model, d_model, true),
            v_proj: Linear::new(d_model, d_model, true),
            out_proj: Linear::new(d_model, d_model, true),
            num_heads,
            head_dim,
            d_model,
        }
    }

    pub fn forward(&self, input: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let b = input.shape()[0];
        let s = input.shape()[1];
        let d = input.shape()[2];
        let nh = self.num_heads;
        let hd = self.head_dim;

        // 1. Project Q, K, V: flatten [B,S,D] → [B*S,D] for Linear
        let flat = input.reshape_tracked(vec![b * s, d]);
        let q = self.q_proj.forward(&flat).reshape_tracked(vec![b, s, d]);
        let k = self.k_proj.forward(&flat).reshape_tracked(vec![b, s, d]);
        let v = self.v_proj.forward(&flat).reshape_tracked(vec![b, s, d]);

        // 2. Split heads + SDPA + merge
        let (attn_out_flat, _) = if nh == 1 {
            // Single head: Q/K/V are already [B, S, D] = [B, S, hd].
            // No reshape/transpose needed — pass directly to SDPA.
            let attn_out = scaled_dot_product_attention(&q, &k, &v, mask);
            (attn_out.reshape_tracked(vec![b * s, d]), 0)
        } else {
            // Multi-head: [B,S,D] → [B,S,nh,hd] → [B,nh,S,hd] → [B*nh,S,hd]
            let q = q.reshape_tracked(vec![b, s, nh, hd])
                     .transpose_tracked(1, 2).contiguous_tracked()
                     .reshape_tracked(vec![b * nh, s, hd]);
            let k = k.reshape_tracked(vec![b, s, nh, hd])
                     .transpose_tracked(1, 2).contiguous_tracked()
                     .reshape_tracked(vec![b * nh, s, hd]);
            let v = v.reshape_tracked(vec![b, s, nh, hd])
                     .transpose_tracked(1, 2).contiguous_tracked()
                     .reshape_tracked(vec![b * nh, s, hd]);

            let attn_out = scaled_dot_product_attention(&q, &k, &v, mask);

            let merged = attn_out.reshape_tracked(vec![b, nh, s, hd])
                                 .transpose_tracked(1, 2).contiguous_tracked()
                                 .reshape_tracked(vec![b * s, d]);
            (merged, 0)
        };
        let merged = attn_out_flat;

        // 5. Output projection → [B,S,D]
        self.out_proj.forward(&merged).reshape_tracked(vec![b, s, d])
    }
}

impl Module for MultiheadAttention {
    fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.q_proj.parameters();
        p.extend(self.k_proj.parameters());
        p.extend(self.v_proj.parameters());
        p.extend(self.out_proj.parameters());
        p
    }

    fn state_dict(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut d = self.q_proj.state_dict(&format!("{}q_proj.", prefix));
        d.extend(self.k_proj.state_dict(&format!("{}k_proj.", prefix)));
        d.extend(self.v_proj.state_dict(&format!("{}v_proj.", prefix)));
        d.extend(self.out_proj.state_dict(&format!("{}out_proj.", prefix)));
        d
    }

    fn load_state_dict(&mut self, dict: &HashMap<String, Tensor>, prefix: &str) -> Result<(), AutogradError> {
        self.q_proj.load_state_dict(dict, &format!("{}q_proj.", prefix))?;
        self.k_proj.load_state_dict(dict, &format!("{}k_proj.", prefix))?;
        self.v_proj.load_state_dict(dict, &format!("{}v_proj.", prefix))?;
        self.out_proj.load_state_dict(dict, &format!("{}out_proj.", prefix))?;
        Ok(())
    }
}
