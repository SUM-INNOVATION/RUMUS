// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Mock/generic transformer model for the inference server.
//!
//! Demonstrates how an LLM handles attention masking and KV-caching.
//! Replace with a real model loaded from safetensors in production.

use std::collections::HashMap;

use rumus::tensor::Tensor;

/// KV-cache entry for one layer: (K_cached, V_cached).
/// Shape: K = [batch, num_heads, cached_len, head_dim]
///        V = [batch, num_heads, cached_len, head_dim]
pub type KvCache = HashMap<usize, (Tensor, Tensor)>;

/// A minimal transformer model that supports:
/// - Attention masking (ignores padded tokens)
/// - KV-caching (only recomputes the new token's K/V)
pub struct MockTransformer {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub head_dim: usize,
}

impl MockTransformer {
    pub fn new(vocab_size: usize, hidden_dim: usize, num_heads: usize, num_layers: usize) -> Self {
        assert!(hidden_dim % num_heads == 0);
        Self {
            vocab_size,
            hidden_dim,
            num_heads,
            num_layers,
            head_dim: hidden_dim / num_heads,
        }
    }

    /// Full forward pass (prefill): processes all tokens, populates KV-cache.
    ///
    /// `input_ids`: [batch, seq_len] (f32-encoded token IDs)
    /// `attention_mask`: [batch, seq_len] — 1.0 for real tokens, 0.0 for padding
    /// `kv_cache`: empty on entry, populated on exit
    ///
    /// Returns logits: [batch, seq_len, vocab_size]
    pub fn prefill(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        kv_cache: &mut KvCache,
    ) -> Tensor {
        let batch = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];

        // Mock: generate random-ish logits based on input.
        // A real model would run embedding → transformer layers → lm_head.
        let guard = input_ids.data();
        let mask_guard = attention_mask.data();

        let mut logits = vec![0.0f32; batch * seq_len * self.vocab_size];

        for b in 0..batch {
            for s in 0..seq_len {
                let mask_val = mask_guard[b * seq_len + s];
                if mask_val < 0.5 {
                    // Padded position: logits are irrelevant (will be ignored).
                    continue;
                }
                let token_id = guard[b * seq_len + s] as usize;
                // Simple deterministic logits: make the next token = (current + 1) % vocab
                let next_tok = (token_id + 1) % self.vocab_size;
                logits[b * seq_len * self.vocab_size + s * self.vocab_size + next_tok] = 10.0;
            }
        }
        drop(guard);
        drop(mask_guard);

        // Populate KV-cache for each layer.
        // In a real model, K and V come from the attention projections.
        // Here we store dummy tensors with the right shapes.
        for layer in 0..self.num_layers {
            let k = Tensor::new(
                vec![0.0f32; batch * self.num_heads * seq_len * self.head_dim],
                vec![batch, self.num_heads, seq_len, self.head_dim],
            );
            let v = Tensor::new(
                vec![0.0f32; batch * self.num_heads * seq_len * self.head_dim],
                vec![batch, self.num_heads, seq_len, self.head_dim],
            );
            kv_cache.insert(layer, (k, v));
        }

        Tensor::new(logits, vec![batch, seq_len, self.vocab_size])
    }

    /// Decode step: processes only the NEW token, appends to KV-cache.
    ///
    /// `new_token_ids`: [batch, 1] — the most recently generated token
    /// `kv_cache`: updated in-place (K/V extended by 1 along the seq dim)
    ///
    /// Returns logits: [batch, 1, vocab_size]
    pub fn decode_step(
        &self,
        new_token_ids: &Tensor,
        kv_cache: &mut KvCache,
    ) -> Tensor {
        let batch = new_token_ids.shape()[0];
        let guard = new_token_ids.data();

        let mut logits = vec![0.0f32; batch * self.vocab_size];

        for b in 0..batch {
            let token_id = guard[b] as usize;
            let next_tok = (token_id + 1) % self.vocab_size;
            logits[b * self.vocab_size + next_tok] = 10.0;
        }
        drop(guard);

        // Extend the KV-cache: append one position along the seq dimension.
        for layer in 0..self.num_layers {
            if let Some((k_old, v_old)) = kv_cache.get(&layer) {
                let old_seq = k_old.shape()[2];
                let new_seq = old_seq + 1;

                // In a real model: compute new K/V from the new token's
                // projection and concatenate with the cache.
                // Here we just create a new tensor with the extended shape.
                let k_new = Tensor::new(
                    vec![0.0f32; batch * self.num_heads * new_seq * self.head_dim],
                    vec![batch, self.num_heads, new_seq, self.head_dim],
                );
                let v_new = Tensor::new(
                    vec![0.0f32; batch * self.num_heads * new_seq * self.head_dim],
                    vec![batch, self.num_heads, new_seq, self.head_dim],
                );
                kv_cache.insert(layer, (k_new, v_new));
            }
        }

        Tensor::new(logits, vec![batch, 1, self.vocab_size])
    }
}
