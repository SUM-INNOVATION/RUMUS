// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Embedding layer (lookup table).

use std::cell::Cell;
use std::collections::HashMap;

use crate::autograd::AutogradError;
use crate::nn::{Module, Parameter};
use crate::tensor::Tensor;

thread_local! {
    static EMB_RNG: Cell<u64> = Cell::new(987654321);
}

fn lcg_uniform(bound: f32) -> f32 {
    EMB_RNG.with(|state| {
        let s = state.get().wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        state.set(s);
        let u = (s >> 33) as f32 / (1u64 << 31) as f32;
        (2.0 * u - 1.0) * bound
    })
}

/// Embedding layer: a lookup table of fixed-size vectors.
///
/// `weight` shape: `[vocab_size, embed_dim]`.
/// `forward(indices)` where `indices` are integer token IDs (stored as f32).
pub struct Embedding {
    pub weight: Parameter,
    pub vocab_size: usize,
    pub embed_dim: usize,
}

impl Embedding {
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        let data: Vec<f32> = (0..vocab_size * embed_dim)
            .map(|_| lcg_uniform(1.0))
            .collect();
        let weight = Parameter::new(Tensor::new(data, vec![vocab_size, embed_dim]));
        Self { weight, vocab_size, embed_dim }
    }

    /// Forward: lookup embedding vectors for each index.
    ///
    /// `indices` shape: `[...]` — any shape, elements are integer token IDs.
    /// Output shape: `[..., embed_dim]`.
    pub fn forward(&self, indices: &Tensor) -> Tensor {
        indices.embedding_forward(&self.weight.tensor)
    }
}

impl Module for Embedding {
    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone()]
    }

    fn state_dict(&self, prefix: &str) -> HashMap<String, Tensor> {
        self.weight.state_dict(&format!("{}weight.", prefix))
    }

    fn load_state_dict(
        &mut self,
        dict: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), AutogradError> {
        self.weight.load_state_dict(dict, &format!("{}weight.", prefix))
    }
}
