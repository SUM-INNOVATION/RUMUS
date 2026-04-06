// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Pre-Norm Transformer block (GPT-2/3 style).

use std::collections::HashMap;

use crate::autograd::AutogradError;
use crate::nn::{gelu, LayerNorm, Linear, Module, Parameter};
use crate::nn::attention::MultiheadAttention;
use crate::tensor::Tensor;

/// Pre-Norm Transformer block.
///
/// ```text
/// h = input + Attention(LayerNorm(input))
/// h = h     + MLP(LayerNorm(h))
/// ```
///
/// MLP: `Linear(D, 4*D) → GELU → Linear(4*D, D)`.
pub struct TransformerBlock {
    pub ln1: LayerNorm,
    pub attn: MultiheadAttention,
    pub ln2: LayerNorm,
    pub mlp_fc1: Linear,
    pub mlp_fc2: Linear,
    pub d_model: usize,
}

impl TransformerBlock {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        let mlp_dim = 4 * d_model;
        Self {
            ln1: LayerNorm::new(d_model, 1e-5),
            attn: MultiheadAttention::new(d_model, num_heads),
            ln2: LayerNorm::new(d_model, 1e-5),
            mlp_fc1: Linear::new(d_model, mlp_dim, true),
            mlp_fc2: Linear::new(mlp_dim, d_model, true),
            d_model,
        }
    }

    pub fn forward(&self, input: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let b = input.shape()[0];
        let s = input.shape()[1];
        let d = input.shape()[2];

        // Self-attention with residual
        let h_normed = self.ln1.forward(input);
        let attn_out = self.attn.forward(&h_normed, mask);
        let h = input.broadcast_add(&attn_out);

        // MLP with residual
        let h_normed = self.ln2.forward(&h);
        let flat = h_normed.reshape_tracked(vec![b * s, d]);
        let mlp_h = gelu(&self.mlp_fc1.forward(&flat));
        let mlp_out = self.mlp_fc2.forward(&mlp_h).reshape_tracked(vec![b, s, d]);
        h.broadcast_add(&mlp_out)
    }
}

impl Module for TransformerBlock {
    fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.ln1.parameters();
        p.extend(self.attn.parameters());
        p.extend(self.ln2.parameters());
        p.extend(self.mlp_fc1.parameters());
        p.extend(self.mlp_fc2.parameters());
        p
    }

    fn state_dict(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut d = self.ln1.state_dict(&format!("{}ln1.", prefix));
        d.extend(self.attn.state_dict(&format!("{}attn.", prefix)));
        d.extend(self.ln2.state_dict(&format!("{}ln2.", prefix)));
        d.extend(self.mlp_fc1.state_dict(&format!("{}mlp_fc1.", prefix)));
        d.extend(self.mlp_fc2.state_dict(&format!("{}mlp_fc2.", prefix)));
        d
    }

    fn load_state_dict(&mut self, dict: &HashMap<String, Tensor>, prefix: &str) -> Result<(), AutogradError> {
        self.ln1.load_state_dict(dict, &format!("{}ln1.", prefix))?;
        self.attn.load_state_dict(dict, &format!("{}attn.", prefix))?;
        self.ln2.load_state_dict(dict, &format!("{}ln2.", prefix))?;
        self.mlp_fc1.load_state_dict(dict, &format!("{}mlp_fc1.", prefix))?;
        self.mlp_fc2.load_state_dict(dict, &format!("{}mlp_fc2.", prefix))?;
        Ok(())
    }
}
