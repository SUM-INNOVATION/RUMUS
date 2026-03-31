//! Fully connected (dense) linear layer.

use crate::nn::{Module, Parameter};
use crate::tensor::Tensor;

use std::cell::Cell;

// ---------------------------------------------------------------------------
// Simple LCG PRNG — zero external dependencies
// ---------------------------------------------------------------------------

thread_local! {
    static RNG_STATE: Cell<u64> = Cell::new(42);
}

/// Generate a pseudo-random f32 in `[-bound, +bound]` using a thread-local
/// LCG (linear congruential generator).
///
/// Quality is sufficient for symmetry-breaking initialization — not
/// cryptographic or statistically rigorous.
fn lcg_uniform(bound: f32) -> f32 {
    RNG_STATE.with(|state| {
        // Knuth LCG parameters.
        let s = state.get().wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        state.set(s);
        // Map upper 32 bits to [0, 1), then scale to [-bound, +bound].
        let u = (s >> 33) as f32 / (1u64 << 31) as f32; // [0, 1)
        (2.0 * u - 1.0) * bound
    })
}

// ---------------------------------------------------------------------------
// Linear layer
// ---------------------------------------------------------------------------

/// Fully connected layer: `y = x @ weight + bias`.
///
/// Weight layout is `[in_features, out_features]` — this avoids a transpose
/// in the forward pass, keeping the hot path allocation-free.
///
/// Initialization uses Kaiming Uniform:
/// `bound = sqrt(1 / in_features)`, each element ~ U(-bound, +bound)`.
pub struct Linear {
    /// Weight matrix, shape `[in_features, out_features]`.
    pub weight: Parameter,
    /// Optional bias vector, shape `[out_features]`.
    pub bias: Option<Parameter>,
}

impl Linear {
    /// Create a new `Linear` layer.
    ///
    /// - `in_features`: size of each input sample.
    /// - `out_features`: size of each output sample.
    /// - `with_bias`: if `true`, a learnable bias of shape `[out_features]`
    ///   is added to the output.
    pub fn new(in_features: usize, out_features: usize, with_bias: bool) -> Self {
        let bound = (1.0 / in_features as f32).sqrt();

        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| lcg_uniform(bound))
            .collect();
        let weight = Parameter::new(Tensor::new(
            weight_data,
            vec![in_features, out_features],
        ));

        let bias = if with_bias {
            let bias_data: Vec<f32> = (0..out_features)
                .map(|_| lcg_uniform(bound))
                .collect();
            Some(Parameter::new(Tensor::new(bias_data, vec![out_features])))
        } else {
            None
        };

        Self { weight, bias }
    }

    /// Forward pass: `y = input @ weight + bias`.
    ///
    /// `input` shape: `[batch, in_features]`.
    /// Output shape: `[batch, out_features]`.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let out = input.matmul(&self.weight.tensor);
        match &self.bias {
            Some(bias) => out.add_bias(&bias.tensor),
            None => out,
        }
    }
}

impl Module for Linear {
    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
}
