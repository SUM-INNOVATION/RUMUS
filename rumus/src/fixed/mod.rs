// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Deterministic fixed-point (`i16`) CPU inference.
//!
//! This module is a first-class, **CPU-only, integer-only** inference path.
//! It exists so RUMUS can produce bit-exact, reproducible fixed-point results
//! suitable as proof fixtures for zero-knowledge backends (e.g. halo2).
//!
//! # Representation
//!
//! Every tensor in a fixed-point path shares one **global** scale.  A raw
//! `i16` value `q` represents the real number `q / 2^scale_log2`.  There are
//! no per-block or per-channel scales — that is what distinguishes this path
//! from the [`DType::Q8`](crate::tensor::DType::Q8) block-quantized path.
//!
//! # Arithmetic
//!
//! For a dense layer `y = x @ W + b` with all operands `i16` at scale
//! `S = 2^scale_log2`:
//!
//! 1. **Multiply** — `x[k] * W[k]` is computed as `i16 × i16 → i32` (exact;
//!    `32768 * 32768 < i32::MAX`).
//! 2. **Accumulate** — products are summed into an `i64` accumulator.  This
//!    accumulator lives in the *widened* domain at scale `S²`.
//! 3. **Bias** — the bias is an `i16` at scale `S`; it is promoted into the
//!    widened `S²` domain (`<< scale_log2`) and added **before** any
//!    saturation.
//! 4. **Requantize** — the `S²`-domain accumulator is shifted back to scale
//!    `S` via [`requantize`], using round-half-away-from-zero.
//! 5. **Saturate** — the result is clamped to the `i16` range.
//!
//! No `f32` value is ever produced or consumed on this path.

use crate::tensor::{DType, Tensor};

mod fixture;
pub use fixture::{FixedFixture, FixedFixtureBuilder, FixedLayerSpec};

// ---------------------------------------------------------------------------
// Rounding / overflow semantics — named, explicit API concepts
// ---------------------------------------------------------------------------

/// Rounding rule used when a widened accumulator is requantized down to `i16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedRounding {
    /// Round to nearest; exact halves (ties) round **away from zero**.
    ///
    /// `1.5 → 2`, `2.5 → 3`, `-1.5 → -2`, `-2.5 → -3`.
    NearestTiesAwayFromZero,
}

impl FixedRounding {
    /// Stable lowercase identifier used in fixture serialization.
    pub fn as_str(self) -> &'static str {
        match self {
            FixedRounding::NearestTiesAwayFromZero => "nearest_ties_away_from_zero",
        }
    }
}

/// Overflow rule applied to a requantized value that falls outside `i16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedOverflow {
    /// Clamp to `[i16::MIN, i16::MAX]` (no wraparound).
    SaturateI16,
}

impl FixedOverflow {
    /// Stable lowercase identifier used in fixture serialization.
    pub fn as_str(self) -> &'static str {
        match self {
            FixedOverflow::SaturateI16 => "saturate_i16",
        }
    }
}

/// The canonical rounding rule for every op in this module.
pub const FIXED_ROUNDING: FixedRounding = FixedRounding::NearestTiesAwayFromZero;

/// The canonical overflow rule for every op in this module.
pub const FIXED_OVERFLOW: FixedOverflow = FixedOverflow::SaturateI16;

// ---------------------------------------------------------------------------
// Requantization
// ---------------------------------------------------------------------------

/// Requantize a widened `i64` accumulator back to an `i16` fixed-point value.
///
/// The accumulator `acc` is assumed to be at scale `2^(2 * scale_log2)` (the
/// natural scale of a sum of `i16 × i16` products).  The result is at scale
/// `2^scale_log2`.
///
/// Semantics — these are fixed and exhaustively tested:
/// - **Rounding**: [`FixedRounding::NearestTiesAwayFromZero`].
/// - **Overflow**: [`FixedOverflow::SaturateI16`].
///
/// `scale_log2 == 0` is a no-op shift (the accumulator is already at the
/// target scale) and only the saturation step applies.
pub fn requantize(acc: i64, scale_log2: u8) -> i16 {
    debug_assert!(acc != i64::MIN, "requantize: accumulator magnitude out of range");

    let shift = scale_log2 as u32;
    let shifted: i64 = if shift == 0 {
        acc
    } else {
        // Round half away from zero: bias by half-an-LSB in the direction of
        // the sign, then truncate toward zero via an arithmetic shift of the
        // magnitude.
        let half: i64 = 1i64 << (shift - 1);
        if acc >= 0 {
            (acc + half) >> shift
        } else {
            -(((-acc) + half) >> shift)
        }
    };

    // Saturating cast to i16.
    shifted.clamp(i16::MIN as i64, i16::MAX as i64) as i16
}

// ---------------------------------------------------------------------------
// FixedLinear — integer-domain dense layer
// ---------------------------------------------------------------------------

/// A dense (fully connected) layer that runs entirely in the integer domain.
///
/// Weight layout is `[in_features, out_features]` — matching RUMUS's native
/// [`Linear`](crate::nn::Linear) layout, so no transpose is needed.
///
/// This is **not** a [`Module`](crate::nn::Module): it has no learnable
/// parameters and never participates in autograd.  State is held directly in
/// the public [`weight`](Self::weight) / [`bias`](Self::bias) tensors.
pub struct FixedLinear {
    /// Weight tensor — [`DType::FixedI16`], shape `[in_features, out_features]`.
    pub weight: Tensor,
    /// Optional bias tensor — [`DType::FixedI16`], shape `[out_features]`.
    pub bias: Option<Tensor>,
    /// Input feature count.
    pub in_features: usize,
    /// Output feature count.
    pub out_features: usize,
    /// Global fixed-point scale exponent shared by weight, bias, and I/O.
    pub scale_log2: u8,
}

impl FixedLinear {
    /// Construct a `FixedLinear` from raw `i16` weight (and optional bias) data.
    ///
    /// `weight` is row-major `[in_features, out_features]`; `bias`, if present,
    /// is `[out_features]`.
    ///
    /// # Panics
    ///
    /// Panics on a length mismatch against `in_features` / `out_features`.
    pub fn new(
        weight: Vec<i16>,
        bias: Option<Vec<i16>>,
        in_features: usize,
        out_features: usize,
        scale_log2: u8,
    ) -> Self {
        assert_eq!(
            weight.len(),
            in_features * out_features,
            "FixedLinear::new: weight has {} elements, expected {}x{}={}",
            weight.len(),
            in_features,
            out_features,
            in_features * out_features,
        );
        if let Some(ref b) = bias {
            assert_eq!(
                b.len(),
                out_features,
                "FixedLinear::new: bias has {} elements, expected {}",
                b.len(),
                out_features,
            );
        }

        let weight = Tensor::from_i16_fixed(weight, vec![in_features, out_features], scale_log2);
        let bias = bias.map(|b| Tensor::from_i16_fixed(b, vec![out_features], scale_log2));

        Self { weight, bias, in_features, out_features, scale_log2 }
    }

    /// Construct a `FixedLinear` from already-built fixed-point tensors.
    ///
    /// `weight` and `bias` **must be contiguous** — the integer kernel reads
    /// the raw `fixed_i16_data()` slice as row-major storage, so a transposed
    /// or otherwise non-contiguous view would compute the wrong result.
    ///
    /// # Panics
    ///
    /// Panics unless `weight` is a 2-D contiguous `FixedI16` tensor and `bias`
    /// (if any) is a 1-D contiguous `FixedI16` tensor at the **same**
    /// `scale_log2`.
    pub fn from_parts(weight: Tensor, bias: Option<Tensor>) -> Self {
        let scale_log2 = match weight.dtype() {
            DType::FixedI16 { scale_log2 } => scale_log2,
            other => panic!("FixedLinear::from_parts: weight must be FixedI16, got {:?}", other),
        };
        assert_eq!(weight.ndim(), 2, "FixedLinear::from_parts: weight must be 2-D");
        assert!(
            weight.is_contiguous(),
            "FixedLinear::from_parts: weight must be contiguous \
             (the integer kernel reads raw row-major storage)",
        );
        let in_features = weight.shape()[0];
        let out_features = weight.shape()[1];

        if let Some(ref b) = bias {
            match b.dtype() {
                DType::FixedI16 { scale_log2: bs } => assert_eq!(
                    bs, scale_log2,
                    "FixedLinear::from_parts: bias scale_log2 {} != weight scale_log2 {}",
                    bs, scale_log2,
                ),
                other => panic!("FixedLinear::from_parts: bias must be FixedI16, got {:?}", other),
            }
            assert_eq!(b.ndim(), 1, "FixedLinear::from_parts: bias must be 1-D");
            assert!(
                b.is_contiguous(),
                "FixedLinear::from_parts: bias must be contiguous",
            );
            assert_eq!(
                b.shape()[0],
                out_features,
                "FixedLinear::from_parts: bias length mismatch",
            );
        }

        Self { weight, bias, in_features, out_features, scale_log2 }
    }

    /// Run the integer-domain forward pass `y = x @ W + b`.
    ///
    /// `input` must be a contiguous 2-D `FixedI16` tensor of shape
    /// `[batch, in_features]` at this layer's `scale_log2`.  The output is a
    /// `FixedI16` tensor of shape `[batch, out_features]` at the same scale.
    ///
    /// # Panics
    ///
    /// Panics on any dtype, scale, rank, or shape mismatch.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let s = self.scale_log2;

        match input.dtype() {
            DType::FixedI16 { scale_log2 } => assert_eq!(
                scale_log2, s,
                "FixedLinear::forward: input scale_log2 {} != layer scale_log2 {}",
                scale_log2, s,
            ),
            other => panic!("FixedLinear::forward: input must be FixedI16, got {:?}", other),
        }
        assert!(
            input.is_contiguous(),
            "FixedLinear::forward: input must be contiguous",
        );
        assert_eq!(
            input.ndim(),
            2,
            "FixedLinear::forward: input must be 2-D [batch, in_features]",
        );
        let batch = input.shape()[0];
        assert_eq!(
            input.shape()[1],
            self.in_features,
            "FixedLinear::forward: input feature count {} != in_features {}",
            input.shape()[1],
            self.in_features,
        );

        // Weight and bias contiguity is normally enforced at construction,
        // but re-check here so the kernel never reads strided storage as
        // row-major (which would silently compute the wrong result).
        assert!(
            self.weight.is_contiguous(),
            "FixedLinear::forward: weight must be contiguous",
        );
        if let Some(ref b) = self.bias {
            assert!(b.is_contiguous(), "FixedLinear::forward: bias must be contiguous");
        }

        let x = input.fixed_i16_data();
        let w = self.weight.fixed_i16_data();
        let bias_guard = self.bias.as_ref().map(|b| b.fixed_i16_data());

        let mut out = vec![0i16; batch * self.out_features];
        for m in 0..batch {
            for n in 0..self.out_features {
                // Accumulate i16 x i16 products in the widened i64 domain.
                let mut acc: i64 = 0;
                for k in 0..self.in_features {
                    let xv = x[m * self.in_features + k] as i32;
                    let wv = w[k * self.out_features + n] as i32;
                    acc += (xv * wv) as i64;
                }
                // Bias add happens in the widened (scale^2) domain, before
                // saturation: promote the i16 bias by `s` bits.
                if let Some(ref b) = bias_guard {
                    acc += (b[n] as i64) << s;
                }
                out[m * self.out_features + n] = requantize(acc, s);
            }
        }

        Tensor::from_i16_fixed(out, vec![batch, self.out_features], s)
    }
}

// ---------------------------------------------------------------------------
// ReLU
// ---------------------------------------------------------------------------

/// Element-wise integer ReLU for fixed-point tensors: `max(0, x)`.
///
/// In fixed-point, the real value `0` is the integer `0` at every scale, so
/// this is an exact integer clamp with no rounding or saturation.
///
/// # Panics
///
/// Panics unless `input` is a contiguous [`DType::FixedI16`] tensor.
pub fn relu(input: &Tensor) -> Tensor {
    let scale_log2 = match input.dtype() {
        DType::FixedI16 { scale_log2 } => scale_log2,
        other => panic!("fixed::relu: expected FixedI16, got {:?}", other),
    };
    assert!(input.is_contiguous(), "fixed::relu: input must be contiguous");

    let data = input.fixed_i16_data();
    let out: Vec<i16> = data.iter().map(|&v| v.max(0)).collect();
    drop(data);

    Tensor::from_i16_fixed(out, input.shape().to_vec(), scale_log2)
}

impl Tensor {
    /// Integer-domain ReLU for fixed-point tensors — see [`crate::fixed::relu`].
    pub fn relu_fixed(&self) -> Tensor {
        relu(self)
    }
}

// ---------------------------------------------------------------------------
// Unit tests for the requantization primitive
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requantize_no_shift_is_saturating_identity() {
        assert_eq!(requantize(0, 0), 0);
        assert_eq!(requantize(123, 0), 123);
        assert_eq!(requantize(-123, 0), -123);
        assert_eq!(requantize(40_000, 0), i16::MAX);
        assert_eq!(requantize(-40_000, 0), i16::MIN);
    }

    #[test]
    fn requantize_rounds_half_away_from_zero() {
        // scale_log2 = 1  →  divide by 2.
        assert_eq!(requantize(2, 1), 1); //  1.0  -> 1
        assert_eq!(requantize(3, 1), 2); //  1.5  -> 2 (tie, away)
        assert_eq!(requantize(5, 1), 3); //  2.5  -> 3 (tie, away)
        assert_eq!(requantize(1, 1), 1); //  0.5  -> 1 (tie, away)
        assert_eq!(requantize(-1, 1), -1); // -0.5 -> -1 (tie, away)
        assert_eq!(requantize(-3, 1), -2); // -1.5 -> -2 (tie, away)
        assert_eq!(requantize(-5, 1), -3); // -2.5 -> -3 (tie, away)

        // scale_log2 = 2  →  divide by 4.
        assert_eq!(requantize(6, 2), 2); //  1.5  -> 2 (tie, away)
        assert_eq!(requantize(2, 2), 1); //  0.5  -> 1 (tie, away)
        assert_eq!(requantize(1, 2), 0); //  0.25 -> 0
        assert_eq!(requantize(-1, 2), 0); // -0.25 -> 0
        assert_eq!(requantize(-6, 2), -2); // -1.5 -> -2 (tie, away)
    }

    #[test]
    fn requantize_saturates_to_i16_range() {
        // 100_000 / 2 = 50_000 > i16::MAX.
        assert_eq!(requantize(100_000, 1), i16::MAX);
        assert_eq!(requantize(-100_000, 1), i16::MIN);
    }
}
