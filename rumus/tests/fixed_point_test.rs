// SPDX-License-Identifier: Apache-2.0 OR MIT
//! End-to-end tests for the deterministic fixed-point (`i16`) inference path.
//!
//! Every assertion here is on **exact integer** behavior — there is no
//! tolerance, because the whole point of this path is bit-reproducibility.

use rumus::fixed::{self, FixedFixtureBuilder, FixedLinear};
use rumus::tensor::{DType, Tensor};

// ---------------------------------------------------------------------------
// 1. Construction & raw integer access
// ---------------------------------------------------------------------------

#[test]
fn fixed_tensor_construction_and_raw_access() {
    let t = Tensor::from_i16_fixed(vec![-3, 0, 7, 32_000], vec![2, 2], 8);

    assert_eq!(t.dtype(), DType::FixedI16 { scale_log2: 8 });
    assert!(t.is_fixed_point());
    assert_eq!(t.fixed_scale_log2(), Some(8));
    assert_eq!(t.shape(), &[2, 2]);

    // Raw i16 data is returned verbatim — no conversion.
    let data = t.fixed_i16_data();
    assert_eq!(&*data, &[-3, 0, 7, 32_000]);
}

#[test]
#[should_panic(expected = "fixed-point")]
fn data_accessor_rejects_fixed_tensor() {
    // Tensor::data() is the f32 accessor; it must NOT silently convert.
    let t = Tensor::from_i16_fixed(vec![1, 2, 3], vec![3], 8);
    let _ = t.data();
}

#[test]
#[should_panic(expected = "FixedI16")]
fn fixed_accessor_rejects_float_tensor() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let _ = t.fixed_i16_data();
}

#[test]
#[should_panic(expected = "scale_log2")]
fn scale_log2_out_of_range_panics() {
    let _ = Tensor::from_i16_fixed(vec![1, 2], vec![2], 16);
}

// ---------------------------------------------------------------------------
// 2. FixedLinear — exact integer output
// ---------------------------------------------------------------------------

#[test]
fn fixed_linear_exact_output() {
    // scale_log2 = 4  (scale = 16).
    // weight [2,1] = [16, 48]  -> real [1.0, 3.0]
    // input  [1,2] = [16, 32]  -> real [1.0, 2.0]
    // acc = 16*16 + 32*48 = 1792  (scale 16^2 = 256)
    // requantize(1792, 4) = round(1792 / 16) = round(112.0) = 112  -> real 7.0
    let layer = FixedLinear::new(vec![16, 48], None, 2, 1, 4);
    let input = Tensor::from_i16_fixed(vec![16, 32], vec![1, 2], 4);

    let out = layer.forward(&input);

    assert_eq!(out.dtype(), DType::FixedI16 { scale_log2: 4 });
    assert_eq!(out.shape(), &[1, 1]);
    assert_eq!(&*out.fixed_i16_data(), &[112]);
}

// ---------------------------------------------------------------------------
// 3. Bias is added in the widened domain, BEFORE saturation
// ---------------------------------------------------------------------------

#[test]
fn widened_domain_bias_before_saturation() {
    // scale_log2 = 8  (scale = 256).
    // weight [2,1] = [256, 256], input [1,2] = [20000, 20000]
    //   pre-bias acc = 20000*256 + 20000*256 = 10_240_000
    //   pre-bias requantize = 10_240_000 / 256 = 40000  -> would SATURATE to 32767
    // bias [-10000] promoted to the widened domain: -10000 << 8 = -2_560_000
    //   acc + bias = 7_680_000  ->  requantize = 30000  (in range)
    //
    // A "saturate-then-add" bug would give 32767 - 10000 = 22767.
    // A "bias added at scale S, not S^2" bug would give ~32767.
    let layer = FixedLinear::new(vec![256, 256], Some(vec![-10000]), 2, 1, 8);
    let input = Tensor::from_i16_fixed(vec![20_000, 20_000], vec![1, 2], 8);

    let out = layer.forward(&input);
    assert_eq!(&*out.fixed_i16_data(), &[30_000]);
}

// ---------------------------------------------------------------------------
// 4. Integer ReLU over signed i16
// ---------------------------------------------------------------------------

#[test]
fn relu_over_signed_i16() {
    let t = Tensor::from_i16_fixed(vec![-32_768, -100, -1, 0, 1, 100, 32_767], vec![7], 8);

    let via_fn = fixed::relu(&t);
    let via_method = t.relu_fixed();

    let expected: &[i16] = &[0, 0, 0, 0, 1, 100, 32_767];
    assert_eq!(&*via_fn.fixed_i16_data(), expected);
    assert_eq!(&*via_method.fixed_i16_data(), expected);
    // ReLU preserves dtype/scale.
    assert_eq!(via_fn.dtype(), DType::FixedI16 { scale_log2: 8 });
}

// ---------------------------------------------------------------------------
// 5. Requantization & rounding rule (nearest, ties away from zero)
// ---------------------------------------------------------------------------

#[test]
fn requantize_rounding_rule() {
    // scale_log2 = 1  ->  divide by 2; halves are exact ties.
    assert_eq!(fixed::requantize(2, 1), 1); //  1.0  -> 1
    assert_eq!(fixed::requantize(3, 1), 2); //  1.5  -> 2  (tie -> away)
    assert_eq!(fixed::requantize(5, 1), 3); //  2.5  -> 3  (tie -> away)
    assert_eq!(fixed::requantize(1, 1), 1); //  0.5  -> 1  (tie -> away)
    assert_eq!(fixed::requantize(-1, 1), -1); // -0.5 -> -1 (tie -> away)
    assert_eq!(fixed::requantize(-3, 1), -2); // -1.5 -> -2 (tie -> away)
    assert_eq!(fixed::requantize(-5, 1), -3); // -2.5 -> -3 (tie -> away)

    // Non-tie rounding.
    assert_eq!(fixed::requantize(7, 2), 2); //  1.75 -> 2
    assert_eq!(fixed::requantize(-7, 2), -2); // -1.75 -> -2

    // scale_log2 = 0 is a pure (saturating) identity.
    assert_eq!(fixed::requantize(12_345, 0), 12_345);
}

// ---------------------------------------------------------------------------
// 6. Saturation / overflow rule (clamp to i16 range)
// ---------------------------------------------------------------------------

#[test]
fn requantize_saturates() {
    assert_eq!(fixed::requantize(100_000, 1), i16::MAX);
    assert_eq!(fixed::requantize(-100_000, 1), i16::MIN);
}

#[test]
fn fixed_linear_saturates_on_overflow() {
    // scale_log2 = 0: acc IS the output (before clamping).
    // 32767 * 32767 = 1_073_676_289  -> clamps to i16::MAX.
    let pos = FixedLinear::new(vec![32_767], None, 1, 1, 0);
    let pos_in = Tensor::from_i16_fixed(vec![32_767], vec![1, 1], 0);
    assert_eq!(&*pos.forward(&pos_in).fixed_i16_data(), &[i16::MAX]);

    // 32767 * -32768 = -1_073_709_056  -> clamps to i16::MIN.
    let neg = FixedLinear::new(vec![-32_768], None, 1, 1, 0);
    let neg_in = Tensor::from_i16_fixed(vec![32_767], vec![1, 1], 0);
    assert_eq!(&*neg.forward(&neg_in).fixed_i16_data(), &[i16::MIN]);
}

// ---------------------------------------------------------------------------
// 7. Full fixed-point MLP: Linear -> ReLU -> Linear
// ---------------------------------------------------------------------------

/// Builds the canonical tiny MLP used by both the MLP test and the fixture
/// test, so they are guaranteed to describe the same model.
fn canonical_mlp() -> (FixedLinear, FixedLinear, Tensor) {
    // scale_log2 = 8 (scale = 256).
    let layer1 = FixedLinear::new(vec![256, -256, 256, 256], None, 2, 2, 8);
    let layer2 = FixedLinear::new(vec![256, 256], Some(vec![256]), 2, 1, 8);
    let input = Tensor::from_i16_fixed(vec![256, 256], vec![1, 2], 8);
    (layer1, layer2, input)
}

#[test]
fn fixed_mlp_linear_relu_linear() {
    let (layer1, layer2, input) = canonical_mlp();

    // Layer 1: real [1,1] @ [[1,-1],[1,1]] = [2, 0]  -> raw [512, 0]
    let h = layer1.forward(&input);
    assert_eq!(&*h.fixed_i16_data(), &[512, 0]);

    // ReLU: [512, 0] unchanged (both >= 0).
    let a = fixed::relu(&h);
    assert_eq!(&*a.fixed_i16_data(), &[512, 0]);

    // Layer 2: real [2,0] @ [1,1] + 1 = 3  -> raw [768]
    let out = layer2.forward(&a);
    assert_eq!(&*out.fixed_i16_data(), &[768]);
    assert_eq!(out.dtype(), DType::FixedI16 { scale_log2: 8 });
}

// ---------------------------------------------------------------------------
// 8. Canonical fixture export — deterministic & stable bytes
// ---------------------------------------------------------------------------

#[test]
fn fixture_export_is_stable_and_canonical() {
    let (layer1, layer2, input) = canonical_mlp();
    let h = layer1.forward(&input);
    let a = fixed::relu(&h);
    let output = layer2.forward(&a);

    let mut builder = FixedFixtureBuilder::new(8);
    builder.add_linear(&layer1);
    builder.add_relu();
    builder.add_linear(&layer2);
    let fixture = builder.finish(&input, &output);

    // Stability: repeated serialization yields byte-identical output.
    let bytes_a = fixture.to_canonical_bytes();
    let bytes_b = fixture.to_canonical_bytes();
    assert_eq!(bytes_a, bytes_b, "fixture serialization must be deterministic");

    // Canonical bytes: must match the committed golden fixture exactly.
    let golden = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/canonical_fixed_mlp.json"
    ))
    .expect("golden fixture file must exist");
    assert_eq!(
        String::from_utf8(bytes_a).unwrap(),
        String::from_utf8(golden).unwrap(),
        "canonical fixture bytes drifted from the golden file",
    );
}

// ---------------------------------------------------------------------------
// 9. Existing float behavior is unchanged
// ---------------------------------------------------------------------------

#[test]
fn existing_float_path_unchanged() {
    // f32 construction + access still works exactly as before.
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert_eq!(t.dtype(), DType::F32);
    assert!(!t.is_fixed_point());
    assert_eq!(&*t.data(), &[1.0, 2.0, 3.0, 4.0]);

    // f32 matmul is untouched by the new dtype.
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let c = a.matmul(&b);
    assert_eq!(&*c.data(), &[1.0, 2.0, 3.0, 4.0]);

    // F32 autograd tracking is untouched.
    let mut f = Tensor::new(vec![1.0, 2.0], vec![2]);
    f.set_requires_grad(true);
    assert!(f.requires_grad());
    f.set_requires_grad(false);
    assert!(!f.requires_grad());
}

// ---------------------------------------------------------------------------
// 10. API boundary: fixed tensors never enter autograd
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "fixed-point")]
fn set_requires_grad_true_on_fixed_panics() {
    let mut t = Tensor::from_i16_fixed(vec![1, 2, 3], vec![3], 8);
    t.set_requires_grad(true);
}

#[test]
fn set_requires_grad_false_on_fixed_is_noop() {
    // A fixed tensor is already AutogradState::None; explicitly clearing
    // requires_grad must be a clean no-op (no panic, no state change).
    let mut t = Tensor::from_i16_fixed(vec![1, 2, 3], vec![3], 8);
    assert!(!t.requires_grad());
    t.set_requires_grad(false);
    assert!(!t.requires_grad());
    assert!(t.is_fixed_point());
}

// ---------------------------------------------------------------------------
// 11. API boundary: contiguous weights/bias required
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "contiguous")]
fn from_parts_rejects_non_contiguous_weight() {
    // Build a contiguous 2x2 weight, then transpose it (metadata-only) — the
    // result has the FixedI16 dtype but is no longer contiguous, so reading
    // its raw storage as row-major would compute the wrong matmul.
    let w = Tensor::from_i16_fixed(vec![1, 2, 3, 4], vec![2, 2], 8);
    let w_t = w.transpose(0, 1);
    assert!(!w_t.is_contiguous());
    let _ = FixedLinear::from_parts(w_t, None);
}

// ---------------------------------------------------------------------------
// 12. API boundary: to_dtype rejects fixed-point cleanly
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "fixed-point")]
fn to_dtype_from_fixed_to_float_panics() {
    let t = Tensor::from_i16_fixed(vec![1, 2, 3], vec![3], 8);
    let _ = t.to_dtype(DType::F32);
}

#[test]
#[should_panic(expected = "fixed-point")]
fn to_dtype_from_float_to_fixed_panics() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let _ = t.to_dtype(DType::FixedI16 { scale_log2: 8 });
}
