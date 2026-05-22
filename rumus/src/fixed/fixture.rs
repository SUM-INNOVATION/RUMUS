// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Canonical, deterministic fixed-point proof-fixture export.
//!
//! A [`FixedFixture`] captures everything a downstream zero-knowledge backend
//! (e.g. a halo2 circuit) needs to reproduce a fixed-point inference: the
//! global scale, the rounding/overflow rules, the layer structure, the
//! weights and biases, and the canonical input/output pair.
//!
//! Serialization is **canonical JSON**: the field order is fixed by the
//! writer, all payloads are integers (no float formatting), and whitespace is
//! deterministic.  [`FixedFixture::to_canonical_bytes`] therefore produces
//! byte-identical output across repeated calls and across machines.
//!
//! The serializer is hand-written on purpose — the schema is small, closed,
//! and integer-only, so pulling in `serde`/`serde_json` would add several
//! transitive dependencies for no benefit and make the byte-stability
//! guarantee harder to audit.

use std::fmt::Write as _;
use std::io;
use std::path::Path;

use super::{FixedLinear, FixedOverflow, FixedRounding, FIXED_OVERFLOW, FIXED_ROUNDING};
use crate::tensor::{DType, Tensor};

/// Format identifier embedded in every fixture.
const FIXTURE_FORMAT: &str = "rumus-fixed-fixture";
/// Schema version embedded in every fixture.
const FIXTURE_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Layer specification
// ---------------------------------------------------------------------------

/// A single layer captured in a [`FixedFixture`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FixedLayerSpec {
    /// A dense layer: `[in_features, out_features]` row-major weights and an
    /// optional `[out_features]` bias.
    Linear {
        in_features: usize,
        out_features: usize,
        weight: Vec<i16>,
        bias: Option<Vec<i16>>,
    },
    /// An element-wise integer ReLU.
    Relu,
}

// ---------------------------------------------------------------------------
// FixedFixture
// ---------------------------------------------------------------------------

/// A complete, serializable fixed-point model fixture.
///
/// Build one with [`FixedFixtureBuilder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedFixture {
    scale_log2: u8,
    rounding: FixedRounding,
    overflow: FixedOverflow,
    layers: Vec<FixedLayerSpec>,
    input: Vec<i16>,
    input_shape: Vec<usize>,
    output: Vec<i16>,
    output_shape: Vec<usize>,
}

impl FixedFixture {
    /// The global fixed-point scale exponent.
    pub fn scale_log2(&self) -> u8 {
        self.scale_log2
    }

    /// The global fixed-point scale, `2^scale_log2`.
    pub fn scale(&self) -> u32 {
        1u32 << self.scale_log2
    }

    /// The captured layers, in execution order.
    pub fn layers(&self) -> &[FixedLayerSpec] {
        &self.layers
    }

    /// Serialize to canonical JSON.
    ///
    /// The output is deterministic: identical inputs always yield identical
    /// bytes.  Field order is fixed and every numeric payload is an integer.
    pub fn to_canonical_json(&self) -> String {
        let mut s = String::new();
        s.push_str("{\n");

        write_str(&mut s, 1, "format", FIXTURE_FORMAT, true);
        write_u64(&mut s, 1, "version", FIXTURE_VERSION as u64, true);
        write_str(&mut s, 1, "dtype", "FixedI16", true);
        write_u64(&mut s, 1, "scale_log2", self.scale_log2 as u64, true);
        write_u64(&mut s, 1, "scale", self.scale() as u64, true);
        write_str(&mut s, 1, "rounding", self.rounding.as_str(), true);
        write_str(&mut s, 1, "overflow", self.overflow.as_str(), true);

        // Layers.
        indent(&mut s, 1);
        s.push_str("\"layers\": [");
        if self.layers.is_empty() {
            s.push_str("],\n");
        } else {
            s.push('\n');
            for (i, layer) in self.layers.iter().enumerate() {
                let last = i + 1 == self.layers.len();
                write_layer(&mut s, layer, last);
            }
            indent(&mut s, 1);
            s.push_str("],\n");
        }

        write_usize_array(&mut s, 1, "input_shape", &self.input_shape, true);
        write_i16_array(&mut s, 1, "input", &self.input, true);
        write_usize_array(&mut s, 1, "output_shape", &self.output_shape, true);
        write_i16_array(&mut s, 1, "output", &self.output, false);

        s.push_str("}\n");
        s
    }

    /// Serialize to canonical JSON bytes.  See [`to_canonical_json`](Self::to_canonical_json).
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        self.to_canonical_json().into_bytes()
    }

    /// Write the canonical fixture bytes to `path`.
    pub fn write(&self, path: impl AsRef<Path>) -> io::Result<()> {
        std::fs::write(path, self.to_canonical_bytes())
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Incremental builder for a [`FixedFixture`].
///
/// Every layer added must share the builder's `scale_log2`.
pub struct FixedFixtureBuilder {
    scale_log2: u8,
    layers: Vec<FixedLayerSpec>,
}

impl FixedFixtureBuilder {
    /// Start a new fixture with the given global scale exponent.
    pub fn new(scale_log2: u8) -> Self {
        Self { scale_log2, layers: Vec::new() }
    }

    /// Append a dense layer, copying its weight and bias out of the
    /// [`FixedLinear`].
    ///
    /// # Panics
    ///
    /// Panics if the layer's `scale_log2` differs from the builder's.
    pub fn add_linear(&mut self, layer: &FixedLinear) -> &mut Self {
        assert_eq!(
            layer.scale_log2, self.scale_log2,
            "FixedFixtureBuilder: layer scale_log2 {} != fixture scale_log2 {}",
            layer.scale_log2, self.scale_log2,
        );
        let weight = layer.weight.fixed_i16_data().to_vec();
        let bias = layer.bias.as_ref().map(|b| b.fixed_i16_data().to_vec());
        self.layers.push(FixedLayerSpec::Linear {
            in_features: layer.in_features,
            out_features: layer.out_features,
            weight,
            bias,
        });
        self
    }

    /// Append an element-wise ReLU layer.
    pub fn add_relu(&mut self) -> &mut Self {
        self.layers.push(FixedLayerSpec::Relu);
        self
    }

    /// Finalize the fixture with its canonical `input` / `output` pair.
    ///
    /// # Panics
    ///
    /// Panics unless both tensors are `FixedI16` at the builder's `scale_log2`.
    pub fn finish(self, input: &Tensor, output: &Tensor) -> FixedFixture {
        check_io(input, "input", self.scale_log2);
        check_io(output, "output", self.scale_log2);
        FixedFixture {
            scale_log2: self.scale_log2,
            rounding: FIXED_ROUNDING,
            overflow: FIXED_OVERFLOW,
            layers: self.layers,
            input: input.fixed_i16_data().to_vec(),
            input_shape: input.shape().to_vec(),
            output: output.fixed_i16_data().to_vec(),
            output_shape: output.shape().to_vec(),
        }
    }
}

fn check_io(t: &Tensor, name: &str, scale_log2: u8) {
    match t.dtype() {
        DType::FixedI16 { scale_log2: s } => assert_eq!(
            s, scale_log2,
            "FixedFixtureBuilder: {} scale_log2 {} != fixture scale_log2 {}",
            name, s, scale_log2,
        ),
        other => panic!("FixedFixtureBuilder: {} must be FixedI16, got {:?}", name, other),
    }
}

// ---------------------------------------------------------------------------
// Canonical JSON writers
// ---------------------------------------------------------------------------

fn indent(s: &mut String, level: usize) {
    for _ in 0..level {
        s.push_str("  ");
    }
}

fn write_str(s: &mut String, level: usize, key: &str, val: &str, comma: bool) {
    indent(s, level);
    write!(s, "\"{}\": \"{}\"", key, val).unwrap();
    s.push_str(if comma { ",\n" } else { "\n" });
}

fn write_u64(s: &mut String, level: usize, key: &str, val: u64, comma: bool) {
    indent(s, level);
    write!(s, "\"{}\": {}", key, val).unwrap();
    s.push_str(if comma { ",\n" } else { "\n" });
}

fn write_null(s: &mut String, level: usize, key: &str, comma: bool) {
    indent(s, level);
    write!(s, "\"{}\": null", key).unwrap();
    s.push_str(if comma { ",\n" } else { "\n" });
}

fn write_i16_array(s: &mut String, level: usize, key: &str, arr: &[i16], comma: bool) {
    indent(s, level);
    write!(s, "\"{}\": [", key).unwrap();
    for (i, v) in arr.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        write!(s, "{}", v).unwrap();
    }
    s.push(']');
    s.push_str(if comma { ",\n" } else { "\n" });
}

fn write_usize_array(s: &mut String, level: usize, key: &str, arr: &[usize], comma: bool) {
    indent(s, level);
    write!(s, "\"{}\": [", key).unwrap();
    for (i, v) in arr.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        write!(s, "{}", v).unwrap();
    }
    s.push(']');
    s.push_str(if comma { ",\n" } else { "\n" });
}

fn write_layer(s: &mut String, layer: &FixedLayerSpec, last: bool) {
    indent(s, 2);
    s.push_str("{\n");
    match layer {
        FixedLayerSpec::Linear { in_features, out_features, weight, bias } => {
            write_str(s, 3, "kind", "linear", true);
            write_u64(s, 3, "in_features", *in_features as u64, true);
            write_u64(s, 3, "out_features", *out_features as u64, true);
            write_usize_array(s, 3, "weight_shape", &[*in_features, *out_features], true);
            write_i16_array(s, 3, "weight", weight, true);
            match bias {
                Some(b) => {
                    write_usize_array(s, 3, "bias_shape", &[*out_features], true);
                    write_i16_array(s, 3, "bias", b, false);
                }
                None => {
                    write_null(s, 3, "bias_shape", true);
                    write_null(s, 3, "bias", false);
                }
            }
        }
        FixedLayerSpec::Relu => {
            write_str(s, 3, "kind", "relu", false);
        }
    }
    indent(s, 2);
    s.push_str(if last { "}\n" } else { "},\n" });
}
