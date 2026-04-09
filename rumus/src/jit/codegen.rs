// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Dynamic WGSL code generation from a `FusionBlock`.
//!
//! Produces a straight-line `@compute` kernel where each thread processes
//! one element.  Each `FusedOp` maps to one line of WGSL.

use crate::jit::tracer::{FusedOp, FusionBlock};

/// Generate a WGSL shader source string for the given fusion block.
///
/// The `scalar` alias is NOT prepended here — the caller must run
/// `preprocess_shader()` to inject the correct alias for the block's dtype.
pub fn generate_wgsl(block: &FusionBlock) -> String {
    let mut src = String::with_capacity(1024);

    // --- Bindings: inputs (read-only) ---
    let mut input_idx = 0usize;
    let mut output_idx = 0usize;
    let mut binding = 0u32;

    // Count inputs and outputs to pre-allocate binding numbers.
    for op in &block.ops {
        if matches!(op, FusedOp::Input(_)) {
            src.push_str(&format!(
                "@group(0) @binding({}) var<storage, read> in_{}: array<scalar>;\n",
                binding, input_idx,
            ));
            input_idx += 1;
            binding += 1;
        }
    }

    // Output bindings.
    for i in 0..block.num_outputs {
        src.push_str(&format!(
            "@group(0) @binding({}) var<storage, read_write> out_{}: array<scalar>;\n",
            binding, i,
        ));
        binding += 1;
    }

    // Uniform binding (numel).
    src.push_str(&format!(
        "struct FusedParams {{ numel: u32, _p0: u32, _p1: u32, _p2: u32, }}\n\
         @group(0) @binding({}) var<uniform> fused_params: FusedParams;\n\n",
        binding,
    ));
    let _uniform_binding = binding;

    // --- Entry point ---
    src.push_str(
        "@compute @workgroup_size(256)\n\
         fn fused_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
         \tlet idx = gid.x;\n\
         \tif (idx >= fused_params.numel) { return; }\n\n",
    );

    // --- Body: one line per op ---
    input_idx = 0;
    output_idx = 0;

    for op in &block.ops {
        match op {
            FusedOp::Input(v) => {
                src.push_str(&format!("\tlet v{} = in_{}[idx];\n", v, input_idx));
                input_idx += 1;
            }
            FusedOp::Add(d, l, r) => {
                src.push_str(&format!("\tlet v{} = v{} + v{};\n", d, l, r));
            }
            FusedOp::Sub(d, l, r) => {
                src.push_str(&format!("\tlet v{} = v{} - v{};\n", d, l, r));
            }
            FusedOp::Mul(d, l, r) => {
                src.push_str(&format!("\tlet v{} = v{} * v{};\n", d, l, r));
            }
            FusedOp::Relu(d, s) => {
                src.push_str(&format!("\tlet v{} = max(scalar(0.0), v{});\n", d, s));
            }
            FusedOp::Sigmoid(d, s) => {
                src.push_str(&format!(
                    "\tlet v{} = scalar(1.0) / (scalar(1.0) + exp(-v{}));\n",
                    d, s,
                ));
            }
            FusedOp::Tanh(d, s) => {
                src.push_str(&format!("\tlet v{} = tanh(v{});\n", d, s));
            }
            FusedOp::Gelu(d, s) => {
                src.push_str(&format!(
                    "\tlet g{d}_inner = scalar(0.7978845608) * (v{s} + scalar(0.044715) * v{s} * v{s} * v{s});\n\
                     \tlet g{d}_t = tanh(g{d}_inner);\n\
                     \tlet v{d} = scalar(0.5) * v{s} * (scalar(1.0) + g{d}_t);\n",
                    d = d, s = s,
                ));
            }
            FusedOp::LeakyRelu(d, s, alpha) => {
                src.push_str(&format!(
                    "\tlet v{} = select(scalar({:e}) * v{}, v{}, v{} > scalar(0.0));\n",
                    d, alpha, s, s, s,
                ));
            }
            FusedOp::Scale(d, s, scalar) => {
                src.push_str(&format!(
                    "\tlet v{} = v{} * scalar({:e});\n",
                    d, s, scalar,
                ));
            }
            FusedOp::Neg(d, s) => {
                src.push_str(&format!("\tlet v{} = -v{};\n", d, s));
            }
            FusedOp::Output(v) => {
                src.push_str(&format!("\tout_{}[idx] = v{};\n", output_idx, v));
                output_idx += 1;
            }
        }
    }

    src.push_str("}\n");
    src
}
