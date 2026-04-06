// SPDX-License-Identifier: Apache-2.0 OR MIT
fn main() {
    if std::env::var("CARGO_FEATURE_ONNX").is_ok() {
        prost_build::compile_protos(&["proto/onnx.proto"], &["proto/"])
            .expect("Failed to compile onnx.proto");
    }
}
