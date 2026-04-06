// SPDX-License-Identifier: Apache-2.0 OR MIT
//! ONNX model export: converts a `TracedGraph` into a serialized `.onnx` file.

use std::collections::HashMap;

use prost::Message;

use crate::nn::Module;
use crate::onnx::proto::onnx as pb;
use crate::onnx::tracer::{self, TracedAttribute, TracedGraph, TracedNode, ValueInfo};
use crate::tensor::{DType, Tensor};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during ONNX export.
#[derive(Debug)]
pub enum OnnxError {
    Io(std::io::Error),
    UnsupportedDType(DType),
    ShapeError(String),
}

impl From<std::io::Error> for OnnxError {
    fn from(e: std::io::Error) -> Self {
        OnnxError::Io(e)
    }
}

impl std::fmt::Display for OnnxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OnnxError::Io(e) => write!(f, "IO error: {}", e),
            OnnxError::UnsupportedDType(dt) => write!(f, "Unsupported dtype: {:?}", dt),
            OnnxError::ShapeError(msg) => write!(f, "Shape error: {}", msg),
        }
    }
}

impl std::error::Error for OnnxError {}

// ---------------------------------------------------------------------------
// ONNX data type constants
// ---------------------------------------------------------------------------

const ONNX_FLOAT: i32 = 1;
const ONNX_FLOAT16: i32 = 10;

fn dtype_to_onnx(dtype: DType) -> i32 {
    match dtype {
        DType::F32 => ONNX_FLOAT,
        DType::F16 => ONNX_FLOAT16,
        // Q8 tensors are dequantized to F32 for ONNX export.
        DType::Q8 { .. } => ONNX_FLOAT,
    }
}

// ---------------------------------------------------------------------------
// Tensor → TensorProto
// ---------------------------------------------------------------------------

/// Convert a RUMUS tensor to an ONNX TensorProto with raw byte data.
///
/// - **F32**: downloads f32 bytes via `data()`.
/// - **F16**: downloads raw f16 bytes directly via `download_raw_bytes()`
///   to preserve native half-precision without casting to f32.
/// - **Q8**: dequantizes to F32 via `data()` (ONNX doesn't support our
///   symmetric Q8 format natively).
fn tensor_to_proto(name: &str, tensor: &Tensor, shape_override: Option<&[usize]>) -> pb::TensorProto {
    let shape = shape_override.unwrap_or(tensor.shape());

    match tensor.dtype() {
        DType::F16 => {
            // Preserve native F16 bytes — do NOT cast to F32.
            #[cfg(feature = "gpu")]
            let raw_bytes = tensor.storage.download_raw_bytes();
            #[cfg(not(feature = "gpu"))]
            let raw_bytes = {
                let guard = tensor.storage.data();
                bytemuck::cast_slice(&*guard).to_vec()
            };

            pb::TensorProto {
                dims: shape.iter().map(|&d| d as i64).collect(),
                data_type: ONNX_FLOAT16,
                name: name.to_string(),
                raw_data: raw_bytes,
                ..Default::default()
            }
        }
        _ => {
            // F32 and Q8 (Q8 auto-dequantizes to F32 via data()).
            let guard = tensor.storage.data();
            let raw_bytes: Vec<u8> = bytemuck::cast_slice(&*guard).to_vec();
            drop(guard);

            pb::TensorProto {
                dims: shape.iter().map(|&d| d as i64).collect(),
                data_type: ONNX_FLOAT,
                name: name.to_string(),
                raw_data: raw_bytes,
                ..Default::default()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ValueInfo → ValueInfoProto
// ---------------------------------------------------------------------------

fn value_info_to_proto(name: &str, info: &ValueInfo) -> pb::ValueInfoProto {
    let elem_type = dtype_to_onnx(info.dtype);

    let shape = pb::TensorShapeProto {
        dim: info
            .shape
            .iter()
            .map(|&d| pb::tensor_shape_proto::Dimension {
                denotation: String::new(),
                value: Some(pb::tensor_shape_proto::dimension::Value::DimValue(d as i64)),
            })
            .collect(),
    };

    pb::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(pb::TypeProto {
            denotation: String::new(),
            value: Some(pb::type_proto::Value::TensorType(pb::type_proto::Tensor {
                elem_type,
                shape: Some(shape),
            })),
        }),
        doc_string: String::new(),
    }
}

// ---------------------------------------------------------------------------
// TracedAttribute → AttributeProto
// ---------------------------------------------------------------------------

fn attr_to_proto(attr: &TracedAttribute) -> pb::AttributeProto {
    match attr {
        TracedAttribute::Int(name, val) => pb::AttributeProto {
            name: name.clone(),
            r#type: 2, // INT
            i: *val,
            ..Default::default()
        },
        TracedAttribute::Float(name, val) => pb::AttributeProto {
            name: name.clone(),
            r#type: 1, // FLOAT
            f: *val,
            ..Default::default()
        },
        TracedAttribute::Ints(name, vals) => pb::AttributeProto {
            name: name.clone(),
            r#type: 7, // INTS
            ints: vals.clone(),
            ..Default::default()
        },
        TracedAttribute::Floats(name, vals) => pb::AttributeProto {
            name: name.clone(),
            r#type: 6, // FLOATS
            floats: vals.clone(),
            ..Default::default()
        },
        TracedAttribute::String(name, val) => pb::AttributeProto {
            name: name.clone(),
            r#type: 3, // STRING
            s: val.clone(),
            ..Default::default()
        },
    }
}

// ---------------------------------------------------------------------------
// TracedNode → NodeProto
// ---------------------------------------------------------------------------

fn node_to_proto(node: &TracedNode, idx: usize) -> pb::NodeProto {
    pb::NodeProto {
        input: node.inputs.clone(),
        output: node.outputs.clone(),
        name: format!("{}_{}", node.op_type, idx),
        op_type: node.op_type.clone(),
        domain: String::new(),
        attribute: node.attributes.iter().map(attr_to_proto).collect(),
        doc_string: String::new(),
    }
}

// ---------------------------------------------------------------------------
// TracedGraph → GraphProto
// ---------------------------------------------------------------------------

fn graph_to_proto(graph: &TracedGraph, value_shapes: &HashMap<String, ValueInfo>) -> pb::GraphProto {
    let nodes: Vec<pb::NodeProto> = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| node_to_proto(n, i))
        .collect();

    let inputs: Vec<pb::ValueInfoProto> = graph
        .inputs
        .iter()
        .map(|(name, info)| value_info_to_proto(name, info))
        .collect();

    let outputs: Vec<pb::ValueInfoProto> = graph
        .outputs
        .iter()
        .map(|(name, info)| value_info_to_proto(name, info))
        .collect();

    let initializers: Vec<pb::TensorProto> = graph
        .initializers
        .iter()
        .map(|(name, tensor)| {
            // Use override shape from value_shapes if available (e.g., Conv weight reshape).
            let shape_override = value_shapes.get(name).map(|vi| vi.shape.as_slice());
            tensor_to_proto(name, tensor, shape_override)
        })
        .collect();

    pb::GraphProto {
        node: nodes,
        name: "rumus_graph".to_string(),
        initializer: initializers,
        doc_string: String::new(),
        input: inputs,
        output: outputs,
    }
}

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

fn build_model_proto(graph: &TracedGraph, value_shapes: &HashMap<String, ValueInfo>, opset: u32) -> pb::ModelProto {
    pb::ModelProto {
        ir_version: 9,
        opset_import: vec![pb::OperatorSetIdProto {
            domain: String::new(),
            version: opset as i64,
        }],
        producer_name: "rumus".to_string(),
        producer_version: env!("CARGO_PKG_VERSION").to_string(),
        domain: String::new(),
        model_version: 1,
        doc_string: String::new(),
        graph: Some(graph_to_proto(graph, value_shapes)),
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Export a RUMUS model to an ONNX file.
///
/// # Arguments
///
/// - `model`: any type implementing `Module` (for `state_dict`).
/// - `input_specs`: `(name, shape, dtype)` for each model input.
/// - `path`: output `.onnx` file path.
/// - `forward_fn`: closure that runs the model's forward pass.
///
/// # Example
///
/// ```ignore
/// export_onnx(
///     &model,
///     &[("input", vec![1, 3, 224, 224], DType::F32)],
///     "model.onnx",
///     |inputs| model.forward(&inputs[0]),
/// )?;
/// ```
pub fn export_onnx<M, F>(
    model: &M,
    input_specs: &[(&str, Vec<usize>, DType)],
    path: &str,
    forward_fn: F,
) -> Result<(), OnnxError>
where
    M: Module,
    F: FnOnce(&[Tensor]) -> Tensor,
{
    export_onnx_with_opset(model, input_specs, path, forward_fn, 17)
}

/// Export with a specific ONNX opset version.
pub fn export_onnx_with_opset<M, F>(
    model: &M,
    input_specs: &[(&str, Vec<usize>, DType)],
    path: &str,
    forward_fn: F,
    opset: u32,
) -> Result<(), OnnxError>
where
    M: Module,
    F: FnOnce(&[Tensor]) -> Tensor,
{
    // Collect state dict for initializers.
    let state_dict = model.state_dict("");

    // Trace the forward pass.
    let graph = tracer::trace(&state_dict, input_specs, forward_fn);

    // Build value shape map from tracer's value_info (read from thread-local
    // during trace — we extract what we need from the graph initializers).
    let mut value_shapes = HashMap::new();
    // Re-read the tracer's value_info isn't possible after take(), but the
    // graph outputs and inputs already have their ValueInfo.
    // For initializer shape overrides, we rely on what trace_conv2d registered.
    // The simple approach: just use tensor shapes as-is.
    for (name, tensor) in &graph.initializers {
        value_shapes.insert(
            name.clone(),
            ValueInfo {
                shape: tensor.shape().to_vec(),
                dtype: tensor.dtype(),
            },
        );
    }

    let model_proto = build_model_proto(&graph, &value_shapes, opset);
    let bytes = model_proto.encode_to_vec();
    std::fs::write(path, bytes)?;
    Ok(())
}
