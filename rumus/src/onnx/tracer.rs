// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Thread-local ONNX graph tracer.
//!
//! When active, tensor ops record `TracedNode` entries instead of (or in
//! addition to) their normal eager execution.  The tracer intercepts at
//! two levels:
//!
//! 1. **Primitive ops** (matmul, relu, etc.) — record individual ONNX nodes.
//! 2. **Module-level fusions** (Linear → Gemm, Conv2d → Conv) — suppress
//!    primitive recording and emit a single fused ONNX node.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::tensor::{DType, Tensor};

// ---------------------------------------------------------------------------
// Traced graph types
// ---------------------------------------------------------------------------

/// A single ONNX operator node captured during tracing.
#[derive(Debug, Clone)]
pub struct TracedNode {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: Vec<TracedAttribute>,
}

/// An ONNX attribute value.
#[derive(Debug, Clone)]
pub enum TracedAttribute {
    Int(String, i64),
    Float(String, f32),
    Ints(String, Vec<i64>),
    Floats(String, Vec<f32>),
    String(String, String),
}

/// Shape + dtype information for a traced value.
#[derive(Debug, Clone)]
pub struct ValueInfo {
    pub shape: Vec<usize>,
    pub dtype: DType,
}

/// The complete traced graph, ready for conversion to ONNX proto.
pub struct TracedGraph {
    pub nodes: Vec<TracedNode>,
    pub inputs: Vec<(String, ValueInfo)>,
    pub outputs: Vec<(String, ValueInfo)>,
    pub initializers: Vec<(String, Tensor)>,
}

// ---------------------------------------------------------------------------
// Thread-local tracer context
// ---------------------------------------------------------------------------

struct Tracer {
    nodes: Vec<TracedNode>,
    /// Maps storage pointer → value name for name propagation.
    value_names: HashMap<usize, String>,
    /// Shape + dtype per named value.
    value_info: HashMap<String, ValueInfo>,
    /// Collected parameter tensors.
    initializers: Vec<(String, Tensor)>,
    /// Monotonic counter for auto-naming.
    next_id: usize,
    /// When > 0, primitive op recording is suppressed (module-level fusion).
    suppress_depth: usize,
}

impl Tracer {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            value_names: HashMap::new(),
            value_info: HashMap::new(),
            initializers: Vec::new(),
            next_id: 0,
            suppress_depth: 0,
        }
    }

    fn fresh_name(&mut self, prefix: &str) -> String {
        let name = format!("{}_{}", prefix, self.next_id);
        self.next_id += 1;
        name
    }
}

thread_local! {
    static TRACER: RefCell<Option<Tracer>> = RefCell::new(None);
}

/// Returns `true` if the ONNX tracer is currently active.
pub fn is_tracing() -> bool {
    TRACER.with(|t| t.borrow().is_some())
}

/// Returns `true` if primitive op recording is suppressed (inside a fused module).
pub fn is_suppressed() -> bool {
    TRACER.with(|t| {
        t.borrow()
            .as_ref()
            .map_or(false, |tr| tr.suppress_depth > 0)
    })
}

/// Execute a closure with mutable access to the active tracer.
///
/// Panics if no tracer is active.
pub fn with_tracer<F, R>(f: F) -> R
where
    F: FnOnce(&mut TracerHandle<'_>) -> R,
{
    TRACER.with(|t| {
        let mut borrow = t.borrow_mut();
        let tracer = borrow.as_mut().expect("ONNX tracer not active");
        let mut handle = TracerHandle { inner: tracer };
        f(&mut handle)
    })
}

/// Safe handle to the tracer, providing the public recording API.
pub struct TracerHandle<'a> {
    inner: &'a mut Tracer,
}

impl TracerHandle<'_> {
    /// Get or assign a name for a tensor (identified by storage pointer).
    pub fn name_of(&mut self, tensor: &Tensor) -> String {
        let key = tensor.storage.ptr_id();
        if let Some(name) = self.inner.value_names.get(&key) {
            return name.clone();
        }
        let name = self.inner.fresh_name("val");
        self.inner.value_names.insert(key, name.clone());
        self.inner.value_info.insert(
            name.clone(),
            ValueInfo {
                shape: tensor.shape().to_vec(),
                dtype: tensor.dtype(),
            },
        );
        name
    }

    /// Assign a specific name to a tensor.
    pub fn set_name(&mut self, tensor: &Tensor, name: String) {
        let key = tensor.storage.ptr_id();
        self.inner.value_names.insert(key, name.clone());
        self.inner.value_info.insert(
            name,
            ValueInfo {
                shape: tensor.shape().to_vec(),
                dtype: tensor.dtype(),
            },
        );
    }

    /// Record a traced node.  Suppressed nodes are silently dropped.
    pub fn record_node(&mut self, node: TracedNode) {
        if self.inner.suppress_depth == 0 {
            self.inner.nodes.push(node);
        }
    }

    /// Register a parameter tensor as an initializer.
    pub fn add_initializer(&mut self, name: &str, tensor: &Tensor) {
        // Avoid duplicates.
        if !self.inner.initializers.iter().any(|(n, _)| n == name) {
            self.inner.initializers.push((name.to_string(), tensor.clone()));
        }
    }

    /// Enter a fused-module scope: suppress primitive op recording.
    pub fn enter_fusion(&mut self) {
        self.inner.suppress_depth += 1;
    }

    /// Exit a fused-module scope.
    pub fn leave_fusion(&mut self) {
        self.inner.suppress_depth = self.inner.suppress_depth.saturating_sub(1);
    }

    /// Allocate a fresh unique name.
    pub fn fresh_name(&mut self, prefix: &str) -> String {
        self.inner.fresh_name(prefix)
    }

    /// Register value info for a name.
    pub fn register_value(&mut self, name: &str, info: ValueInfo) {
        self.inner.value_info.insert(name.to_string(), info);
    }
}

// ---------------------------------------------------------------------------
// Public tracing API
// ---------------------------------------------------------------------------

/// Trace a model's forward pass and capture the ONNX computational graph.
///
/// Creates dummy input tensors, runs the forward function, and collects
/// the traced nodes + initializers.
///
/// # Arguments
///
/// - `state_dict`: parameter name → tensor map (from `model.state_dict("")`).
/// - `input_specs`: list of `(name, shape, dtype)` for each model input.
/// - `forward_fn`: closure executing the model's forward pass.
pub fn trace<F>(
    state_dict: &HashMap<String, Tensor>,
    input_specs: &[(&str, Vec<usize>, DType)],
    forward_fn: F,
) -> TracedGraph
where
    F: FnOnce(&[Tensor]) -> Tensor,
{
    // Create dummy inputs.
    let mut inputs = Vec::with_capacity(input_specs.len());
    let mut input_infos = Vec::new();

    // Initialize the tracer.
    let mut tracer = Tracer::new();

    // Pre-register all parameters as initializers and named values.
    for (name, tensor) in state_dict {
        let key = tensor.storage.ptr_id();
        tracer.value_names.insert(key, name.clone());
        tracer.value_info.insert(
            name.clone(),
            ValueInfo {
                shape: tensor.shape().to_vec(),
                dtype: tensor.dtype(),
            },
        );
        tracer.initializers.push((name.clone(), tensor.clone()));
    }

    for (name, shape, dtype) in input_specs {
        let numel: usize = shape.iter().product();
        let tensor = Tensor::new(vec![0.0f32; numel], shape.clone());
        let key = tensor.storage.ptr_id();
        tracer.value_names.insert(key, name.to_string());
        tracer.value_info.insert(
            name.to_string(),
            ValueInfo {
                shape: shape.clone(),
                dtype: *dtype,
            },
        );
        input_infos.push((name.to_string(), ValueInfo { shape: shape.clone(), dtype: *dtype }));
        inputs.push(tensor);
    }

    // Activate the tracer.
    TRACER.with(|t| *t.borrow_mut() = Some(tracer));

    // Run the forward pass — ops will record TracedNodes.
    let output = forward_fn(&inputs);

    // Deactivate and extract.
    let tracer = TRACER.with(|t| t.borrow_mut().take()).expect("tracer lost");

    let output_name = tracer
        .value_names
        .get(&output.storage.ptr_id())
        .cloned()
        .unwrap_or_else(|| "output_0".to_string());
    let output_info = ValueInfo {
        shape: output.shape().to_vec(),
        dtype: output.dtype(),
    };

    TracedGraph {
        nodes: tracer.nodes,
        inputs: input_infos,
        outputs: vec![(output_name, output_info)],
        initializers: tracer.initializers,
    }
}

// ---------------------------------------------------------------------------
// Convenience helpers for recording common ops
// ---------------------------------------------------------------------------

/// Record a simple unary op (Relu, Sigmoid, Tanh, etc.).
pub fn record_unary(input: &Tensor, output: &Tensor, op_type: &str) {
    if !is_tracing() || is_suppressed() {
        return;
    }
    with_tracer(|t| {
        let in_name = t.name_of(input);
        let out_name = t.name_of(output);
        t.record_node(TracedNode {
            op_type: op_type.to_string(),
            inputs: vec![in_name],
            outputs: vec![out_name],
            attributes: vec![],
        });
    });
}

/// Record a binary op (Add, Sub, Mul, MatMul, etc.).
pub fn record_binary(lhs: &Tensor, rhs: &Tensor, output: &Tensor, op_type: &str) {
    if !is_tracing() || is_suppressed() {
        return;
    }
    with_tracer(|t| {
        let l = t.name_of(lhs);
        let r = t.name_of(rhs);
        let o = t.name_of(output);
        t.record_node(TracedNode {
            op_type: op_type.to_string(),
            inputs: vec![l, r],
            outputs: vec![o],
            attributes: vec![],
        });
    });
}

/// Record a fused Gemm node (Linear layer: y = alpha*A@B + beta*C).
pub fn trace_linear(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    output: &Tensor,
    weight_name: &str,
    bias_name: Option<&str>,
) {
    if !is_tracing() {
        return;
    }
    with_tracer(|t| {
        let in_name = t.name_of(input);
        let out_name = t.name_of(output);
        t.add_initializer(weight_name, weight);
        let mut inputs = vec![in_name, weight_name.to_string()];
        if let (Some(b), Some(bn)) = (bias, bias_name) {
            t.add_initializer(bn, b);
            inputs.push(bn.to_string());
        }
        t.record_node(TracedNode {
            op_type: "Gemm".to_string(),
            inputs,
            outputs: vec![out_name],
            attributes: vec![
                TracedAttribute::Float("alpha".to_string(), 1.0),
                TracedAttribute::Float("beta".to_string(), 1.0),
                TracedAttribute::Int("transB".to_string(), 0), // weight is [in, out], no transpose needed
            ],
        });
    });
}

/// Record a fused Conv node.
pub fn trace_conv2d(
    input: &Tensor,
    output: &Tensor,
    weight_name: &str,
    weight: &Tensor,
    bias_name: Option<&str>,
    bias: Option<&Tensor>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    in_channels: usize,
    out_channels: usize,
) {
    if !is_tracing() {
        return;
    }
    with_tracer(|t| {
        let in_name = t.name_of(input);
        let out_name = t.name_of(output);

        // Register weight as initializer.
        // ONNX Conv expects weight shape [C_out, C_in, K, K].
        // Our weight is [C_out, C_in*K*K] — register with the ONNX-expected shape
        // in the value info (the raw data is correct when reshaped).
        t.add_initializer(weight_name, weight);
        t.register_value(weight_name, ValueInfo {
            shape: vec![out_channels, in_channels, kernel_size, kernel_size],
            dtype: weight.dtype(),
        });

        let mut inputs = vec![in_name, weight_name.to_string()];
        if let (Some(b), Some(bn)) = (bias, bias_name) {
            t.add_initializer(bn, b);
            inputs.push(bn.to_string());
        }

        t.record_node(TracedNode {
            op_type: "Conv".to_string(),
            inputs,
            outputs: vec![out_name],
            attributes: vec![
                TracedAttribute::Ints("kernel_shape".to_string(), vec![kernel_size as i64, kernel_size as i64]),
                TracedAttribute::Ints("strides".to_string(), vec![stride as i64, stride as i64]),
                TracedAttribute::Ints("pads".to_string(), vec![padding as i64, padding as i64, padding as i64, padding as i64]),
                TracedAttribute::Ints("dilations".to_string(), vec![1, 1]),
            ],
        });
    });
}
