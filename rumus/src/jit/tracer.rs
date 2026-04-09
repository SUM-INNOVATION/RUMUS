// SPDX-License-Identifier: Apache-2.0 OR MIT
//! JIT fusion tracer: captures element-wise ops into a fuseable IR.
//!
//! When active, element-wise ops record `FusedOp` entries instead of
//! dispatching individual GPU kernels.  The fusion block is flushed
//! (codegen + compile + dispatch) when `compile()` scope exits.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::tensor::{DType, StorageHandle, Tensor};

// ---------------------------------------------------------------------------
// Fused IR
// ---------------------------------------------------------------------------

pub type VarId = usize;

/// A single fused operation in the IR.
#[derive(Debug, Clone)]
pub enum FusedOp {
    Input(VarId),
    Add(VarId, VarId, VarId),               // dst, lhs, rhs
    Sub(VarId, VarId, VarId),
    Mul(VarId, VarId, VarId),
    Relu(VarId, VarId),                     // dst, src
    Sigmoid(VarId, VarId),
    Tanh(VarId, VarId),
    Gelu(VarId, VarId),
    LeakyRelu(VarId, VarId, f32),           // dst, src, alpha
    Scale(VarId, VarId, f32),               // dst, src, scalar
    Neg(VarId, VarId),
    Output(VarId),
}

/// A hashable tag for cache key construction.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum FusedOpTag {
    Input,
    Add, Sub, Mul,
    Relu, Sigmoid, Tanh, Gelu,
    LeakyRelu(u32),     // f32::to_bits()
    Scale(u32),         // f32::to_bits()
    Neg,
    Output,
}

impl FusedOp {
    pub fn tag(&self) -> FusedOpTag {
        match self {
            FusedOp::Input(_) => FusedOpTag::Input,
            FusedOp::Add(..) => FusedOpTag::Add,
            FusedOp::Sub(..) => FusedOpTag::Sub,
            FusedOp::Mul(..) => FusedOpTag::Mul,
            FusedOp::Relu(..) => FusedOpTag::Relu,
            FusedOp::Sigmoid(..) => FusedOpTag::Sigmoid,
            FusedOp::Tanh(..) => FusedOpTag::Tanh,
            FusedOp::Gelu(..) => FusedOpTag::Gelu,
            FusedOp::LeakyRelu(_, _, a) => FusedOpTag::LeakyRelu(a.to_bits()),
            FusedOp::Scale(_, _, s) => FusedOpTag::Scale(s.to_bits()),
            FusedOp::Neg(..) => FusedOpTag::Neg,
            FusedOp::Output(_) => FusedOpTag::Output,
        }
    }
}

/// A complete fusion block ready for codegen.
pub struct FusionBlock {
    pub ops: Vec<FusedOp>,
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub numel: usize,
    pub dtype: DType,
    /// StorageHandles for input tensors (in binding order).
    pub input_storages: Vec<StorageHandle>,
    /// StorageHandles for output tensors (deferred, to be materialized).
    pub output_storages: Vec<StorageHandle>,
}

// ---------------------------------------------------------------------------
// Thread-local tracer
// ---------------------------------------------------------------------------

pub struct JitTracer {
    ops: Vec<FusedOp>,
    /// Maps storage ptr_id → VarId for input tracking.
    tensor_map: HashMap<usize, VarId>,
    input_storages: Vec<StorageHandle>,
    output_storages: Vec<StorageHandle>,
    next_var: VarId,
    num_inputs: usize,
    num_outputs: usize,
    numel: usize,
    dtype: DType,
}

impl JitTracer {
    fn new() -> Self {
        Self {
            ops: Vec::new(),
            tensor_map: HashMap::new(),
            input_storages: Vec::new(),
            output_storages: Vec::new(),
            next_var: 0,
            num_inputs: 0,
            num_outputs: 0,
            numel: 0,
            dtype: DType::F32,
        }
    }

    fn fresh_var(&mut self) -> VarId {
        let v = self.next_var;
        self.next_var += 1;
        v
    }

    /// Get or create a VarId for an input tensor.
    pub fn get_or_create_input(&mut self, tensor: &Tensor) -> VarId {
        let key = tensor.storage.ptr_id();
        if let Some(&v) = self.tensor_map.get(&key) {
            return v;
        }
        let v = self.fresh_var();
        self.tensor_map.insert(key, v);
        self.ops.push(FusedOp::Input(v));
        self.input_storages.push(tensor.storage.clone());
        self.num_inputs += 1;

        // Set numel/dtype from first input.
        if self.num_inputs == 1 {
            self.numel = tensor.numel();
            self.dtype = tensor.dtype();
        }
        v
    }

    /// Record a unary op and return (output VarId, deferred StorageHandle).
    pub fn record_unary(
        &mut self,
        src: &Tensor,
        make_op: impl FnOnce(VarId, VarId) -> FusedOp,
    ) -> (VarId, StorageHandle) {
        let src_v = self.resolve_var(src);
        let dst_v = self.fresh_var();
        self.ops.push(make_op(dst_v, src_v));

        let storage = StorageHandle::new_deferred(dst_v, self.numel, self.dtype);
        self.tensor_map.insert(storage.ptr_id(), dst_v);
        (dst_v, storage)
    }

    /// Record a binary op and return (output VarId, deferred StorageHandle).
    pub fn record_binary(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        make_op: impl FnOnce(VarId, VarId, VarId) -> FusedOp,
    ) -> (VarId, StorageHandle) {
        let lhs_v = self.resolve_var(lhs);
        let rhs_v = self.resolve_var(rhs);
        let dst_v = self.fresh_var();
        self.ops.push(make_op(dst_v, lhs_v, rhs_v));

        let storage = StorageHandle::new_deferred(dst_v, self.numel, self.dtype);
        self.tensor_map.insert(storage.ptr_id(), dst_v);
        (dst_v, storage)
    }

    /// Resolve a tensor to its VarId — if it's a deferred tensor, look up
    /// by ptr_id; otherwise register it as a new input.
    fn resolve_var(&mut self, tensor: &Tensor) -> VarId {
        let key = tensor.storage.ptr_id();
        if let Some(&v) = self.tensor_map.get(&key) {
            return v;
        }
        // Not seen before → it's an external input.
        self.get_or_create_input(tensor)
    }

    /// Mark a VarId as an output and track its deferred storage for materialization.
    pub fn mark_output(&mut self, var_id: VarId, storage: &StorageHandle) {
        self.ops.push(FusedOp::Output(var_id));
        self.output_storages.push(storage.clone());
        self.num_outputs += 1;
    }

    /// Convert to a FusionBlock for codegen.
    pub fn into_block(self) -> FusionBlock {
        FusionBlock {
            ops: self.ops,
            num_inputs: self.num_inputs,
            num_outputs: self.num_outputs,
            numel: self.numel,
            dtype: self.dtype,
            input_storages: self.input_storages,
            output_storages: self.output_storages,
        }
    }
}

thread_local! {
    static JIT_TRACER: RefCell<Option<JitTracer>> = RefCell::new(None);
}

/// Returns true if JIT tracing is active.
pub fn is_tracing() -> bool {
    JIT_TRACER.with(|t| t.borrow().is_some())
}

/// Execute a closure with mutable access to the active tracer.
pub fn with_tracer<F, R>(f: F) -> R
where
    F: FnOnce(&mut JitTracer) -> R,
{
    JIT_TRACER.with(|t| {
        let mut borrow = t.borrow_mut();
        let tracer = borrow.as_mut().expect("JIT tracer not active");
        f(tracer)
    })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Trace a block of element-wise ops and fuse them into a single GPU kernel.
///
/// Autograd tape recording proceeds normally inside the closure — only the
/// forward GPU dispatch is intercepted.  The returned tensor is backed by
/// real GPU memory (materialized during flush).
///
/// # Example
///
/// ```ignore
/// let output = jit::compile(|| {
///     let h = x.add(&bias);
///     h.relu()
/// });
/// ```
pub fn compile<F>(f: F) -> Tensor
where
    F: FnOnce() -> Tensor,
{
    // Activate the tracer.
    JIT_TRACER.with(|t| *t.borrow_mut() = Some(JitTracer::new()));

    // Run user code — element-wise ops record FusedOps + return deferred tensors.
    // Autograd tape is recorded normally.
    let result = f();

    // Extract the tracer.
    let mut tracer = JIT_TRACER.with(|t| t.borrow_mut().take().expect("JIT tracer lost"));

    // Mark the result tensor as an output.
    let result_var = tracer.resolve_var(&result);
    tracer.mark_output(result_var, &result.storage);

    // Flush: codegen → cache lookup → dispatch → materialize deferred storages.
    let block = tracer.into_block();
    super::flush_block(block);

    result
}
