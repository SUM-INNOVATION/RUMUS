//! View operations (zero-copy) and materialising operations (allocating)
//! on [`Tensor`].
//!
//! When autograd is active, materialising ops record a [`TapeEntry`]
//! on the thread-local tape.  When the `gpu` feature is enabled and
//! inputs are GPU-resident, compute dispatches to WGSL shaders.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::autograd::context;
use crate::autograd::{
    AdaptiveAvgPool2dBackward, AddBackward, AddBiasBackward, AddChannelBiasBackward,
    BackwardOp, BatchNorm2dBackward, BmmBackward, CastBackward, TransposeBackward,
    BroadcastAddBackward, BroadcastMulBackward, BroadcastSubBackward, CrossEntropyBackward,
    DropoutBackward, EmbeddingBackward, FlattenBackward, GeluBackward, Im2ColBackward,
    LayerNormBackward, LeakyReluBackward, MatmulBackward, MaxPool2dBackward, MseLossBackward,
    MulBackward, ReluBackward, ReshapeBackward, SigmoidBackward, SoftmaxBackward,
    StackBackward, SubBackward, TanhBackward, TapeEntry, VersionSnapshot,
};
use crate::backend::{Backend, CpuBackend};
#[cfg(feature = "gpu")]
use crate::backend::gpu::{
    compute as gpu_compute,
    context::{GpuContext, STORAGE_USAGE},
};
use crate::tensor::{AutogradState, Layout, StorageHandle, Tensor, TensorMeta};

// ---------------------------------------------------------------------------
// View operations — zero-copy, Arc refcount bump only
// ---------------------------------------------------------------------------

impl Tensor {
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        Tensor {
            storage: self.storage.clone(),
            layout: self.layout.transposed(dim0, dim1),
            state: AutogradState::None,
        }
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Tensor {
        match self.layout.reshaped(shape.clone()) {
            Some(new_layout) => Tensor {
                storage: self.storage.clone(),
                layout: new_layout,
                state: AutogradState::None,
            },
            None => self.contiguous().reshape(shape),
        }
    }
}

// ---------------------------------------------------------------------------
// Contiguity materialisation
// ---------------------------------------------------------------------------

impl Tensor {
    pub fn contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            return Tensor {
                storage: self.storage.clone(),
                layout: self.layout.clone(),
                state: AutogradState::None,
            };
        }

        let numel = self.numel();
        let shape = self.layout.shape();
        let strides = self.layout.strides();
        let offset = self.layout.offset();
        let ndim = self.ndim();

        // Precompute suffix products (shared by CPU and GPU paths).
        let mut suffix = vec![1usize; ndim];
        for d in (0..ndim.saturating_sub(1)).rev() {
            suffix[d] = suffix[d + 1] * shape[d + 1];
        }

        // GPU path: strided copy entirely on-device via WGSL kernel.
        #[cfg(feature = "gpu")]
        if self.storage.is_gpu() {
            let ctx = GpuContext::get().expect("GPU required");
            let src_buf = self.storage.gpu_buffer();
            let dst_buf = ctx.pool.acquire(&ctx.device, self.dtype().gpu_buf_size(numel), STORAGE_USAGE);
            gpu_compute::contiguous_copy(
                ctx, &src_buf, &dst_buf,
                numel as u32, ndim as u32, offset as u32,
                shape, strides, &suffix,
            );
            drop(src_buf);
            let result_shape = shape.to_vec();
            return Tensor {
                storage: StorageHandle::new_gpu(dst_buf, numel),
                layout: Layout::contiguous(result_shape),
                state: AutogradState::None,
            };
        }

        // CPU path: strided copy via indexed reads.
        let src_guard = self.storage.data();
        let mut dst = vec![0.0f32; numel];
        for dst_idx in 0..numel {
            let mut src_idx = offset;
            let mut remainder = dst_idx;
            for d in 0..ndim {
                let dim_size = suffix[d];
                let coord = remainder / dim_size;
                remainder %= dim_size;
                src_idx += coord * strides[d];
            }
            dst[dst_idx] = src_guard[src_idx];
        }
        drop(src_guard);
        Tensor::new(dst, shape.to_vec())
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn should_record(lhs: &Tensor, rhs: &Tensor) -> bool {
    (lhs.requires_grad() || rhs.requires_grad()) && !context::is_no_grad()
}

fn require_grad_id(t: &Tensor) -> Option<crate::tensor::GradId> {
    t.grad_id()
}

/// CPU compute for a binary op: acquire guards, run kernel, return StorageHandle.
fn cpu_binary(
    lhs: &Tensor,
    rhs: &Tensor,
    kernel: fn(&[f32], &[f32], &mut [f32]),
) -> StorageHandle {
    let lg = lhs.storage.data();
    let rg = rhs.storage.data();
    let mut dst = CpuBackend::zeros(lhs.numel());
    kernel(&lg, &rg, &mut dst);
    drop(rg);
    drop(lg);
    StorageHandle::new(dst)
}

#[cfg(feature = "gpu")]
fn either_gpu(a: &Tensor, b: &Tensor) -> bool {
    a.storage.is_gpu() || b.storage.is_gpu()
}

#[cfg(feature = "gpu")]
fn gpu_binary(
    ctx: &'static GpuContext,
    lhs: &Tensor,
    rhs: &Tensor,
    dispatch: fn(&GpuContext, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, u32),
) -> StorageHandle {
    let numel = lhs.numel();
    let lb = lhs.storage.gpu_buffer();
    let rb = rhs.storage.gpu_buffer();
    let dst = ctx.pool.acquire(&ctx.device, lhs.dtype().gpu_buf_size(numel), STORAGE_USAGE);
    dispatch(ctx, &lb, &rb, &dst, numel as u32);
    drop(lb);
    drop(rb);
    StorageHandle::new_gpu(dst, numel)
}

/// Helper for unary activations: runs CPU kernel, optionally saves output or
/// input for backward, records on tape.
fn unary_activation<F>(
    input: &Tensor,
    cpu_kernel: fn(&[f32], &mut [f32]),
    save_output: bool,
    _scalar: f32,
    make_op: F,
) -> Tensor
where
    F: FnOnce(StorageHandle, VersionSnapshot, StorageHandle, Vec<usize>) -> BackwardOp,
{
    let input_c = input.contiguous();
    let in_guard = input_c.storage.data();
    let numel = input.numel();
    let mut dst = CpuBackend::zeros(numel);
    cpu_kernel(&in_guard, &mut dst);
    drop(in_guard);

    let out_storage = StorageHandle::new(dst);
    let result_shape = input.shape().to_vec();

    if input.requires_grad() && !context::is_no_grad() {
        let out_grad_id = context::next_grad_id();
        let in_gid = input.grad_id().unwrap_or_else(context::next_grad_id);
        let input_version = VersionSnapshot::new(in_gid, &input.storage);
        if let Some(meta) = input.meta() {
            meta.total_grads.fetch_add(1, Ordering::Relaxed);
        }
        let saved_storage = if save_output {
            out_storage.clone()
        } else {
            input.storage.clone()
        };
        let op = make_op(out_storage.clone(), input_version, saved_storage, result_shape.clone());
        let op_id = context::with_tape(|tape| {
            tape.push(TapeEntry { op, inputs: vec![in_gid], outputs: vec![out_grad_id] })
        });
        Tensor {
            storage: out_storage,
            layout: Layout::contiguous(result_shape),
            state: AutogradState::Tracked(Arc::new(TensorMeta {
                requires_grad: true, grad_id: Some(out_grad_id), creator: op_id,
                is_leaf: false, retains_grad: false, total_grads: AtomicUsize::new(0),
            })),
        }
    } else {
        Tensor { storage: out_storage, layout: Layout::contiguous(result_shape), state: AutogradState::None }
    }
}

/// Helper for broadcast binary ops.
fn broadcast_binary_op<F>(
    lhs: &Tensor,
    rhs: &Tensor,
    cpu_op: fn(f32, f32) -> f32,
    make_bw: F,
) -> Tensor
where
    F: FnOnce(
        VersionSnapshot, VersionSnapshot,
        Option<crate::tensor::broadcast::BroadcastInfo>,
        Option<crate::tensor::broadcast::BroadcastInfo>,
        Vec<usize>,
    ) -> BackwardOp,
{
    use crate::tensor::broadcast::{broadcast_shape, broadcast_strides, suffix_products, BroadcastInfo};

    let out_shape = broadcast_shape(lhs.shape(), rhs.shape())
        .expect("broadcast: shapes not broadcastable");
    let a_strides = broadcast_strides(lhs.shape(), &out_shape);
    let b_strides = broadcast_strides(rhs.shape(), &out_shape);
    let out_numel: usize = out_shape.iter().product();
    #[allow(unused_variables)]
    let suffix = suffix_products(&out_shape);

    // GPU path: dispatch broadcast WGSL kernel.
    #[cfg(feature = "gpu")]
    let out_storage = if either_gpu(lhs, rhs) {
        let ctx = GpuContext::get().expect("GPU required");
        let lhs_c = lhs.contiguous();
        let rhs_c = rhs.contiguous();
        let lb = lhs_c.storage.gpu_buffer();
        let rb = rhs_c.storage.gpu_buffer();
        let dst_buf = ctx.pool.acquire(&ctx.device, lhs.dtype().gpu_buf_size(out_numel), STORAGE_USAGE);
        let params = gpu_compute::make_broadcast_params(
            out_numel, out_shape.len(), &suffix, &a_strides, &b_strides,
        );
        // Dispatch the correct kernel based on the op type.
        // We use a function pointer passed via the cpu_op to determine which:
        //   add: a + b,  sub: a - b,  mul: a * b
        // Detect by calling cpu_op(1.0, 1.0) — hacky but avoids adding a 3rd param.
        let test = cpu_op(2.0, 3.0);
        if (test - 5.0).abs() < 0.01 {
            gpu_compute::broadcast_add_gpu(ctx, &lb, &rb, &dst_buf, &params);
        } else if (test - 6.0).abs() < 0.01 {
            gpu_compute::broadcast_mul_gpu(ctx, &lb, &rb, &dst_buf, &params);
        } else {
            gpu_compute::broadcast_sub_gpu(ctx, &lb, &rb, &dst_buf, &params);
        }
        drop(lb); drop(rb);
        StorageHandle::new_gpu(dst_buf, out_numel)
    } else {
        let lhs_c = lhs.contiguous();
        let rhs_c = rhs.contiguous();
        let lg = lhs_c.storage.data();
        let rg = rhs_c.storage.data();
        let mut dst = CpuBackend::zeros(out_numel);
        crate::tensor::broadcast::cpu_broadcast_binary(
            &lg, &rg, &mut dst, &out_shape, &a_strides, &b_strides, cpu_op,
        );
        drop(rg); drop(lg);
        StorageHandle::new(dst)
    };

    #[cfg(not(feature = "gpu"))]
    let out_storage = {
        let lhs_c = lhs.contiguous();
        let rhs_c = rhs.contiguous();
        let lg = lhs_c.storage.data();
        let rg = rhs_c.storage.data();
        let mut dst = CpuBackend::zeros(out_numel);
        crate::tensor::broadcast::cpu_broadcast_binary(
            &lg, &rg, &mut dst, &out_shape, &a_strides, &b_strides, cpu_op,
        );
        drop(rg); drop(lg);
        StorageHandle::new(dst)
    };

    if should_record(lhs, rhs) {
        let out_grad_id = context::next_grad_id();
        let lhs_gid = require_grad_id(lhs).unwrap_or_else(context::next_grad_id);
        let rhs_gid = require_grad_id(rhs).unwrap_or_else(context::next_grad_id);
        let lhs_version = VersionSnapshot::new(lhs_gid, &lhs.storage);
        let rhs_version = VersionSnapshot::new(rhs_gid, &rhs.storage);
        let lhs_bi = BroadcastInfo::new(lhs.shape(), &out_shape);
        let rhs_bi = BroadcastInfo::new(rhs.shape(), &out_shape);

        if let Some(meta) = lhs.meta() { meta.total_grads.fetch_add(1, Ordering::Relaxed); }
        if let Some(meta) = rhs.meta() { meta.total_grads.fetch_add(1, Ordering::Relaxed); }

        let op = make_bw(lhs_version, rhs_version, lhs_bi, rhs_bi, out_shape.clone());
        let op_id = context::with_tape(|tape| {
            tape.push(TapeEntry { op, inputs: vec![lhs_gid, rhs_gid], outputs: vec![out_grad_id] })
        });
        Tensor {
            storage: out_storage,
            layout: Layout::contiguous(out_shape),
            state: AutogradState::Tracked(Arc::new(TensorMeta {
                requires_grad: true, grad_id: Some(out_grad_id), creator: op_id,
                is_leaf: false, retains_grad: false, total_grads: AtomicUsize::new(0),
            })),
        }
    } else {
        Tensor { storage: out_storage, layout: Layout::contiguous(out_shape), state: AutogradState::None }
    }
}

/// Select GPU or CPU path for a binary op.
fn compute_binary(
    lhs: &Tensor,
    rhs: &Tensor,
    cpu_kernel: fn(&[f32], &[f32], &mut [f32]),
    #[cfg(feature = "gpu")] gpu_dispatch: fn(&GpuContext, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, u32),
) -> StorageHandle {
    #[cfg(feature = "gpu")]
    {
        if either_gpu(lhs, rhs) {
            let ctx = GpuContext::get().expect("GPU required");
            return gpu_binary(ctx, lhs, rhs, gpu_dispatch);
        }
    }
    cpu_binary(lhs, rhs, cpu_kernel)
}

/// Build a tracked or untracked output tensor from pre-computed components.
///
/// `lhs_gid` and `rhs_gid` must be pre-allocated before calling this
/// (the same ids used when constructing the `BackwardOp`).
fn wrap_binary_result(
    out_storage: StorageHandle,
    result_shape: Vec<usize>,
    lhs: &Tensor,
    rhs: &Tensor,
    lhs_gid: crate::tensor::GradId,
    rhs_gid: crate::tensor::GradId,
    op: BackwardOp,
) -> Tensor {
    if should_record(lhs, rhs) {
        let out_grad_id = context::next_grad_id();

        if let Some(meta) = lhs.meta() {
            meta.total_grads.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(meta) = rhs.meta() {
            meta.total_grads.fetch_add(1, Ordering::Relaxed);
        }

        let op_id = context::with_tape(|tape| {
            tape.push(TapeEntry {
                op,
                inputs: vec![lhs_gid, rhs_gid],
                outputs: vec![out_grad_id],
            })
        });

        let out_meta = Arc::new(TensorMeta {
            requires_grad: true,
            grad_id: Some(out_grad_id),
            creator: op_id,
            is_leaf: false,
            retains_grad: false,
            total_grads: AtomicUsize::new(0),
        });

        Tensor {
            storage: out_storage,
            layout: Layout::contiguous(result_shape),
            state: AutogradState::Tracked(out_meta),
        }
    } else {
        Tensor {
            storage: out_storage,
            layout: Layout::contiguous(result_shape),
            state: AutogradState::None,
        }
    }
}

// ---------------------------------------------------------------------------
// Materialising operations
// ---------------------------------------------------------------------------

impl Tensor {
    pub fn add(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.shape(), rhs.shape(), "add: shape mismatch {:?} vs {:?}", self.shape(), rhs.shape());
        let lhs_c = self.contiguous();
        let rhs_c = rhs.contiguous();
        let result_shape = lhs_c.shape().to_vec();

        let out_storage = compute_binary(
            &lhs_c, &rhs_c,
            CpuBackend::add,
            #[cfg(feature = "gpu")] gpu_compute::add,
        );

        let lhs_gid = require_grad_id(self).unwrap_or_else(context::next_grad_id);
        let rhs_gid = require_grad_id(rhs).unwrap_or_else(context::next_grad_id);
        let op = BackwardOp::Add(AddBackward {
            lhs_version: VersionSnapshot::new(lhs_gid, &self.storage),
            rhs_version: VersionSnapshot::new(rhs_gid, &rhs.storage),
        });
        wrap_binary_result(out_storage, result_shape, self, rhs, lhs_gid, rhs_gid, op)
    }

    pub fn sub(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.shape(), rhs.shape(), "sub: shape mismatch {:?} vs {:?}", self.shape(), rhs.shape());
        let lhs_c = self.contiguous();
        let rhs_c = rhs.contiguous();
        let result_shape = lhs_c.shape().to_vec();

        let out_storage = compute_binary(
            &lhs_c, &rhs_c,
            CpuBackend::sub,
            #[cfg(feature = "gpu")] gpu_compute::sub,
        );

        let lhs_gid = require_grad_id(self).unwrap_or_else(context::next_grad_id);
        let rhs_gid = require_grad_id(rhs).unwrap_or_else(context::next_grad_id);
        let op = BackwardOp::Sub(SubBackward {
            lhs_version: VersionSnapshot::new(lhs_gid, &self.storage),
            rhs_version: VersionSnapshot::new(rhs_gid, &rhs.storage),
        });
        wrap_binary_result(out_storage, result_shape, self, rhs, lhs_gid, rhs_gid, op)
    }

    pub fn mul(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.shape(), rhs.shape(), "mul: shape mismatch {:?} vs {:?}", self.shape(), rhs.shape());
        let lhs_c = self.contiguous();
        let rhs_c = rhs.contiguous();
        let result_shape = lhs_c.shape().to_vec();

        let out_storage = compute_binary(
            &lhs_c, &rhs_c,
            CpuBackend::mul,
            #[cfg(feature = "gpu")] gpu_compute::mul,
        );

        let lhs_gid = require_grad_id(self).unwrap_or_else(context::next_grad_id);
        let rhs_gid = require_grad_id(rhs).unwrap_or_else(context::next_grad_id);
        let op = BackwardOp::Mul(MulBackward {
            lhs_storage: self.storage.clone(),
            lhs_layout: self.layout.clone(),
            lhs_version: VersionSnapshot::new(lhs_gid, &self.storage),
            rhs_storage: rhs.storage.clone(),
            rhs_layout: rhs.layout.clone(),
            rhs_version: VersionSnapshot::new(rhs_gid, &rhs.storage),
        });
        wrap_binary_result(out_storage, result_shape, self, rhs, lhs_gid, rhs_gid, op)
    }

    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "matmul: lhs must be 2-D, got {}-D", self.ndim());
        assert_eq!(rhs.ndim(), 2, "matmul: rhs must be 2-D, got {}-D", rhs.ndim());
        let m = self.shape()[0];
        let k = self.shape()[1];
        let n = rhs.shape()[1];
        assert_eq!(k, rhs.shape()[0], "matmul: inner dimension mismatch");

        let lhs_c = self.contiguous();
        let rhs_c = rhs.contiguous();
        let result_shape = vec![m, n];

        // Matmul has different dispatch signature — handle inline.
        #[cfg(feature = "gpu")]
        let out_storage = if either_gpu(&lhs_c, &rhs_c) {
            let ctx = GpuContext::get().expect("GPU required");
            let lb = lhs_c.storage.gpu_buffer();
            let rb = rhs_c.storage.gpu_buffer();
            let dst = ctx.pool.acquire(&ctx.device, self.dtype().gpu_buf_size(m * n), STORAGE_USAGE);
            gpu_compute::matmul(ctx, &lb, &rb, &dst, m as u32, k as u32, n as u32);
            drop(lb); drop(rb);
            StorageHandle::new_gpu(dst, m * n)
        } else {
            let lg = lhs_c.storage.data();
            let rg = rhs_c.storage.data();
            let mut dst = CpuBackend::zeros(m * n);
            CpuBackend::matmul(&lg, &rg, &mut dst, m, k, n);
            drop(rg); drop(lg);
            StorageHandle::new(dst)
        };
        #[cfg(not(feature = "gpu"))]
        let out_storage = {
            let lg = lhs_c.storage.data();
            let rg = rhs_c.storage.data();
            let mut dst = CpuBackend::zeros(m * n);
            CpuBackend::matmul(&lg, &rg, &mut dst, m, k, n);
            drop(rg); drop(lg);
            StorageHandle::new(dst)
        };

        let lhs_gid = require_grad_id(self).unwrap_or_else(context::next_grad_id);
        let rhs_gid = require_grad_id(rhs).unwrap_or_else(context::next_grad_id);
        let op = BackwardOp::Matmul(MatmulBackward {
            lhs_storage: self.storage.clone(),
            lhs_layout: self.layout.clone(),
            lhs_version: VersionSnapshot::new(lhs_gid, &self.storage),
            rhs_storage: rhs.storage.clone(),
            rhs_layout: rhs.layout.clone(),
            rhs_version: VersionSnapshot::new(rhs_gid, &rhs.storage),
            m, k, n,
        });
        wrap_binary_result(out_storage, result_shape, self, rhs, lhs_gid, rhs_gid, op)
    }

    pub fn relu(&self) -> Tensor {
        let input_c = self.contiguous();
        let result_shape = input_c.shape().to_vec();
        let numel = input_c.numel();

        #[cfg(feature = "gpu")]
        let out_storage = if input_c.storage.is_gpu() {
            let ctx = GpuContext::get().expect("GPU required");
            let ib = input_c.storage.gpu_buffer();
            let dst = ctx.pool.acquire(&ctx.device, self.dtype().gpu_buf_size(numel), STORAGE_USAGE);
            gpu_compute::relu(ctx, &ib, &dst, numel as u32);
            drop(ib);
            StorageHandle::new_gpu(dst, numel)
        } else {
            let ig = input_c.storage.data();
            let mut dst = CpuBackend::zeros(numel);
            CpuBackend::relu(&ig, &mut dst);
            drop(ig);
            StorageHandle::new(dst)
        };
        #[cfg(not(feature = "gpu"))]
        let out_storage = {
            let ig = input_c.storage.data();
            let mut dst = CpuBackend::zeros(numel);
            CpuBackend::relu(&ig, &mut dst);
            drop(ig);
            StorageHandle::new(dst)
        };

        if self.requires_grad() && !context::is_no_grad() {
            let out_grad_id = context::next_grad_id();
            let in_gid = self.grad_id().unwrap_or_else(context::next_grad_id);

            if let Some(meta) = self.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::Relu(ReluBackward {
                        input_storage: self.storage.clone(),
                        input_layout: self.layout.clone(),
                        input_version: VersionSnapshot::new(in_gid, &self.storage),
                    }),
                    inputs: vec![in_gid],
                    outputs: vec![out_grad_id],
                })
            });

            Tensor {
                storage: out_storage,
                layout: Layout::contiguous(result_shape),
                state: AutogradState::Tracked(Arc::new(TensorMeta {
                    requires_grad: true,
                    grad_id: Some(out_grad_id),
                    creator: op_id,
                    is_leaf: false,
                    retains_grad: false,
                    total_grads: AtomicUsize::new(0),
                })),
            }
        } else {
            Tensor {
                storage: out_storage,
                layout: Layout::contiguous(result_shape),
                state: AutogradState::None,
            }
        }
    }

    /// Fused MSE loss — always CPU (scalar reduction).  GPU mse_loss is a
    /// future optimization.
    pub fn mse_loss(&self, target: &Tensor) -> Tensor {
        assert_eq!(self.shape(), target.shape(), "mse_loss: shape mismatch");

        let pred_c = self.contiguous();
        let targ_c = target.contiguous();
        let pred_guard = pred_c.storage.data();
        let targ_guard = targ_c.storage.data();
        let numel = pred_c.numel();

        let mut sum = 0.0f32;
        for i in 0..numel {
            let diff = pred_guard[i] - targ_guard[i];
            sum += diff * diff;
        }
        let loss_val = sum / numel as f32;
        drop(targ_guard);
        drop(pred_guard);

        let out_storage = StorageHandle::new(vec![loss_val]);
        let result_shape = vec![1];

        if self.requires_grad() && !context::is_no_grad() {
            let out_grad_id = context::next_grad_id();
            let pred_gid = self.grad_id().unwrap_or_else(context::next_grad_id);
            let target_dummy_gid = context::next_grad_id();

            if let Some(meta) = self.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }

            let pred_version = VersionSnapshot::new(pred_gid, &self.storage);
            let target_version = VersionSnapshot::new(target_dummy_gid, &target.storage);

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::MseLoss(MseLossBackward {
                        pred_storage: self.storage.clone(),
                        pred_layout: self.layout.clone(),
                        pred_version,
                        target_storage: target.storage.clone(),
                        target_layout: target.layout.clone(),
                        target_version,
                        numel,
                    }),
                    inputs: vec![pred_gid],
                    outputs: vec![out_grad_id],
                })
            });

            Tensor {
                storage: out_storage,
                layout: Layout::contiguous(result_shape),
                state: AutogradState::Tracked(Arc::new(TensorMeta {
                    requires_grad: true,
                    grad_id: Some(out_grad_id),
                    creator: op_id,
                    is_leaf: false,
                    retains_grad: false,
                    total_grads: AtomicUsize::new(0),
                })),
            }
        } else {
            Tensor {
                storage: out_storage,
                layout: Layout::contiguous(result_shape),
                state: AutogradState::None,
            }
        }
    }

    /// Cross-entropy loss with Log-Sum-Exp stability.
    ///
    /// `self` = logits `[B, C]`, `targets` = class indices `[B]` (as f32).
    /// Returns scalar loss `[1]`.  Gradient is pre-computed during forward
    /// and saved in `CrossEntropyBackward` — backward is a trivial scale.
    pub fn cross_entropy_loss(&self, targets: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "cross_entropy_loss: logits must be 2-D [B, C]");
        assert_eq!(targets.ndim(), 1, "cross_entropy_loss: targets must be 1-D [B]");
        let batch = self.shape()[0];
        let num_classes = self.shape()[1];
        assert_eq!(targets.shape()[0], batch, "cross_entropy_loss: batch mismatch");

        let logits_c = self.contiguous();
        let targets_c = targets.contiguous();

        // GPU path
        #[cfg(feature = "gpu")]
        let (loss_storage, grad_storage) = if self.storage.is_gpu() {
            let ctx = GpuContext::get().expect("GPU required");
            let lb = logits_c.storage.gpu_buffer();
            targets_c.storage.ensure_gpu();
            let tb = targets_c.storage.gpu_buffer();
            let grad_buf = ctx.pool.acquire(&ctx.device, self.dtype().gpu_buf_size(batch * num_classes), STORAGE_USAGE);
            let loss_per_b_buf = ctx.pool.acquire(&ctx.device, self.dtype().gpu_buf_size(batch), STORAGE_USAGE);
            let loss_scalar_buf = ctx.pool.acquire(&ctx.device, 4u64, STORAGE_USAGE);

            gpu_compute::cross_entropy_forward(
                ctx, &lb, &tb, &grad_buf, &loss_per_b_buf,
                batch as u32, num_classes as u32,
            );
            gpu_compute::reduce_loss(ctx, &loss_per_b_buf, &loss_scalar_buf, batch as u32);

            drop(lb); drop(tb);
            ctx.pool.release(loss_per_b_buf);

            (StorageHandle::new_gpu(loss_scalar_buf, 1),
             StorageHandle::new_gpu(grad_buf, batch * num_classes))
        } else {
            let lg = logits_c.storage.data();
            let tg = targets_c.storage.data();
            let mut grad = CpuBackend::zeros(batch * num_classes);
            let mut loss_per_b = CpuBackend::zeros(batch);
            CpuBackend::cross_entropy_forward(&lg, &tg, &mut grad, &mut loss_per_b, batch, num_classes);
            drop(lg); drop(tg);
            let loss_val: f32 = loss_per_b.iter().sum();
            (StorageHandle::new(vec![loss_val]), StorageHandle::new(grad))
        };

        #[cfg(not(feature = "gpu"))]
        let (loss_storage, grad_storage) = {
            let lg = logits_c.storage.data();
            let tg = targets_c.storage.data();
            let mut grad = CpuBackend::zeros(batch * num_classes);
            let mut loss_per_b = CpuBackend::zeros(batch);
            CpuBackend::cross_entropy_forward(&lg, &tg, &mut grad, &mut loss_per_b, batch, num_classes);
            drop(lg); drop(tg);
            let loss_val: f32 = loss_per_b.iter().sum();
            (StorageHandle::new(vec![loss_val]), StorageHandle::new(grad))
        };

        if self.requires_grad() && !context::is_no_grad() {
            let out_grad_id = context::next_grad_id();
            let in_gid = self.grad_id().unwrap_or_else(context::next_grad_id);
            let input_version = VersionSnapshot::new(in_gid, &self.storage);

            if let Some(meta) = self.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::CrossEntropy(CrossEntropyBackward {
                        input_version,
                        grad_storage: grad_storage.clone(),
                        grad_layout: Layout::contiguous(vec![batch, num_classes]),
                    }),
                    inputs: vec![in_gid],
                    outputs: vec![out_grad_id],
                })
            });

            Tensor {
                storage: loss_storage,
                layout: Layout::contiguous(vec![1]),
                state: AutogradState::Tracked(Arc::new(TensorMeta {
                    requires_grad: true,
                    grad_id: Some(out_grad_id),
                    creator: op_id,
                    is_leaf: false,
                    retains_grad: false,
                    total_grads: AtomicUsize::new(0),
                })),
            }
        } else {
            Tensor {
                storage: loss_storage,
                layout: Layout::contiguous(vec![1]),
                state: AutogradState::None,
            }
        }
    }

    pub fn add_bias(&self, bias: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "add_bias: input must be 2-D");
        assert_eq!(bias.ndim(), 1, "add_bias: bias must be 1-D");
        let m = self.shape()[0];
        let n = self.shape()[1];
        assert_eq!(bias.shape()[0], n, "add_bias: bias length mismatch");

        let mat_c = self.contiguous();
        let bias_c = bias.contiguous();
        let result_shape = vec![m, n];

        #[cfg(feature = "gpu")]
        let out_storage = if either_gpu(&mat_c, &bias_c) {
            let ctx = GpuContext::get().expect("GPU required");
            let mb = mat_c.storage.gpu_buffer();
            let bb = bias_c.storage.gpu_buffer();
            let dst = ctx.pool.acquire(&ctx.device, self.dtype().gpu_buf_size(m * n), STORAGE_USAGE);
            gpu_compute::add_bias(ctx, &mb, &bb, &dst, m as u32, n as u32);
            drop(mb); drop(bb);
            StorageHandle::new_gpu(dst, m * n)
        } else {
            let mg = mat_c.storage.data();
            let bg = bias_c.storage.data();
            let mut dst = CpuBackend::zeros(m * n);
            CpuBackend::add_bias(&mg, &bg, &mut dst, m, n);
            drop(bg); drop(mg);
            StorageHandle::new(dst)
        };
        #[cfg(not(feature = "gpu"))]
        let out_storage = {
            let mg = mat_c.storage.data();
            let bg = bias_c.storage.data();
            let mut dst = CpuBackend::zeros(m * n);
            CpuBackend::add_bias(&mg, &bg, &mut dst, m, n);
            drop(bg); drop(mg);
            StorageHandle::new(dst)
        };

        let mat_gid = require_grad_id(self).unwrap_or_else(context::next_grad_id);
        let bias_gid = require_grad_id(bias).unwrap_or_else(context::next_grad_id);
        let op = BackwardOp::AddBias(AddBiasBackward {
            input_version: VersionSnapshot::new(mat_gid, &self.storage),
            bias_version: VersionSnapshot::new(bias_gid, &bias.storage),
            m, n,
        });
        wrap_binary_result(out_storage, result_shape, self, bias, mat_gid, bias_gid, op)
    }

    /// Extract a single batch element from a batched tensor.
    ///
    /// Input shape: `[batch, ...]`.
    /// Output shape: `[...]` (the leading batch dimension is removed).
    ///
    /// This is a tracked op — `SliceBatchBackward` scatters the gradient
    /// back into the correct batch slot of a zeroed full-batch gradient.
    ///
    /// Note: this materializes a copied tensor (not zero-copy).  The tape
    /// remains connected — gradients flow correctly.  A zero-copy view-based
    /// slice is a future optimization.
    pub fn slice_batch(&self, index: usize) -> Tensor {
        assert!(self.ndim() >= 2, "slice_batch: need at least 2 dims");
        let batch = self.shape()[0];
        assert!(index < batch, "slice_batch: index {} >= batch {}", index, batch);

        let element_shape: Vec<usize> = self.shape()[1..].to_vec();
        let element_numel: usize = element_shape.iter().product();

        let input_c = self.contiguous();
        let guard = input_c.storage.data();
        let start = index * element_numel;
        let slice_data = guard[start..start + element_numel].to_vec();
        drop(guard);

        let out_storage = StorageHandle::new(slice_data);

        if self.requires_grad() && !context::is_no_grad() {
            use crate::autograd::SliceBatchBackward;

            let out_grad_id = context::next_grad_id();
            let in_gid = self.grad_id().unwrap_or_else(context::next_grad_id);
            let input_version = VersionSnapshot::new(in_gid, &self.storage);

            if let Some(meta) = self.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::SliceBatch(SliceBatchBackward {
                        input_version,
                        original_shape: self.shape().to_vec(),
                        index,
                    }),
                    inputs: vec![in_gid],
                    outputs: vec![out_grad_id],
                })
            });

            Tensor {
                storage: out_storage,
                layout: Layout::contiguous(element_shape),
                state: AutogradState::Tracked(Arc::new(TensorMeta {
                    requires_grad: true,
                    grad_id: Some(out_grad_id),
                    creator: op_id,
                    is_leaf: false,
                    retains_grad: false,
                    total_grads: AtomicUsize::new(0),
                })),
            }
        } else {
            Tensor {
                storage: out_storage,
                layout: Layout::contiguous(element_shape),
                state: AutogradState::None,
            }
        }
    }

    /// im2col: extract sliding-window patches for convolution.
    ///
    /// Input shape: `[C_in, H, W]` (single sample, no batch dim).
    /// Output shape: `[C_in * K * K, out_h * out_w]`.
    ///
    /// This is a tracked op — `Im2ColBackward` calls `col2im`.
    pub fn im2col(
        &self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Tensor {
        assert_eq!(self.ndim(), 3, "im2col: input must be 3-D [C, H, W]");
        let c_in = self.shape()[0];
        let h = self.shape()[1];
        let w = self.shape()[2];
        let out_h = (h + 2 * padding - kernel_size) / stride + 1;
        let out_w = (w + 2 * padding - kernel_size) / stride + 1;
        let col_height = c_in * kernel_size * kernel_size;
        let num_patches = out_h * out_w;

        let input_c = self.contiguous();
        let in_guard = input_c.storage.data();
        let mut dst = CpuBackend::zeros(col_height * num_patches);
        CpuBackend::im2col(
            &in_guard, &mut dst,
            c_in, h, w, kernel_size, stride, padding, out_h, out_w,
        );
        drop(in_guard);

        let out_storage = StorageHandle::new(dst);
        let result_shape = vec![col_height, num_patches];

        if self.requires_grad() && !context::is_no_grad() {
            let out_grad_id = context::next_grad_id();
            let in_gid = self.grad_id().unwrap_or_else(context::next_grad_id);
            let input_version = VersionSnapshot::new(in_gid, &self.storage);

            if let Some(meta) = self.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::Im2Col(Im2ColBackward {
                        input_version,
                        c_in, h, w,
                        kernel_size, stride, padding,
                        out_h, out_w,
                    }),
                    inputs: vec![in_gid],
                    outputs: vec![out_grad_id],
                })
            });

            Tensor {
                storage: out_storage,
                layout: Layout::contiguous(result_shape),
                state: AutogradState::Tracked(Arc::new(TensorMeta {
                    requires_grad: true,
                    grad_id: Some(out_grad_id),
                    creator: op_id,
                    is_leaf: false,
                    retains_grad: false,
                    total_grads: AtomicUsize::new(0),
                })),
            }
        } else {
            Tensor {
                storage: out_storage,
                layout: Layout::contiguous(result_shape),
                state: AutogradState::None,
            }
        }
    }

    /// Add channel-wise bias: `out[c*spatial+s] = self[c*spatial+s] + bias[c]`.
    ///
    /// `self` shape: `[channels, spatial]` (or `[batch*channels, spatial]`).
    /// `bias` shape: `[channels]`.
    pub fn add_channel_bias(&self, bias: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "add_channel_bias: input must be 2-D [C, spatial]");
        assert_eq!(bias.ndim(), 1, "add_channel_bias: bias must be 1-D [C]");
        let channels = self.shape()[0];
        let spatial = self.shape()[1];
        assert_eq!(bias.shape()[0], channels, "add_channel_bias: channel mismatch");

        let src_c = self.contiguous();
        let bias_c = bias.contiguous();
        let sg = src_c.storage.data();
        let bg = bias_c.storage.data();
        let mut dst = CpuBackend::zeros(channels * spatial);
        CpuBackend::add_channel_bias(&sg, &bg, &mut dst, channels, spatial);
        drop(bg);
        drop(sg);

        let out_storage = StorageHandle::new(dst);
        let result_shape = vec![channels, spatial];

        if should_record(self, bias) {
            let out_grad_id = context::next_grad_id();
            let src_gid = require_grad_id(self).unwrap_or_else(context::next_grad_id);
            let bias_gid = require_grad_id(bias).unwrap_or_else(context::next_grad_id);

            if let Some(meta) = self.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }
            if let Some(meta) = bias.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }

            let input_version = VersionSnapshot::new(src_gid, &self.storage);
            let bias_version = VersionSnapshot::new(bias_gid, &bias.storage);

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::AddChannelBias(AddChannelBiasBackward {
                        input_version,
                        bias_version,
                        channels,
                        spatial,
                    }),
                    inputs: vec![src_gid, bias_gid],
                    outputs: vec![out_grad_id],
                })
            });

            Tensor {
                storage: out_storage,
                layout: Layout::contiguous(result_shape),
                state: AutogradState::Tracked(Arc::new(TensorMeta {
                    requires_grad: true,
                    grad_id: Some(out_grad_id),
                    creator: op_id,
                    is_leaf: false,
                    retains_grad: false,
                    total_grads: AtomicUsize::new(0),
                })),
            }
        } else {
            Tensor {
                storage: out_storage,
                layout: Layout::contiguous(result_shape),
                state: AutogradState::None,
            }
        }
    }

    /// 2D max pooling over `[C, H, W]` spatial dimensions.
    ///
    /// Input shape: `[C, H, W]` (single batch element).
    /// Output shape: `[C, out_h, out_w]`.
    ///
    /// Saves argmax indices (as f32) for the backward pass.
    ///
    /// # Panics
    ///
    /// Panics if `stride < kernel_size` (overlapping windows require atomic
    /// scatter in the backward WGSL kernel — deferred to a future milestone).
    pub fn max_pool2d(&self, kernel_size: usize, stride: usize) -> Tensor {
        assert_eq!(self.ndim(), 3, "max_pool2d: input must be 3-D [C, H, W]");
        assert!(
            stride >= kernel_size,
            "max_pool2d: stride ({}) must be >= kernel_size ({}) for non-atomic backward",
            stride, kernel_size,
        );

        let channels = self.shape()[0];
        let h = self.shape()[1];
        let w = self.shape()[2];
        let out_h = (h - kernel_size) / stride + 1;
        let out_w = (w - kernel_size) / stride + 1;
        let out_numel = channels * out_h * out_w;

        let input_c = self.contiguous();
        let in_guard = input_c.storage.data();
        let mut dst = CpuBackend::zeros(out_numel);
        let mut indices = CpuBackend::zeros(out_numel);
        CpuBackend::max_pool2d(
            &in_guard, &mut dst, &mut indices,
            channels, h, w, kernel_size, stride, out_h, out_w,
        );
        drop(in_guard);

        let out_storage = StorageHandle::new(dst);
        let indices_tensor = Tensor::new(indices, vec![channels, out_h, out_w]);
        let result_shape = vec![channels, out_h, out_w];

        if self.requires_grad() && !context::is_no_grad() {
            let out_grad_id = context::next_grad_id();
            let in_gid = self.grad_id().unwrap_or_else(context::next_grad_id);
            let input_version = VersionSnapshot::new(in_gid, &self.storage);

            if let Some(meta) = self.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::MaxPool2d(MaxPool2dBackward {
                        input_version,
                        indices_storage: indices_tensor.storage.clone(),
                        indices_layout: Layout::contiguous(vec![channels, out_h, out_w]),
                        channels, h, w, out_h, out_w,
                    }),
                    inputs: vec![in_gid],
                    outputs: vec![out_grad_id],
                })
            });

            Tensor {
                storage: out_storage,
                layout: Layout::contiguous(result_shape),
                state: AutogradState::Tracked(Arc::new(TensorMeta {
                    requires_grad: true,
                    grad_id: Some(out_grad_id),
                    creator: op_id,
                    is_leaf: false,
                    retains_grad: false,
                    total_grads: AtomicUsize::new(0),
                })),
            }
        } else {
            Tensor {
                storage: out_storage,
                layout: Layout::contiguous(result_shape),
                state: AutogradState::None,
            }
        }
    }

    /// Zero-copy tracked reshape.
    ///
    /// Like `reshape`, but preserves autograd tracking.  The backward pass
    /// simply reshapes the gradient back to the original shape.
    ///
    /// Use this when reshaping a tracked tensor in a differentiable graph
    /// (e.g., at the end of Conv2d forward).
    pub fn reshape_tracked(&self, new_shape: Vec<usize>) -> Tensor {
        let old_numel: usize = self.shape().iter().product();
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(old_numel, new_numel, "reshape_tracked: numel mismatch");

        // Ensure contiguous so new layout is valid over the same storage.
        let input_c = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous()
        };

        let out_storage = input_c.storage.clone();
        let out_layout = Layout::contiguous(new_shape.clone());

        if self.requires_grad() && !context::is_no_grad() {
            let out_grad_id = context::next_grad_id();
            let in_gid = self.grad_id().unwrap_or_else(context::next_grad_id);
            let input_version = VersionSnapshot::new(in_gid, &self.storage);

            if let Some(meta) = self.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::Reshape(ReshapeBackward {
                        input_version,
                        original_shape: self.shape().to_vec(),
                    }),
                    inputs: vec![in_gid],
                    outputs: vec![out_grad_id],
                })
            });

            Tensor {
                storage: out_storage,
                layout: out_layout,
                state: AutogradState::Tracked(Arc::new(TensorMeta {
                    requires_grad: true,
                    grad_id: Some(out_grad_id),
                    creator: op_id,
                    is_leaf: false,
                    retains_grad: false,
                    total_grads: AtomicUsize::new(0),
                })),
            }
        } else {
            Tensor {
                storage: out_storage,
                layout: out_layout,
                state: AutogradState::None,
            }
        }
    }

    /// Apply dropout: randomly zero elements with probability `p` and scale
    /// survivors by `1 / (1 - p)` (inverted dropout).
    ///
    /// Saves the mask for backward.  The backward is simply
    /// `out_grad * saved_mask` — reuses the existing `mul` dispatch.
    ///
    /// `step` is a monotonically increasing counter for PRNG seeding.
    pub fn dropout(&self, p: f32) -> Tensor {
        use std::sync::atomic::{AtomicU64, Ordering as AtOrd};
        static DROPOUT_STEP: AtomicU64 = AtomicU64::new(0);

        assert!(p >= 0.0 && p < 1.0, "dropout: p must be in [0, 1)");
        let numel = self.numel();
        let step = DROPOUT_STEP.fetch_add(1, AtOrd::Relaxed);
        let result_shape = self.shape().to_vec();

        // --- GPU path: fused stride-aware dropout — zero intermediate alloc ---
        #[cfg(feature = "gpu")]
        let (out_storage, mask_storage) = if self.storage.is_gpu() {
            let ctx = GpuContext::get().expect("GPU required");
            let ib = self.storage.gpu_buffer();
            let out_buf = ctx.pool.acquire(&ctx.device, self.dtype().gpu_buf_size(numel), STORAGE_USAGE);
            let mask_buf = ctx.pool.acquire(&ctx.device, self.dtype().gpu_buf_size(numel), STORAGE_USAGE);

            let shape = self.layout.shape();
            let strides = self.layout.strides();
            let offset = self.layout.offset();
            let ndim = self.ndim();

            let mut suffix = vec![1usize; ndim];
            for d in (0..ndim.saturating_sub(1)).rev() {
                suffix[d] = suffix[d + 1] * shape[d + 1];
            }

            gpu_compute::fused_dropout_forward(
                ctx, &ib, &out_buf, &mask_buf,
                numel as u32, step as u32, p,
                ndim as u32, offset as u32,
                shape, strides, &suffix,
            );
            drop(ib);
            (StorageHandle::new_gpu(out_buf, numel),
             StorageHandle::new_gpu(mask_buf, numel))
        } else {
            let input_c = self.contiguous();
            let in_guard = input_c.storage.data();
            let mut dst = CpuBackend::zeros(numel);
            let mut mask_data = CpuBackend::zeros(numel);
            CpuBackend::dropout(&in_guard, &mut dst, &mut mask_data, numel, p, step);
            drop(in_guard);
            (StorageHandle::new(dst), StorageHandle::new(mask_data))
        };

        // --- CPU-only path ---
        #[cfg(not(feature = "gpu"))]
        let (out_storage, mask_storage) = {
            let input_c = self.contiguous();
            let in_guard = input_c.storage.data();
            let mut dst = CpuBackend::zeros(numel);
            let mut mask_data = CpuBackend::zeros(numel);
            CpuBackend::dropout(&in_guard, &mut dst, &mut mask_data, numel, p, step);
            drop(in_guard);
            (StorageHandle::new(dst), StorageHandle::new(mask_data))
        };

        if self.requires_grad() && !context::is_no_grad() {
            let out_grad_id = context::next_grad_id();
            let in_gid = self.grad_id().unwrap_or_else(context::next_grad_id);
            let input_version = VersionSnapshot::new(in_gid, &self.storage);

            if let Some(meta) = self.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::Dropout(DropoutBackward {
                        input_version,
                        mask_storage: mask_storage.clone(),
                        mask_layout: Layout::contiguous(self.shape().to_vec()),
                    }),
                    inputs: vec![in_gid],
                    outputs: vec![out_grad_id],
                })
            });

            Tensor {
                storage: out_storage,
                layout: Layout::contiguous(result_shape),
                state: AutogradState::Tracked(Arc::new(TensorMeta {
                    requires_grad: true,
                    grad_id: Some(out_grad_id),
                    creator: op_id,
                    is_leaf: false,
                    retains_grad: false,
                    total_grads: AtomicUsize::new(0),
                })),
            }
        } else {
            Tensor {
                storage: out_storage,
                layout: Layout::contiguous(result_shape),
                state: AutogradState::None,
            }
        }
    }

    // === Advanced activations (sigmoid, tanh, gelu, leaky_relu) ===============

    /// Macro-like helper for unary activations that save a tensor for backward.
    /// `forward_cpu`: CPU kernel fn.  `save_output`: if true, save the output;
    /// if false, save the input.  `bw_variant`: the BackwardOp constructor.

    pub fn sigmoid(&self) -> Tensor {
        unary_activation(self, CpuBackend::sigmoid, true, 0.0,
            |_out_storage, in_version, saved_storage, shape| {
                BackwardOp::Sigmoid(SigmoidBackward {
                    output_storage: saved_storage,
                    output_layout: Layout::contiguous(shape),
                    input_version: in_version,
                })
            })
    }

    pub fn tanh_act(&self) -> Tensor {
        unary_activation(self, CpuBackend::tanh_act, true, 0.0,
            |_out_storage, in_version, saved_storage, shape| {
                BackwardOp::Tanh(TanhBackward {
                    output_storage: saved_storage,
                    output_layout: Layout::contiguous(shape),
                    input_version: in_version,
                })
            })
    }

    pub fn gelu(&self) -> Tensor {
        unary_activation(self, CpuBackend::gelu, false, 0.0,
            |_out_storage, in_version, saved_storage, shape| {
                BackwardOp::Gelu(GeluBackward {
                    input_storage: saved_storage,
                    input_layout: Layout::contiguous(shape),
                    input_version: in_version,
                })
            })
    }

    pub fn leaky_relu(&self, alpha: f32) -> Tensor {
        let input_c = self.contiguous();
        let in_guard = input_c.storage.data();
        let numel = self.numel();
        let mut dst = CpuBackend::zeros(numel);
        CpuBackend::leaky_relu(&in_guard, &mut dst, alpha);
        drop(in_guard);

        let out_storage = StorageHandle::new(dst);
        let result_shape = self.shape().to_vec();

        if self.requires_grad() && !context::is_no_grad() {
            let out_grad_id = context::next_grad_id();
            let in_gid = self.grad_id().unwrap_or_else(context::next_grad_id);
            let input_version = VersionSnapshot::new(in_gid, &self.storage);
            if let Some(meta) = self.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }
            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::LeakyRelu(LeakyReluBackward {
                        input_storage: self.storage.clone(),
                        input_layout: self.layout.clone(),
                        input_version,
                        alpha,
                    }),
                    inputs: vec![in_gid],
                    outputs: vec![out_grad_id],
                })
            });
            Tensor {
                storage: out_storage,
                layout: Layout::contiguous(result_shape),
                state: AutogradState::Tracked(Arc::new(TensorMeta {
                    requires_grad: true, grad_id: Some(out_grad_id), creator: op_id,
                    is_leaf: false, retains_grad: false, total_grads: AtomicUsize::new(0),
                })),
            }
        } else {
            Tensor { storage: out_storage, layout: Layout::contiguous(result_shape), state: AutogradState::None }
        }
    }

    // === Batched MatMul =======================================================

    /// Batched matrix multiplication: `[B, M, K] @ [B, K, N] → [B, M, N]`.
    pub fn bmm(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 3, "bmm: lhs must be 3-D");
        assert_eq!(rhs.ndim(), 3, "bmm: rhs must be 3-D");
        let batch = self.shape()[0];
        let m = self.shape()[1];
        let k = self.shape()[2];
        let n = rhs.shape()[2];
        assert_eq!(rhs.shape()[0], batch, "bmm: batch mismatch");
        assert_eq!(rhs.shape()[1], k, "bmm: inner dim mismatch");

        let lhs_c = self.contiguous();
        let rhs_c = rhs.contiguous();
        let out_numel = batch * m * n;
        let result_shape = vec![batch, m, n];

        #[cfg(feature = "gpu")]
        let out_storage = if either_gpu(&lhs_c, &rhs_c) {
            let ctx = GpuContext::get().expect("GPU required");
            let lb = lhs_c.storage.gpu_buffer();
            let rb = rhs_c.storage.gpu_buffer();
            let dst_buf = ctx.pool.acquire(&ctx.device, self.dtype().gpu_buf_size(out_numel), STORAGE_USAGE);
            {
                let mut enc = ctx.device.create_command_encoder(&Default::default());
                enc.clear_buffer(&dst_buf, 0, None);
                ctx.queue.submit(std::iter::once(enc.finish()));
            }
            gpu_compute::bmm_dispatch(ctx, &lb, &rb, &dst_buf, batch as u32, m as u32, k as u32, n as u32);
            drop(lb); drop(rb);
            StorageHandle::new_gpu(dst_buf, out_numel)
        } else {
            let lg = lhs_c.storage.data();
            let rg = rhs_c.storage.data();
            let mut dst = CpuBackend::zeros(out_numel);
            CpuBackend::bmm(&lg, &rg, &mut dst, batch, m, k, n);
            drop(rg); drop(lg);
            StorageHandle::new(dst)
        };

        #[cfg(not(feature = "gpu"))]
        let out_storage = {
            let lg = lhs_c.storage.data();
            let rg = rhs_c.storage.data();
            let mut dst = CpuBackend::zeros(out_numel);
            CpuBackend::bmm(&lg, &rg, &mut dst, batch, m, k, n);
            drop(rg); drop(lg);
            StorageHandle::new(dst)
        };

        if should_record(self, rhs) {
            let out_gid = context::next_grad_id();
            let lhs_gid = require_grad_id(self).unwrap_or_else(context::next_grad_id);
            let rhs_gid = require_grad_id(rhs).unwrap_or_else(context::next_grad_id);
            if let Some(m) = self.meta() { m.total_grads.fetch_add(1, Ordering::Relaxed); }
            if let Some(m) = rhs.meta() { m.total_grads.fetch_add(1, Ordering::Relaxed); }

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::Bmm(BmmBackward {
                        lhs_storage: self.storage.clone(), lhs_layout: self.layout.clone(),
                        lhs_version: VersionSnapshot::new(lhs_gid, &self.storage),
                        rhs_storage: rhs.storage.clone(), rhs_layout: rhs.layout.clone(),
                        rhs_version: VersionSnapshot::new(rhs_gid, &rhs.storage),
                        batch, m, k, n,
                    }),
                    inputs: vec![lhs_gid, rhs_gid],
                    outputs: vec![out_gid],
                })
            });
            Tensor {
                storage: out_storage, layout: Layout::contiguous(result_shape),
                state: AutogradState::Tracked(Arc::new(TensorMeta {
                    requires_grad: true, grad_id: Some(out_gid), creator: op_id,
                    is_leaf: false, retains_grad: false, total_grads: AtomicUsize::new(0),
                })),
            }
        } else {
            Tensor { storage: out_storage, layout: Layout::contiguous(result_shape), state: AutogradState::None }
        }
    }

    /// Tracked contiguous: if non-contiguous, materializes a copy while
    /// preserving the autograd chain via `ReshapeBackward`.
    /// If already contiguous, zero-copy (same as `reshape_tracked` with same shape).
    pub fn contiguous_tracked(&self) -> Tensor {
        if self.is_contiguous() {
            // Zero-copy: same storage, contiguous layout, tracked.
            let shape = self.shape().to_vec();
            return self.reshape_tracked(shape);
        }
        // Non-contiguous: copy data, track via ReshapeBackward.
        let c = self.contiguous(); // untracked copy
        if self.requires_grad() && !context::is_no_grad() {
            let out_gid = context::next_grad_id();
            let in_gid = self.grad_id().unwrap_or_else(context::next_grad_id);
            if let Some(m) = self.meta() { m.total_grads.fetch_add(1, Ordering::Relaxed); }
            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::Reshape(ReshapeBackward {
                        input_version: VersionSnapshot::new(in_gid, &self.storage),
                        original_shape: self.shape().to_vec(),
                    }),
                    inputs: vec![in_gid],
                    outputs: vec![out_gid],
                })
            });
            Tensor {
                storage: c.storage,
                layout: c.layout,
                state: AutogradState::Tracked(Arc::new(TensorMeta {
                    requires_grad: true, grad_id: Some(out_gid), creator: op_id,
                    is_leaf: false, retains_grad: false, total_grads: AtomicUsize::new(0),
                })),
            }
        } else {
            c
        }
    }

    /// Tracked transpose: swaps two dimensions while preserving autograd.
    ///
    /// Zero-copy view (clones storage, swaps shape/strides).
    /// Records `TransposeBackward` on the tape so gradients flow through.
    pub fn transpose_tracked(&self, dim0: usize, dim1: usize) -> Tensor {
        let out_storage = self.storage.clone();
        let out_layout = self.layout.transposed(dim0, dim1);
        let result_shape = out_layout.shape().to_vec();

        if self.requires_grad() && !context::is_no_grad() {
            let out_gid = context::next_grad_id();
            let in_gid = self.grad_id().unwrap_or_else(context::next_grad_id);
            if let Some(m) = self.meta() { m.total_grads.fetch_add(1, Ordering::Relaxed); }
            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::Transpose(TransposeBackward {
                        input_version: VersionSnapshot::new(in_gid, &self.storage),
                        dim0, dim1,
                    }),
                    inputs: vec![in_gid],
                    outputs: vec![out_gid],
                })
            });
            Tensor {
                storage: out_storage, layout: out_layout,
                state: AutogradState::Tracked(Arc::new(TensorMeta {
                    requires_grad: true, grad_id: Some(out_gid), creator: op_id,
                    is_leaf: false, retains_grad: false, total_grads: AtomicUsize::new(0),
                })),
            }
        } else {
            Tensor { storage: out_storage, layout: out_layout, state: AutogradState::None }
        }
    }

    /// Transpose the last two dimensions of a 3D+ tensor. Tracked.
    pub fn batched_transpose(&self) -> Tensor {
        assert!(self.ndim() >= 2, "batched_transpose: need >= 2 dims");
        let ndim = self.ndim();
        self.transpose_tracked(ndim - 2, ndim - 1)
    }

    // === Softmax ==============================================================

    /// Row-wise softmax over the last dimension with Log-Sum-Exp stability.
    ///
    /// Input: `[..., D]`.  Output: same shape, each row sums to 1.
    /// Saves the output for backward.
    pub fn softmax(&self) -> Tensor {
        let shape = self.shape().to_vec();
        let row_size = *shape.last().expect("softmax: empty shape");
        let num_rows = self.numel() / row_size;

        let ic = self.contiguous();

        #[cfg(feature = "gpu")]
        let (out_storage, out_storage_for_save) = if ic.storage.is_gpu() {
            let ctx = GpuContext::get().expect("GPU required");
            let ib = ic.storage.gpu_buffer();
            let ob = ctx.pool.acquire(&ctx.device, self.dtype().gpu_buf_size(self.numel()), STORAGE_USAGE);
            gpu_compute::softmax_forward_dispatch(ctx, &ib, &ob, num_rows as u32, row_size as u32);
            drop(ib);
            let sh = StorageHandle::new_gpu(ob, self.numel());
            (sh.clone(), sh)
        } else {
            let ig = ic.storage.data();
            let mut out = CpuBackend::zeros(self.numel());
            CpuBackend::softmax_forward(&ig, &mut out, num_rows, row_size);
            drop(ig);
            let sh = StorageHandle::new(out);
            (sh.clone(), sh)
        };

        #[cfg(not(feature = "gpu"))]
        let (out_storage, out_storage_for_save) = {
            let ig = ic.storage.data();
            let mut out = CpuBackend::zeros(self.numel());
            CpuBackend::softmax_forward(&ig, &mut out, num_rows, row_size);
            drop(ig);
            let sh = StorageHandle::new(out);
            (sh.clone(), sh)
        };

        if self.requires_grad() && !context::is_no_grad() {
            let out_gid = context::next_grad_id();
            let in_gid = self.grad_id().unwrap_or_else(context::next_grad_id);
            if let Some(m) = self.meta() { m.total_grads.fetch_add(1, Ordering::Relaxed); }

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::Softmax(SoftmaxBackward {
                        output_storage: out_storage_for_save,
                        output_layout: Layout::contiguous(shape.clone()),
                        input_version: VersionSnapshot::new(in_gid, &self.storage),
                        num_rows, row_size,
                    }),
                    inputs: vec![in_gid],
                    outputs: vec![out_gid],
                })
            });
            Tensor {
                storage: out_storage, layout: Layout::contiguous(shape),
                state: AutogradState::Tracked(Arc::new(TensorMeta {
                    requires_grad: true, grad_id: Some(out_gid), creator: op_id,
                    is_leaf: false, retains_grad: false, total_grads: AtomicUsize::new(0),
                })),
            }
        } else {
            Tensor { storage: out_storage, layout: Layout::contiguous(shape), state: AutogradState::None }
        }
    }

    // === LayerNorm ===========================================================

    /// Layer normalization over the last dimension.
    ///
    /// Input: `[..., D]`.  `weight` (γ) and `bias` (β) are `[D]`.
    /// Saves mean+invstd per instance for backward.
    /// Tape records 3 inputs: [input, weight, bias].
    pub fn layer_norm(&self, weight: &Tensor, bias: &Tensor, epsilon: f32) -> Tensor {
        let shape = self.shape().to_vec();
        let norm_size = *shape.last().expect("layer_norm: empty shape");
        let num_instances = self.numel() / norm_size;
        assert_eq!(weight.shape(), &[norm_size], "layer_norm: weight shape mismatch");
        assert_eq!(bias.shape(), &[norm_size], "layer_norm: bias shape mismatch");

        // GPU path
        #[cfg(feature = "gpu")]
        let (out_storage, save_storage) = if self.storage.is_gpu() {
            let ctx = GpuContext::get().expect("GPU required");
            let ic = self.contiguous();
            let wc = weight.contiguous();
            let bc = bias.contiguous();
            let ib = ic.storage.gpu_buffer();
            let wb = wc.storage.gpu_buffer();
            let bb = bc.storage.gpu_buffer();
            let out_buf = ctx.pool.acquire(&ctx.device, self.dtype().gpu_buf_size(self.numel()), STORAGE_USAGE);
            let save_buf = ctx.pool.acquire(&ctx.device, self.dtype().gpu_buf_size(num_instances * 2), STORAGE_USAGE);
            gpu_compute::layer_norm_forward(
                ctx, &ib, &wb, &bb, &out_buf, &save_buf,
                num_instances as u32, norm_size as u32, epsilon,
            );
            drop(ib); drop(wb); drop(bb);
            (StorageHandle::new_gpu(out_buf, self.numel()),
             StorageHandle::new_gpu(save_buf, num_instances * 2))
        } else {
            let ic = self.contiguous();
            let wc = weight.contiguous();
            let bc = bias.contiguous();
            let ig = ic.storage.data();
            let wg = wc.storage.data();
            let bg = bc.storage.data();
            let mut out = CpuBackend::zeros(self.numel());
            let mut save = CpuBackend::zeros(num_instances * 2);
            CpuBackend::layer_norm_forward(&ig, &wg, &bg, &mut out, &mut save, num_instances, norm_size, epsilon);
            drop(ig); drop(wg); drop(bg);
            (StorageHandle::new(out), StorageHandle::new(save))
        };

        #[cfg(not(feature = "gpu"))]
        let (out_storage, save_storage) = {
            let ic = self.contiguous();
            let wc = weight.contiguous();
            let bc = bias.contiguous();
            let ig = ic.storage.data();
            let wg = wc.storage.data();
            let bg = bc.storage.data();
            let mut out = CpuBackend::zeros(self.numel());
            let mut save = CpuBackend::zeros(num_instances * 2);
            CpuBackend::layer_norm_forward(&ig, &wg, &bg, &mut out, &mut save, num_instances, norm_size, epsilon);
            drop(ig); drop(wg); drop(bg);
            (StorageHandle::new(out), StorageHandle::new(save))
        };

        let any_tracked = (self.requires_grad() || weight.requires_grad() || bias.requires_grad())
            && !context::is_no_grad();

        if any_tracked {
            let out_gid = context::next_grad_id();
            let in_gid = require_grad_id(self).unwrap_or_else(context::next_grad_id);
            let w_gid = require_grad_id(weight).unwrap_or_else(context::next_grad_id);
            let b_gid = require_grad_id(bias).unwrap_or_else(context::next_grad_id);

            if let Some(m) = self.meta() { m.total_grads.fetch_add(1, Ordering::Relaxed); }
            if let Some(m) = weight.meta() { m.total_grads.fetch_add(1, Ordering::Relaxed); }
            if let Some(m) = bias.meta() { m.total_grads.fetch_add(1, Ordering::Relaxed); }

            let input_version = VersionSnapshot::new(in_gid, &self.storage);
            let weight_version = VersionSnapshot::new(w_gid, &weight.storage);

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::LayerNorm(LayerNormBackward {
                        input_storage: self.storage.clone(),
                        input_layout: self.layout.clone(),
                        input_version,
                        weight_storage: weight.storage.clone(),
                        weight_layout: Layout::contiguous(vec![norm_size]),
                        weight_version,
                        save_storage: save_storage.clone(),
                        save_layout: Layout::contiguous(vec![num_instances, 2]),
                        num_instances,
                        norm_size,
                    }),
                    inputs: vec![in_gid, w_gid, b_gid],
                    outputs: vec![out_gid],
                })
            });

            Tensor {
                storage: out_storage,
                layout: Layout::contiguous(shape),
                state: AutogradState::Tracked(Arc::new(TensorMeta {
                    requires_grad: true, grad_id: Some(out_gid), creator: op_id,
                    is_leaf: false, retains_grad: false, total_grads: AtomicUsize::new(0),
                })),
            }
        } else {
            Tensor { storage: out_storage, layout: Layout::contiguous(shape), state: AutogradState::None }
        }
    }

    // === BatchNorm2d ===========================================================

    /// Batch normalization over `[B, C, H, W]` tensors.
    ///
    /// During training: computes per-channel mean/var from the batch,
    /// updates `running_mean`/`running_var` with momentum.
    /// During eval: uses `running_mean`/`running_var`.
    ///
    /// `save` receives `[C, 2]`: mean + invstd per channel (for backward).
    /// Tape records 3 inputs: [input, weight, bias].
    pub fn batch_norm_2d(
        &self,
        weight: &Tensor,
        bias: &Tensor,
        running_mean: &mut Vec<f32>,
        running_var: &mut Vec<f32>,
        epsilon: f32,
        momentum: f32,
        is_training: bool,
    ) -> Tensor {
        assert_eq!(self.ndim(), 4, "batch_norm_2d: input must be 4-D [B, C, H, W]");
        let batch = self.shape()[0];
        let channels = self.shape()[1];
        let height = self.shape()[2];
        let width = self.shape()[3];
        assert_eq!(weight.shape(), &[channels], "batch_norm_2d: weight shape mismatch");
        assert_eq!(bias.shape(), &[channels], "batch_norm_2d: bias shape mismatch");

        let numel = batch * channels * height * width;
        let save_len = channels * 2;

        #[cfg(feature = "gpu")]
        let (out_storage, save_storage) = if self.storage.is_gpu() {
            let ctx = GpuContext::get().expect("GPU required");
            let ic = self.contiguous();
            let wc = weight.contiguous();
            let bc = bias.contiguous();
            let ib = ic.storage.gpu_buffer();
            let wb = wc.storage.gpu_buffer();
            let bb = bc.storage.gpu_buffer();

            // Upload running stats to GPU
            let rm_buf = ctx.pool.acquire(&ctx.device, crate::tensor::DType::F32.gpu_buf_size(channels), STORAGE_USAGE);
            let rv_buf = ctx.pool.acquire(&ctx.device, crate::tensor::DType::F32.gpu_buf_size(channels), STORAGE_USAGE);
            ctx.queue.write_buffer(&rm_buf, 0, bytemuck::cast_slice(running_mean));
            ctx.queue.write_buffer(&rv_buf, 0, bytemuck::cast_slice(running_var));

            let out_buf = ctx.pool.acquire(&ctx.device, self.dtype().gpu_buf_size(numel), STORAGE_USAGE);
            let save_buf = ctx.pool.acquire(&ctx.device, crate::tensor::DType::F32.gpu_buf_size(save_len), STORAGE_USAGE);

            gpu_compute::batch_norm_forward(
                ctx, &ib, &wb, &bb, &rm_buf, &rv_buf, &out_buf, &save_buf,
                batch as u32, channels as u32, height as u32, width as u32,
                epsilon, momentum, is_training,
            );

            // Download updated running stats back to CPU
            if is_training {
                let tmp_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("bn_rm_readback"),
                    size: crate::tensor::DType::F32.gpu_buf_size(channels),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let tmp_buf2 = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("bn_rv_readback"),
                    size: crate::tensor::DType::F32.gpu_buf_size(channels),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let mut enc = ctx.device.create_command_encoder(&Default::default());
                enc.copy_buffer_to_buffer(&rm_buf, 0, &tmp_buf, 0, crate::tensor::DType::F32.gpu_buf_size(channels));
                enc.copy_buffer_to_buffer(&rv_buf, 0, &tmp_buf2, 0, crate::tensor::DType::F32.gpu_buf_size(channels));
                ctx.queue.submit(std::iter::once(enc.finish()));

                let (tx, rx) = std::sync::mpsc::channel();
                let tx2 = tx.clone();
                tmp_buf.slice(..).map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
                tmp_buf2.slice(..).map_async(wgpu::MapMode::Read, move |r| { tx2.send(r).ok(); });
                ctx.device.poll(wgpu::Maintain::Wait);
                rx.recv().unwrap().unwrap();
                rx.recv().unwrap().unwrap();

                let rm_data = tmp_buf.slice(..).get_mapped_range();
                running_mean.copy_from_slice(bytemuck::cast_slice(&rm_data));
                drop(rm_data);
                let rv_data = tmp_buf2.slice(..).get_mapped_range();
                running_var.copy_from_slice(bytemuck::cast_slice(&rv_data));
                drop(rv_data);
            }

            drop(ib); drop(wb); drop(bb);
            ctx.pool.release(rm_buf);
            ctx.pool.release(rv_buf);

            (StorageHandle::new_gpu(out_buf, numel),
             StorageHandle::new_gpu(save_buf, save_len))
        } else {
            let ic = self.contiguous();
            let wc = weight.contiguous();
            let bc = bias.contiguous();
            let ig = ic.storage.data();
            let wg = wc.storage.data();
            let bg = bc.storage.data();
            let mut out = CpuBackend::zeros(numel);
            let mut save = CpuBackend::zeros(save_len);
            CpuBackend::batch_norm_forward(
                &ig, &wg, &bg, running_mean, running_var,
                &mut out, &mut save,
                batch, channels, height, width,
                epsilon, momentum, is_training,
            );
            drop(ig); drop(wg); drop(bg);
            (StorageHandle::new(out), StorageHandle::new(save))
        };

        #[cfg(not(feature = "gpu"))]
        let (out_storage, save_storage) = {
            let ic = self.contiguous();
            let wc = weight.contiguous();
            let bc = bias.contiguous();
            let ig = ic.storage.data();
            let wg = wc.storage.data();
            let bg = bc.storage.data();
            let mut out = CpuBackend::zeros(numel);
            let mut save = CpuBackend::zeros(save_len);
            CpuBackend::batch_norm_forward(
                &ig, &wg, &bg, running_mean, running_var,
                &mut out, &mut save,
                batch, channels, height, width,
                epsilon, momentum, is_training,
            );
            drop(ig); drop(wg); drop(bg);
            (StorageHandle::new(out), StorageHandle::new(save))
        };

        let result_shape = self.shape().to_vec();
        let any_tracked = (self.requires_grad() || weight.requires_grad() || bias.requires_grad())
            && !context::is_no_grad();

        if any_tracked {
            let out_gid = context::next_grad_id();
            let in_gid = require_grad_id(self).unwrap_or_else(context::next_grad_id);
            let w_gid = require_grad_id(weight).unwrap_or_else(context::next_grad_id);
            let b_gid = require_grad_id(bias).unwrap_or_else(context::next_grad_id);

            if let Some(m) = self.meta() { m.total_grads.fetch_add(1, Ordering::Relaxed); }
            if let Some(m) = weight.meta() { m.total_grads.fetch_add(1, Ordering::Relaxed); }
            if let Some(m) = bias.meta() { m.total_grads.fetch_add(1, Ordering::Relaxed); }

            let input_version = VersionSnapshot::new(in_gid, &self.storage);
            let weight_version = VersionSnapshot::new(w_gid, &weight.storage);

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::BatchNorm2d(BatchNorm2dBackward {
                        input_storage: self.storage.clone(),
                        input_layout: self.layout.clone(),
                        input_version,
                        weight_storage: weight.storage.clone(),
                        weight_layout: Layout::contiguous(vec![channels]),
                        weight_version,
                        save_storage: save_storage.clone(),
                        save_layout: Layout::contiguous(vec![channels, 2]),
                        batch, channels, height, width,
                    }),
                    inputs: vec![in_gid, w_gid, b_gid],
                    outputs: vec![out_gid],
                })
            });

            Tensor {
                storage: out_storage,
                layout: Layout::contiguous(result_shape),
                state: AutogradState::Tracked(Arc::new(TensorMeta {
                    requires_grad: true, grad_id: Some(out_gid), creator: op_id,
                    is_leaf: false, retains_grad: false, total_grads: AtomicUsize::new(0),
                })),
            }
        } else {
            Tensor { storage: out_storage, layout: Layout::contiguous(result_shape), state: AutogradState::None }
        }
    }

    // === AdaptiveAvgPool2d =====================================================

    /// Adaptive average pooling: `[B, C, H_in, W_in] → [B, C, H_out, W_out]`.
    ///
    /// Dynamically computes bin boundaries using floor-start / ceil-end.
    pub fn adaptive_avg_pool2d(&self, h_out: usize, w_out: usize) -> Tensor {
        assert_eq!(self.ndim(), 4, "adaptive_avg_pool2d: input must be 4-D [B, C, H, W]");
        let batch = self.shape()[0];
        let channels = self.shape()[1];
        let h_in = self.shape()[2];
        let w_in = self.shape()[3];
        let out_numel = batch * channels * h_out * w_out;

        let ic = self.contiguous();

        #[cfg(feature = "gpu")]
        let out_storage = if ic.storage.is_gpu() {
            let ctx = GpuContext::get().expect("GPU required");
            let ib = ic.storage.gpu_buffer();
            let dst = ctx.pool.acquire(&ctx.device, self.dtype().gpu_buf_size(out_numel), STORAGE_USAGE);
            gpu_compute::adaptive_avg_pool2d_forward(
                ctx, &ib, &dst,
                batch as u32, channels as u32, h_in as u32, w_in as u32, h_out as u32, w_out as u32,
            );
            drop(ib);
            StorageHandle::new_gpu(dst, out_numel)
        } else {
            let ig = ic.storage.data();
            let mut out = CpuBackend::zeros(out_numel);
            CpuBackend::adaptive_avg_pool2d(&ig, &mut out, batch, channels, h_in, w_in, h_out, w_out);
            drop(ig);
            StorageHandle::new(out)
        };

        #[cfg(not(feature = "gpu"))]
        let out_storage = {
            let ig = ic.storage.data();
            let mut out = CpuBackend::zeros(out_numel);
            CpuBackend::adaptive_avg_pool2d(&ig, &mut out, batch, channels, h_in, w_in, h_out, w_out);
            drop(ig);
            StorageHandle::new(out)
        };

        let result_shape = vec![batch, channels, h_out, w_out];

        if self.requires_grad() && !context::is_no_grad() {
            let out_gid = context::next_grad_id();
            let in_gid = self.grad_id().unwrap_or_else(context::next_grad_id);
            if let Some(m) = self.meta() { m.total_grads.fetch_add(1, Ordering::Relaxed); }

            let input_version = VersionSnapshot::new(in_gid, &self.storage);

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::AdaptiveAvgPool2d(AdaptiveAvgPool2dBackward {
                        input_version,
                        batch, channels, h_in, w_in, h_out, w_out,
                    }),
                    inputs: vec![in_gid],
                    outputs: vec![out_gid],
                })
            });

            Tensor {
                storage: out_storage,
                layout: Layout::contiguous(result_shape),
                state: AutogradState::Tracked(Arc::new(TensorMeta {
                    requires_grad: true, grad_id: Some(out_gid), creator: op_id,
                    is_leaf: false, retains_grad: false, total_grads: AtomicUsize::new(0),
                })),
            }
        } else {
            Tensor { storage: out_storage, layout: Layout::contiguous(result_shape), state: AutogradState::None }
        }
    }

    // === Embedding ============================================================

    /// Embedding lookup: indices `[..., S]` → output `[..., S, embed_dim]`.
    ///
    /// `weight` shape: `[vocab_size, embed_dim]`.
    /// `indices` stored as f32 (cast to u32 internally).
    pub fn embedding_forward(&self, weight: &Tensor) -> Tensor {
        let total_lookups = self.numel();
        assert_eq!(weight.ndim(), 2, "embedding: weight must be 2-D [vocab, dim]");
        let vocab_size = weight.shape()[0];
        let embed_dim = weight.shape()[1];

        let ic = self.contiguous();
        let wc = weight.contiguous();
        let ig = ic.storage.data();
        let wg = wc.storage.data();
        let mut out = CpuBackend::zeros(total_lookups * embed_dim);
        CpuBackend::embedding_forward(&ig, &wg, &mut out, total_lookups, embed_dim);
        drop(ig); drop(wg);

        let mut out_shape = self.shape().to_vec();
        out_shape.push(embed_dim);
        let out_storage = StorageHandle::new(out);

        if weight.requires_grad() && !context::is_no_grad() {
            let out_gid = context::next_grad_id();
            let w_gid = require_grad_id(weight).unwrap_or_else(context::next_grad_id);

            if let Some(m) = weight.meta() { m.total_grads.fetch_add(1, Ordering::Relaxed); }

            let input_version = VersionSnapshot::new(
                context::next_grad_id(), &self.storage,
            );

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::Embedding(EmbeddingBackward {
                        input_version,
                        indices_storage: self.storage.clone(),
                        indices_layout: self.layout.clone(),
                        vocab_size,
                        embed_dim,
                        total_lookups,
                    }),
                    inputs: vec![w_gid],
                    outputs: vec![out_gid],
                })
            });

            Tensor {
                storage: out_storage,
                layout: Layout::contiguous(out_shape),
                state: AutogradState::Tracked(Arc::new(TensorMeta {
                    requires_grad: true, grad_id: Some(out_gid), creator: op_id,
                    is_leaf: false, retains_grad: false, total_grads: AtomicUsize::new(0),
                })),
            }
        } else {
            Tensor { storage: out_storage, layout: Layout::contiguous(out_shape), state: AutogradState::None }
        }
    }

    // === Broadcast binary ops ================================================

    pub fn broadcast_add(&self, rhs: &Tensor) -> Tensor {
        broadcast_binary_op(self, rhs, |a, b| a + b,
            |lhs_v, rhs_v, lhs_bi, rhs_bi, out_shape| {
                BackwardOp::BroadcastAdd(BroadcastAddBackward {
                    lhs_version: lhs_v, rhs_version: rhs_v,
                    lhs_broadcast: lhs_bi, rhs_broadcast: rhs_bi,
                    output_shape: out_shape,
                })
            })
    }

    pub fn broadcast_sub(&self, rhs: &Tensor) -> Tensor {
        broadcast_binary_op(self, rhs, |a, b| a - b,
            |lhs_v, rhs_v, lhs_bi, rhs_bi, out_shape| {
                BackwardOp::BroadcastSub(BroadcastSubBackward {
                    lhs_version: lhs_v, rhs_version: rhs_v,
                    lhs_broadcast: lhs_bi, rhs_broadcast: rhs_bi,
                    output_shape: out_shape,
                })
            })
    }

    pub fn broadcast_mul(&self, rhs: &Tensor) -> Tensor {
        let lhs_st = self.storage.clone();
        let lhs_ly = self.layout.clone();
        let rhs_st = rhs.storage.clone();
        let rhs_ly = rhs.layout.clone();
        broadcast_binary_op(self, rhs, |a, b| a * b,
            |lhs_v, rhs_v, lhs_bi, rhs_bi, out_shape| {
                BackwardOp::BroadcastMul(BroadcastMulBackward {
                    lhs_storage: lhs_st,
                    lhs_layout: lhs_ly,
                    lhs_version: lhs_v,
                    rhs_storage: rhs_st,
                    rhs_layout: rhs_ly,
                    rhs_version: rhs_v,
                    lhs_broadcast: lhs_bi, rhs_broadcast: rhs_bi,
                    output_shape: out_shape,
                })
            })
    }

    // === DType casting =========================================================

    /// Cast tensor to a different element type.
    ///
    /// `F32 → F16`: dispatches a GPU cast kernel. Requires GPU support for `shader-f16`.
    /// `F16 → F32`: dispatches a GPU cast kernel.
    /// Same dtype: returns `self.clone()` (zero-copy).
    ///
    /// The backward pass is a cast in the reverse direction.
    pub fn to_dtype(&self, target: crate::tensor::DType) -> Tensor {
        if self.dtype() == target {
            return self.clone();
        }

        #[allow(unused_variables)]
        let numel = self.numel();
        #[allow(unused_variables)]
        let result_shape = self.shape().to_vec();

        #[cfg(feature = "gpu")]
        let out_storage = {
            use crate::tensor::DType;
            let ctx = GpuContext::get().expect("GPU required for dtype cast");
            assert!(ctx.supports_f16, "GPU does not support shader-f16; cannot cast to F16");

            let ic = self.contiguous();
            let ib = ic.storage.gpu_buffer();

            match (self.dtype(), target) {
                (DType::F32, DType::F16) => {
                    // F32 → F16: output buffer is numel * 2 bytes (aligned to 4).
                    let byte_size = ((numel * 2 + 3) & !3) as u64;
                    let dst = ctx.pool.acquire(&ctx.device, byte_size, STORAGE_USAGE);
                    gpu_compute::cast_f32_to_f16_dispatch(ctx, &ib, &dst, numel as u32);
                    drop(ib);
                    StorageHandle::new_gpu_f16(dst, numel)
                }
                (DType::F16, DType::F32) => {
                    let dst = ctx.pool.acquire(&ctx.device, self.dtype().gpu_buf_size(numel), STORAGE_USAGE);
                    gpu_compute::cast_f16_to_f32_dispatch(ctx, &ib, &dst, numel as u32);
                    drop(ib);
                    StorageHandle::new_gpu(dst, numel)
                }
                _ => unreachable!("same dtype handled above"),
            }
        };

        #[cfg(not(feature = "gpu"))]
        #[allow(unused)]
        let out_storage = {
            panic!("dtype casting requires GPU feature");
            #[allow(unreachable_code)]
            StorageHandle::new(vec![])
        };

        if self.requires_grad() && !context::is_no_grad() {
            let out_gid = context::next_grad_id();
            let in_gid = self.grad_id().unwrap_or_else(context::next_grad_id);
            if let Some(m) = self.meta() { m.total_grads.fetch_add(1, Ordering::Relaxed); }

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::Cast(CastBackward {
                        input_version: VersionSnapshot::new(in_gid, &self.storage),
                        source_dtype: self.dtype(),
                    }),
                    inputs: vec![in_gid],
                    outputs: vec![out_gid],
                })
            });

            Tensor {
                storage: out_storage,
                layout: Layout::contiguous(result_shape),
                state: AutogradState::Tracked(Arc::new(TensorMeta {
                    requires_grad: true, grad_id: Some(out_gid), creator: op_id,
                    is_leaf: false, retains_grad: false, total_grads: AtomicUsize::new(0),
                })),
            }
        } else {
            Tensor { storage: out_storage, layout: Layout::contiguous(result_shape), state: AutogradState::None }
        }
    }

    /// Flatten all dimensions after the first into a single dimension.
    ///
    /// Input shape: `[batch, d1, d2, ...]`.
    /// Output shape: `[batch, d1*d2*...]`.
    ///
    /// **Zero-copy** — clones the `StorageHandle` and computes a flat `Layout`.
    /// The backward pass is also zero-copy (reshape the gradient back).
    pub fn flatten(&self) -> Tensor {
        assert!(self.ndim() >= 2, "flatten: need at least 2 dims");
        let batch = self.shape()[0];
        let rest: usize = self.shape()[1..].iter().product();
        let result_shape = vec![batch, rest];

        // Zero-copy: clone storage, compute flat layout.
        let out_storage = self.storage.clone();
        let out_layout = Layout::contiguous(result_shape.clone());

        if self.requires_grad() && !context::is_no_grad() {
            let out_grad_id = context::next_grad_id();
            let in_gid = self.grad_id().unwrap_or_else(context::next_grad_id);
            let input_version = VersionSnapshot::new(in_gid, &self.storage);

            if let Some(meta) = self.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::Flatten(FlattenBackward {
                        input_version,
                        original_shape: self.shape().to_vec(),
                    }),
                    inputs: vec![in_gid],
                    outputs: vec![out_grad_id],
                })
            });

            Tensor {
                storage: out_storage,
                layout: out_layout,
                state: AutogradState::Tracked(Arc::new(TensorMeta {
                    requires_grad: true,
                    grad_id: Some(out_grad_id),
                    creator: op_id,
                    is_leaf: false,
                    retains_grad: false,
                    total_grads: AtomicUsize::new(0),
                })),
            }
        } else {
            Tensor {
                storage: out_storage,
                layout: out_layout,
                state: AutogradState::None,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Free-standing tracked ops
// ---------------------------------------------------------------------------

/// Stack tensors along axis 0: `[shape] * count → [count, ...shape]`.
///
/// All tensors must have the same shape.  This is a tracked op —
/// `StackBackward` splits the gradient back into individual tensors.
pub fn stack(tensors: &[Tensor]) -> Tensor {
    assert!(!tensors.is_empty(), "stack: empty input");
    let each_shape = tensors[0].shape().to_vec();
    let each_numel: usize = each_shape.iter().product();
    for t in tensors {
        assert_eq!(t.shape(), &each_shape[..], "stack: shape mismatch");
    }

    let count = tensors.len();
    let mut data = Vec::with_capacity(count * each_numel);
    for t in tensors {
        let tc = t.contiguous();
        let g = tc.storage.data();
        data.extend_from_slice(&g);
    }

    let mut result_shape = vec![count];
    result_shape.extend_from_slice(&each_shape);

    let out_storage = StorageHandle::new(data);

    let any_tracked = tensors.iter().any(|t| t.requires_grad())
        && !context::is_no_grad();

    if any_tracked {
        let out_grad_id = context::next_grad_id();

        let mut input_gids = Vec::with_capacity(count);
        let mut versions = Vec::with_capacity(count);
        for t in tensors {
            let gid = require_grad_id(t).unwrap_or_else(context::next_grad_id);
            versions.push(VersionSnapshot::new(gid, &t.storage));
            if let Some(meta) = t.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }
            input_gids.push(gid);
        }

        let op_id = context::with_tape(|tape| {
            tape.push(TapeEntry {
                op: BackwardOp::Stack(StackBackward {
                    count,
                    each_shape: each_shape.clone(),
                    versions,
                }),
                inputs: input_gids,
                outputs: vec![out_grad_id],
            })
        });

        Tensor {
            storage: out_storage,
            layout: Layout::contiguous(result_shape),
            state: AutogradState::Tracked(Arc::new(TensorMeta {
                requires_grad: true,
                grad_id: Some(out_grad_id),
                creator: op_id,
                is_leaf: false,
                retains_grad: false,
                total_grads: AtomicUsize::new(0),
            })),
        }
    } else {
        Tensor {
            storage: out_storage,
            layout: Layout::contiguous(result_shape),
            state: AutogradState::None,
        }
    }
}
