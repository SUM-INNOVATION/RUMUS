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
    AddBackward, AddBiasBackward, BackwardOp, MatmulBackward, MseLossBackward, MulBackward,
    ReluBackward, SubBackward, TapeEntry, VersionSnapshot,
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
        let src_guard = self.storage.data();

        let mut suffix = vec![1usize; ndim];
        for d in (0..ndim.saturating_sub(1)).rev() {
            suffix[d] = suffix[d + 1] * shape[d + 1];
        }

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
    let dst = ctx.pool.acquire(&ctx.device, (numel * 4) as u64, STORAGE_USAGE);
    dispatch(ctx, &lb, &rb, &dst, numel as u32);
    drop(lb);
    drop(rb);
    StorageHandle::new_gpu(dst, numel)
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
            let dst = ctx.pool.acquire(&ctx.device, (m * n * 4) as u64, STORAGE_USAGE);
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
            let dst = ctx.pool.acquire(&ctx.device, (numel * 4) as u64, STORAGE_USAGE);
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
            let dst = ctx.pool.acquire(&ctx.device, (m * n * 4) as u64, STORAGE_USAGE);
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
}
