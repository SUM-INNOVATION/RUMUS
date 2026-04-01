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
    AddBackward, AddBiasBackward, AddChannelBiasBackward, BackwardOp, Im2ColBackward,
    MatmulBackward, MseLossBackward, MulBackward, ReluBackward, StackBackward, SubBackward,
    TapeEntry, VersionSnapshot,
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
