//! View operations (zero-copy) and materialising operations (allocating)
//! on [`Tensor`].
//!
//! When autograd is active (at least one input has `requires_grad` and
//! `no_grad` is not in effect), materialising ops record a [`TapeEntry`]
//! on the thread-local tape and return a tracked output tensor.
//! View operations always return `AutogradState::None` for now — view
//! autograd will be added in a later milestone.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::autograd::context;
use crate::autograd::{
    AddBackward, AddBiasBackward, BackwardOp, MatmulBackward, MseLossBackward, MulBackward,
    ReluBackward, SubBackward, TapeEntry, VersionSnapshot,
};
use crate::backend::{Backend, CpuBackend};
use crate::tensor::{AutogradState, Layout, StorageHandle, Tensor, TensorMeta};

// ---------------------------------------------------------------------------
// View operations — zero-copy, Arc refcount bump only
// ---------------------------------------------------------------------------

impl Tensor {
    /// Return a view with two axes swapped.  **Zero-copy.**
    ///
    /// # Panics
    ///
    /// Panics if `dim0` or `dim1` is out of range for this tensor's rank.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        Tensor {
            storage: self.storage.clone(),
            layout: self.layout.transposed(dim0, dim1),
            state: AutogradState::None,
        }
    }

    /// Return a view with a different shape.  **Zero-copy when contiguous.**
    ///
    /// # Panics
    ///
    /// Panics if the product of `shape` does not match [`Tensor::numel`].
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
    /// If already contiguous, return a new handle to the same storage with
    /// `AutogradState::None`.  Otherwise, allocate a fresh dense buffer and
    /// copy every element using the stride formula.
    ///
    /// Complexity: `O(numel x ndim)` for the strided path.
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

        // Acquire read lock on source data for the duration of the copy.
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
// Internal helpers for autograd recording
// ---------------------------------------------------------------------------

fn should_record(lhs: &Tensor, rhs: &Tensor) -> bool {
    (lhs.requires_grad() || rhs.requires_grad()) && !context::is_no_grad()
}

fn require_grad_id(t: &Tensor) -> Option<crate::tensor::GradId> {
    t.grad_id()
}

// ---------------------------------------------------------------------------
// Materialising operations — allocate new storage via CpuBackend
// ---------------------------------------------------------------------------

impl Tensor {
    /// Element-wise addition.  Returns a new contiguous tensor.
    ///
    /// # Panics
    ///
    /// Panics if shapes differ.
    pub fn add(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "add: shape mismatch {:?} vs {:?}",
            self.shape(),
            rhs.shape(),
        );

        let lhs_c = self.contiguous();
        let rhs_c = rhs.contiguous();

        let lhs_guard = lhs_c.storage.data();
        let rhs_guard = rhs_c.storage.data();

        let mut dst = CpuBackend::zeros(lhs_c.numel());
        CpuBackend::add(&lhs_guard, &rhs_guard, &mut dst);

        let result_shape = lhs_c.shape().to_vec();

        drop(rhs_guard);
        drop(lhs_guard);

        if should_record(self, rhs) {
            let out_grad_id = context::next_grad_id();

            let lhs_gid = require_grad_id(self)
                .unwrap_or_else(context::next_grad_id);
            let rhs_gid = require_grad_id(rhs)
                .unwrap_or_else(context::next_grad_id);

            let lhs_version = VersionSnapshot::new(lhs_gid, &self.storage);
            let rhs_version = VersionSnapshot::new(rhs_gid, &rhs.storage);

            if let Some(meta) = self.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }
            if let Some(meta) = rhs.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::Add(AddBackward {
                        lhs_version,
                        rhs_version,
                    }),
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
                storage: StorageHandle::new(dst),
                layout: Layout::contiguous(result_shape),
                state: AutogradState::Tracked(out_meta),
            }
        } else {
            Tensor::new(dst, result_shape)
        }
    }

    /// Element-wise multiplication.  Returns a new contiguous tensor.
    ///
    /// # Panics
    ///
    /// Panics if shapes differ.
    pub fn mul(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "mul: shape mismatch {:?} vs {:?}",
            self.shape(),
            rhs.shape(),
        );

        let lhs_c = self.contiguous();
        let rhs_c = rhs.contiguous();

        let lhs_guard = lhs_c.storage.data();
        let rhs_guard = rhs_c.storage.data();

        let mut dst = CpuBackend::zeros(lhs_c.numel());
        CpuBackend::mul(&lhs_guard, &rhs_guard, &mut dst);

        let result_shape = lhs_c.shape().to_vec();

        drop(rhs_guard);
        drop(lhs_guard);

        if should_record(self, rhs) {
            let out_grad_id = context::next_grad_id();

            let lhs_gid = require_grad_id(self)
                .unwrap_or_else(context::next_grad_id);
            let rhs_gid = require_grad_id(rhs)
                .unwrap_or_else(context::next_grad_id);

            let lhs_version = VersionSnapshot::new(lhs_gid, &self.storage);
            let rhs_version = VersionSnapshot::new(rhs_gid, &rhs.storage);

            let lhs_storage = self.storage.clone();
            let lhs_layout = self.layout.clone();
            let rhs_storage = rhs.storage.clone();
            let rhs_layout = rhs.layout.clone();

            if let Some(meta) = self.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }
            if let Some(meta) = rhs.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::Mul(MulBackward {
                        lhs_storage,
                        lhs_layout,
                        lhs_version,
                        rhs_storage,
                        rhs_layout,
                        rhs_version,
                    }),
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
                storage: StorageHandle::new(dst),
                layout: Layout::contiguous(result_shape),
                state: AutogradState::Tracked(out_meta),
            }
        } else {
            Tensor::new(dst, result_shape)
        }
    }

    /// Matrix multiplication (`self @ rhs`).  Returns a new contiguous tensor.
    ///
    /// # Panics
    ///
    /// - If either operand is not 2-D.
    /// - If the inner dimensions do not match.
    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "matmul: lhs must be 2-D, got {}-D", self.ndim());
        assert_eq!(rhs.ndim(), 2, "matmul: rhs must be 2-D, got {}-D", rhs.ndim());

        let m = self.shape()[0];
        let k = self.shape()[1];
        let n = rhs.shape()[1];

        assert_eq!(
            k,
            rhs.shape()[0],
            "matmul: inner dimension mismatch: ({} x {}) @ ({} x {})",
            m, k, rhs.shape()[0], n,
        );

        let lhs_c = self.contiguous();
        let rhs_c = rhs.contiguous();

        let lhs_guard = lhs_c.storage.data();
        let rhs_guard = rhs_c.storage.data();

        let mut dst = CpuBackend::zeros(m * n);
        CpuBackend::matmul(&lhs_guard, &rhs_guard, &mut dst, m, k, n);

        let result_shape = vec![m, n];

        drop(rhs_guard);
        drop(lhs_guard);

        if should_record(self, rhs) {
            let out_grad_id = context::next_grad_id();

            let lhs_gid = require_grad_id(self)
                .unwrap_or_else(context::next_grad_id);
            let rhs_gid = require_grad_id(rhs)
                .unwrap_or_else(context::next_grad_id);

            let lhs_version = VersionSnapshot::new(lhs_gid, &self.storage);
            let rhs_version = VersionSnapshot::new(rhs_gid, &rhs.storage);

            let lhs_storage = self.storage.clone();
            let lhs_layout = self.layout.clone();
            let rhs_storage = rhs.storage.clone();
            let rhs_layout = rhs.layout.clone();

            if let Some(meta) = self.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }
            if let Some(meta) = rhs.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::Matmul(MatmulBackward {
                        lhs_storage,
                        lhs_layout,
                        lhs_version,
                        rhs_storage,
                        rhs_layout,
                        rhs_version,
                        m,
                        k,
                        n,
                    }),
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
                storage: StorageHandle::new(dst),
                layout: Layout::contiguous(result_shape),
                state: AutogradState::Tracked(out_meta),
            }
        } else {
            Tensor::new(dst, result_shape)
        }
    }

    /// Element-wise subtraction.  Returns a new contiguous tensor.
    pub fn sub(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "sub: shape mismatch {:?} vs {:?}",
            self.shape(),
            rhs.shape(),
        );

        let lhs_c = self.contiguous();
        let rhs_c = rhs.contiguous();

        let lhs_guard = lhs_c.storage.data();
        let rhs_guard = rhs_c.storage.data();

        let mut dst = CpuBackend::zeros(lhs_c.numel());
        CpuBackend::sub(&lhs_guard, &rhs_guard, &mut dst);

        let result_shape = lhs_c.shape().to_vec();
        drop(rhs_guard);
        drop(lhs_guard);

        if should_record(self, rhs) {
            let out_grad_id = context::next_grad_id();
            let lhs_gid = require_grad_id(self).unwrap_or_else(context::next_grad_id);
            let rhs_gid = require_grad_id(rhs).unwrap_or_else(context::next_grad_id);
            let lhs_version = VersionSnapshot::new(lhs_gid, &self.storage);
            let rhs_version = VersionSnapshot::new(rhs_gid, &rhs.storage);

            if let Some(meta) = self.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }
            if let Some(meta) = rhs.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::Sub(SubBackward { lhs_version, rhs_version }),
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
                storage: StorageHandle::new(dst),
                layout: Layout::contiguous(result_shape),
                state: AutogradState::Tracked(out_meta),
            }
        } else {
            Tensor::new(dst, result_shape)
        }
    }

    /// Element-wise ReLU: `max(0, x)`.  Returns a new contiguous tensor.
    pub fn relu(&self) -> Tensor {
        let input_c = self.contiguous();
        let in_guard = input_c.storage.data();

        let mut dst = CpuBackend::zeros(input_c.numel());
        CpuBackend::relu(&in_guard, &mut dst);

        let result_shape = input_c.shape().to_vec();
        drop(in_guard);

        if self.requires_grad() && !context::is_no_grad() {
            let out_grad_id = context::next_grad_id();
            let in_gid = self.grad_id().unwrap_or_else(context::next_grad_id);
            let input_version = VersionSnapshot::new(in_gid, &self.storage);

            if let Some(meta) = self.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::Relu(ReluBackward {
                        input_storage: self.storage.clone(),
                        input_layout: self.layout.clone(),
                        input_version,
                    }),
                    inputs: vec![in_gid],
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
                storage: StorageHandle::new(dst),
                layout: Layout::contiguous(result_shape),
                state: AutogradState::Tracked(out_meta),
            }
        } else {
            Tensor::new(dst, result_shape)
        }
    }

    /// Fused mean squared error loss: `sum((self - target)^2) / N`.
    ///
    /// Returns a scalar tensor (shape `[1]`).  Only `self` (the prediction)
    /// receives a gradient; `target` is treated as a constant.
    pub fn mse_loss(&self, target: &Tensor) -> Tensor {
        assert_eq!(
            self.shape(),
            target.shape(),
            "mse_loss: shape mismatch {:?} vs {:?}",
            self.shape(),
            target.shape(),
        );

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

        let dst = vec![loss_val];
        let result_shape = vec![1];

        if self.requires_grad() && !context::is_no_grad() {
            let out_grad_id = context::next_grad_id();
            let pred_gid = self.grad_id().unwrap_or_else(context::next_grad_id);
            let pred_version = VersionSnapshot::new(pred_gid, &self.storage);
            let target_version = VersionSnapshot::new(
                context::next_grad_id(),
                &target.storage,
            );

            if let Some(meta) = self.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }

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

            let out_meta = Arc::new(TensorMeta {
                requires_grad: true,
                grad_id: Some(out_grad_id),
                creator: op_id,
                is_leaf: false,
                retains_grad: false,
                total_grads: AtomicUsize::new(0),
            });

            Tensor {
                storage: StorageHandle::new(dst),
                layout: Layout::contiguous(result_shape),
                state: AutogradState::Tracked(out_meta),
            }
        } else {
            Tensor::new(dst, result_shape)
        }
    }

    /// Add a 1-D bias `[n]` to each row of a 2-D matrix `[m, n]`.
    ///
    /// Returns a new `[m, n]` tensor.  Used by `Linear::forward`.
    pub fn add_bias(&self, bias: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "add_bias: input must be 2-D");
        assert_eq!(bias.ndim(), 1, "add_bias: bias must be 1-D");
        let m = self.shape()[0];
        let n = self.shape()[1];
        assert_eq!(
            bias.shape()[0],
            n,
            "add_bias: bias length {} != input columns {}",
            bias.shape()[0],
            n,
        );

        let mat_c = self.contiguous();
        let bias_c = bias.contiguous();
        let mat_guard = mat_c.storage.data();
        let bias_guard = bias_c.storage.data();

        let mut dst = CpuBackend::zeros(m * n);
        CpuBackend::add_bias(&mat_guard, &bias_guard, &mut dst, m, n);

        let result_shape = vec![m, n];
        drop(bias_guard);
        drop(mat_guard);

        if should_record(self, bias) {
            let out_grad_id = context::next_grad_id();
            let mat_gid = require_grad_id(self).unwrap_or_else(context::next_grad_id);
            let bias_gid = require_grad_id(bias).unwrap_or_else(context::next_grad_id);
            let input_version = VersionSnapshot::new(mat_gid, &self.storage);
            let bias_version = VersionSnapshot::new(bias_gid, &bias.storage);

            if let Some(meta) = self.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }
            if let Some(meta) = bias.meta() {
                meta.total_grads.fetch_add(1, Ordering::Relaxed);
            }

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::AddBias(AddBiasBackward {
                        input_version,
                        bias_version,
                        m,
                        n,
                    }),
                    inputs: vec![mat_gid, bias_gid],
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
                storage: StorageHandle::new(dst),
                layout: Layout::contiguous(result_shape),
                state: AutogradState::Tracked(out_meta),
            }
        } else {
            Tensor::new(dst, result_shape)
        }
    }
}
