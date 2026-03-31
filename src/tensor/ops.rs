//! View operations (zero-copy) and materialising operations (allocating)
//! on [`Tensor`].
//!
//! All materialising ops dispatch to the [`CpuBackend`] and return tensors
//! with [`AutogradState::None`].  Autograd tracking will be layered on in
//! Milestone 2.

use crate::backend::{Backend, CpuBackend};
use crate::tensor::{AutogradState, Tensor};

// ---------------------------------------------------------------------------
// View operations — zero-copy, Arc refcount bump only
// ---------------------------------------------------------------------------

impl Tensor {
    /// Return a view with two axes swapped.  **Zero-copy.**
    ///
    /// Clones the [`StorageHandle`] (incrementing the `Arc` refcount) and
    /// produces a new [`Layout`] with the two dimensions' shape and strides
    /// swapped.  No data is moved or allocated.
    ///
    /// The returned tensor is almost always non-contiguous.  Call
    /// [`.contiguous()`](Tensor::contiguous) before passing it to a
    /// materialising op if needed (the ops do this automatically).
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
    /// If the tensor is already contiguous the operation is free — a new
    /// [`Layout`] is computed over the same storage.  If the tensor is
    /// strided (e.g. after a transpose), a contiguous copy is materialised
    /// first, then reshaped.
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
            // Layout is non-contiguous — materialise a dense copy, then reshape.
            // The recursive call is guaranteed to hit the `Some` branch because
            // `contiguous()` always returns a contiguous tensor.
            None => self.contiguous().reshape(shape),
        }
    }
}

// ---------------------------------------------------------------------------
// Contiguity materialisation
// ---------------------------------------------------------------------------

impl Tensor {
    /// If this tensor is already contiguous, return a new handle to the same
    /// storage with `AutogradState::None` (cheap `Arc` refcount bump — no
    /// data copy).  Otherwise, allocate a fresh dense buffer and copy every
    /// element using the stride formula.
    ///
    /// # Autograd invariant
    ///
    /// The returned tensor **always** has `AutogradState::None`, even on the
    /// fast path.  This prevents autograd state from leaking through
    /// materialisation helpers into ops that must produce clean, untracked
    /// output tensors.
    ///
    /// # Stride-to-linear index math
    ///
    /// For a tensor with shape `[s_0, s_1, …, s_{n-1}]`, strides
    /// `[t_0, t_1, …, t_{n-1}]`, and storage offset `off`, the element
    /// at multi-index `(i_0, i_1, …, i_{n-1})` lives at flat position:
    ///
    /// ```text
    /// src_idx = off + i_0*t_0 + i_1*t_1 + … + i_{n-1}*t_{n-1}
    /// ```
    ///
    /// We iterate every linear output index `dst_idx ∈ 0..numel`, decompose
    /// it into the multi-index using the *shape* (standard row-major
    /// decomposition via precomputed suffix products), then compute `src_idx`
    /// via the formula above.
    ///
    /// Complexity: `O(numel × ndim)`.  Suffix products are computed once in
    /// `O(ndim)` before the main loop, avoiding the `O(ndim²)` cost of
    /// recomputing them per element.
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
        let src = self.storage.data();

        // Precompute suffix products of the shape so that multi-index
        // decomposition is O(ndim) per element, not O(ndim²).
        //
        //   suffix[d] = shape[d+1] * shape[d+2] * … * shape[ndim-1]
        //   suffix[ndim-1] = 1
        //
        // This lets us decompose a linear index into coordinates via:
        //   coord_d   = remainder / suffix[d]
        //   remainder = remainder % suffix[d]
        let mut suffix = vec![1usize; ndim];
        for d in (0..ndim.saturating_sub(1)).rev() {
            suffix[d] = suffix[d + 1] * shape[d + 1];
        }

        let mut dst = vec![0.0f32; numel];

        for dst_idx in 0..numel {
            // Decompose dst_idx → multi-index (row-major / C-order) and
            // simultaneously accumulate the strided source index.
            let mut src_idx = offset;
            let mut remainder = dst_idx;
            for d in 0..ndim {
                let dim_size = suffix[d];
                let coord = remainder / dim_size;
                remainder %= dim_size;
                src_idx += coord * strides[d];
            }

            dst[dst_idx] = src[src_idx];
        }

        Tensor::new(dst, shape.to_vec())
    }
}

// ---------------------------------------------------------------------------
// Materialising operations — allocate new storage via CpuBackend
// ---------------------------------------------------------------------------

impl Tensor {
    /// Element-wise addition.  Returns a new contiguous tensor.
    ///
    /// Both operands are made contiguous before dispatch so the backend
    /// receives simple flat slices.
    ///
    /// # Panics
    ///
    /// Panics if `self` and `rhs` have different shapes.  Broadcasting is
    /// deferred to a later milestone.
    pub fn add(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "add: shape mismatch {:?} vs {:?}",
            self.shape(),
            rhs.shape(),
        );

        let lhs = self.contiguous();
        let rhs = rhs.contiguous();

        let mut dst = CpuBackend::zeros(lhs.numel());
        CpuBackend::add(lhs.data(), rhs.data(), &mut dst);

        Tensor::new(dst, lhs.shape().to_vec())
    }

    /// Element-wise multiplication.  Returns a new contiguous tensor.
    ///
    /// # Panics
    ///
    /// Panics if `self` and `rhs` have different shapes.
    pub fn mul(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "mul: shape mismatch {:?} vs {:?}",
            self.shape(),
            rhs.shape(),
        );

        let lhs = self.contiguous();
        let rhs = rhs.contiguous();

        let mut dst = CpuBackend::zeros(lhs.numel());
        CpuBackend::mul(lhs.data(), rhs.data(), &mut dst);

        Tensor::new(dst, lhs.shape().to_vec())
    }

    /// Matrix multiplication (`self @ rhs`).  Returns a new contiguous tensor.
    ///
    /// Both operands must be 2-D.  `self` is `(m × k)` and `rhs` is
    /// `(k × n)`; the result is `(m × n)`.
    ///
    /// # Panics
    ///
    /// - If either operand is not 2-D.
    /// - If the inner dimensions do not match (`self.shape()[1] != rhs.shape()[0]`).
    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "matmul: lhs must be 2-D, got {}-D", self.ndim());
        assert_eq!(rhs.ndim(), 2, "matmul: rhs must be 2-D, got {}-D", rhs.ndim());

        let m = self.shape()[0];
        let k = self.shape()[1];
        let n = rhs.shape()[1];

        assert_eq!(
            k,
            rhs.shape()[0],
            "matmul: inner dimension mismatch: ({} × {}) @ ({} × {})",
            m,
            k,
            rhs.shape()[0],
            n,
        );

        let lhs = self.contiguous();
        let rhs = rhs.contiguous();

        let mut dst = CpuBackend::zeros(m * n);
        CpuBackend::matmul(lhs.data(), rhs.data(), &mut dst, m, k, n);

        Tensor::new(dst, vec![m, n])
    }
}
