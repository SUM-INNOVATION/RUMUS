//! Core tensor data model for RUMUS.
//!
//! The tensor is partitioned into three strict layers:
//!
//! 1. **Storage** (`StorageHandle` / `StorageInner`) — owns raw memory, tracks mutations.
//! 2. **Layout** — maps multidimensional indices onto flat storage.
//! 3. **Autograd** (`AutogradState` / `TensorMeta`) — gradient tracking metadata.
//!
//! This separation ensures that views, reshapes, and slices share storage without
//! duplicating autograd bookkeeping, and that inference mode carries zero extra
//! allocation overhead.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

// ---------------------------------------------------------------------------
// Newtypes — zero-cost, strongly-typed identifiers
// ---------------------------------------------------------------------------

/// Opaque handle for a WGPU fence used to synchronize GPU work.
///
/// When a kernel is dispatched, the resulting fence id is stored on the
/// [`StorageInner`] so that any subsequent CPU read can wait for completion.
///
/// The inner value is **not** publicly constructible.  Construction goes
/// through [`FenceId::new`], which enforces that `usize::MAX` (the
/// [`NO_FENCE`] sentinel) is never used as a valid fence id.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FenceId(usize);

impl FenceId {
    /// Create a new fence id.
    ///
    /// # Panics (debug builds)
    ///
    /// Panics if `raw == usize::MAX`, which is reserved as the [`NO_FENCE`]
    /// sentinel inside [`StorageInner`].  In release builds this is a no-op
    /// for performance — callers must never pass the sentinel value.
    pub(crate) const fn new(raw: usize) -> Self {
        debug_assert!(raw != NO_FENCE, "FenceId cannot be usize::MAX (reserved sentinel)");
        Self(raw)
    }

    /// Return the underlying `usize` value.
    pub fn get(self) -> usize {
        self.0
    }
}

/// Identifies a gradient accumulation buffer in the autograd tape.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GradId(pub usize);

/// Identifies an operation node in the computational graph.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpId(pub usize);

/// Globally unique identifier for a learnable parameter.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParamId(pub usize);

// ---------------------------------------------------------------------------
// Storage layer
// ---------------------------------------------------------------------------

/// Sentinel value stored in the atomic fence slot to represent "no fence".
///
/// We use `usize::MAX` because valid [`FenceId`] values are allocated from a
/// monotonically increasing counter starting at 0, so `usize::MAX` will never
/// collide with a real fence in any practical program lifetime.
const NO_FENCE: usize = usize::MAX;

/// The inner, heap-allocated storage that one or more [`StorageHandle`]s share.
///
/// # Thread Safety
///
/// `StorageInner` is `Send + Sync` by construction:
/// - `data` is only mutated through paths that hold exclusive logical access
///   (enforced at a higher level by the version check, not by a lock here).
/// - `version` and `fence` are atomics and therefore safe to access from any
///   thread without external synchronization.
pub struct StorageInner {
    /// Raw element buffer.  Currently `Vec<f32>` (CPU-only placeholder);
    /// will be replaced by a `Backend`-generic allocation in a later milestone.
    pub data: Vec<f32>,

    /// Monotonically increasing mutation counter.
    ///
    /// Every in-place operation (`add_`, `zero_`, etc.) bumps this counter
    /// **after** writing to `data`.  The autograd engine snapshots the version
    /// when it records an operation; before executing backward it re-checks the
    /// snapshot against the live counter.  A mismatch means the tensor was
    /// mutated after being recorded, and backward must abort — the same
    /// invariant PyTorch enforces with its `_version` counter.
    ///
    /// ## Memory Ordering: `AcqRel`
    ///
    /// We use [`Ordering::AcqRel`] on the `fetch_add` that bumps this counter:
    ///
    /// - **Release** semantics on the writer side ensure that all preceding
    ///   stores (including writes to `data`) are visible to any thread that
    ///   later *acquires* the new version value.
    /// - **Acquire** semantics on the same RMW mean that the bumping thread
    ///   itself sees all prior writes from other threads (relevant if two
    ///   in-place ops race — the second bump observes the first).
    ///
    /// Readers use [`Ordering::Acquire`] on their `load`, which pairs with the
    /// release half of the writer's `AcqRel` to form a full acquire–release
    /// chain.
    ///
    /// ## Important: What this does *not* guarantee
    ///
    /// The version counter provides **publication ordering** for the atomic
    /// metadata itself, but it does **not** make concurrent non-atomic access
    /// to the `Vec<f32>` data buffer race-free on its own.  Data-race freedom
    /// on `data` is enforced by **aliasing exclusivity**: [`StorageHandle::data_mut`]
    /// requires sole `Arc` ownership (`Arc::get_mut`), which statically
    /// prevents any concurrent reader or writer.  The version counter is an
    /// *autograd correctness* mechanism (detect illegal in-place mutation after
    /// graph recording), not a synchronization primitive for `data` access.
    version: AtomicUsize,

    /// Placeholder for WGPU synchronization.
    ///
    /// Stored as a raw `AtomicUsize` rather than `Mutex<Option<FenceId>>` so
    /// that the GPU dispatch hot-path remains entirely lock-free.
    ///
    /// ## Sentinel Encoding
    ///
    /// - [`NO_FENCE`] (`usize::MAX`) — no pending GPU work; CPU may read
    ///   `data` immediately.
    /// - Any other value `v` — a [`FenceId(v)`](FenceId) that must be waited
    ///   on before the CPU touches `data`.
    ///
    /// Loads use `Acquire` to pair with the `Release` store that sets a new
    /// fence, ensuring the CPU sees the kernel submission that produced the
    /// fence before it tries to wait on it.
    fence: AtomicUsize,
}

impl fmt::Debug for StorageInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StorageInner")
            .field("len", &self.data.len())
            .field("version", &self.version.load(Ordering::Relaxed))
            .field("fence", &self.fence.load(Ordering::Relaxed))
            .finish()
    }
}

/// Thread-safe, reference-counted handle to a shared [`StorageInner`].
///
/// Multiple [`Tensor`]s (e.g. a tensor and its view) may hold handles to the
/// same inner storage.  The `Arc` provides shared ownership; the atomic
/// `version` inside provides mutation tracking without locking.
#[derive(Clone, Debug)]
pub struct StorageHandle {
    inner: Arc<StorageInner>,
}

impl StorageHandle {
    /// Allocate new storage owning the given `data` buffer.
    ///
    /// The version counter starts at 0 and no fence is set.
    pub fn new(data: Vec<f32>) -> Self {
        Self {
            inner: Arc::new(StorageInner {
                data,
                version: AtomicUsize::new(0),
                fence: AtomicUsize::new(NO_FENCE),
            }),
        }
    }

    /// Read the current version counter.
    ///
    /// Uses [`Ordering::Acquire`] so the caller is guaranteed to observe all
    /// data writes that happened before the most recent version bump.
    pub fn version(&self) -> usize {
        self.inner.version.load(Ordering::Acquire)
    }

    /// Atomically increment the version counter and return the **previous**
    /// version.
    ///
    /// Must be called by every in-place mutation path **after** writing to the
    /// data buffer.  See [`StorageInner::version`] for the ordering rationale.
    pub fn bump_version(&self) -> usize {
        self.inner.version.fetch_add(1, Ordering::AcqRel)
    }

    /// Return the currently stored [`FenceId`], or `None` if no GPU work is
    /// pending.
    ///
    /// Uses `Acquire` ordering so the caller sees the full submission state
    /// associated with the fence.
    pub fn fence(&self) -> Option<FenceId> {
        let raw = self.inner.fence.load(Ordering::Acquire);
        if raw == NO_FENCE {
            None
        } else {
            Some(FenceId::new(raw))
        }
    }

    /// Record a pending GPU fence on this storage.
    ///
    /// Uses `Release` ordering so that any thread that later loads this fence
    /// (with `Acquire`) also sees the kernel dispatch that produced it.
    pub fn set_fence(&self, fence: FenceId) {
        self.inner.fence.store(fence.get(), Ordering::Release);
    }

    /// Clear the pending fence, signalling that GPU work has completed.
    ///
    /// Uses `Release` ordering for symmetry with [`set_fence`](Self::set_fence).
    pub fn clear_fence(&self) {
        self.inner.fence.store(NO_FENCE, Ordering::Release);
    }

    /// Returns a shared reference to the underlying data buffer.
    ///
    /// # Safety Contract (caller-enforced)
    ///
    /// The caller must ensure no concurrent in-place mutation is in flight.
    /// In practice this is upheld by the autograd version check and by
    /// requiring `&mut self` on mutation paths at the `Tensor` API level.
    pub fn data(&self) -> &[f32] {
        &self.inner.data
    }

    /// Returns a mutable reference to the underlying data buffer.
    ///
    /// This requires an `Arc::get_mut`, which succeeds only when this
    /// `StorageHandle` is the sole owner — i.e., no views or aliases exist.
    /// Returns `None` if the storage is shared.
    pub fn data_mut(&mut self) -> Option<&mut [f32]> {
        Arc::get_mut(&mut self.inner).map(|inner| inner.data.as_mut_slice())
    }

    /// Create a [`WeakStorageHandle`] that does not keep the storage alive.
    ///
    /// Used by [`crate::autograd::VersionSnapshot`] so that version checks
    /// do not artificially extend the lifetime of intermediate tensors.
    pub fn downgrade(&self) -> WeakStorageHandle {
        WeakStorageHandle {
            inner: Arc::downgrade(&self.inner),
        }
    }
}

/// Non-owning handle to a [`StorageInner`].
///
/// Does **not** prevent the storage from being deallocated.  Used by the
/// autograd engine's [`crate::autograd::VersionSnapshot`] to check version
/// counters without keeping intermediate tensor memory alive.
///
/// Call [`upgrade`](WeakStorageHandle::upgrade) to temporarily obtain a
/// strong reference.  If the storage has already been dropped, `upgrade`
/// returns `None` — which is fine: a dead tensor cannot have been mutated
/// in-place.
#[derive(Clone)]
pub struct WeakStorageHandle {
    inner: Weak<StorageInner>,
}

impl fmt::Debug for WeakStorageHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.inner.upgrade() {
            Some(inner) => f
                .debug_struct("WeakStorageHandle")
                .field("alive", &true)
                .field("version", &inner.version.load(Ordering::Relaxed))
                .finish(),
            None => f
                .debug_struct("WeakStorageHandle")
                .field("alive", &false)
                .finish(),
        }
    }
}

impl WeakStorageHandle {
    /// Attempt to obtain a strong reference to the storage.
    ///
    /// Returns `None` if all strong references have been dropped (the
    /// tensor data has been freed).
    pub fn upgrade(&self) -> Option<StorageHandle> {
        self.inner.upgrade().map(|inner| StorageHandle { inner })
    }
}

// StorageInner is Send + Sync: Vec<f32> is Send+Sync, atomics are Send+Sync.
// The compiler derives this automatically, but we assert it here so a future
// field addition that breaks the invariant becomes a compile error.
const _: () = {
    fn _assert_send<T: Send>() {}
    fn _assert_sync<T: Sync>() {}
    fn _assertions() {
        _assert_send::<StorageInner>();
        _assert_sync::<StorageInner>();
        _assert_send::<StorageHandle>();
        _assert_sync::<StorageHandle>();
    }
};

// ---------------------------------------------------------------------------
// Layout layer
// ---------------------------------------------------------------------------

/// Describes how a multidimensional tensor maps onto flat storage.
///
/// A tensor's element at index `[i_0, i_1, ..., i_{n-1}]` lives at position
/// `offset + sum(i_k * strides[k])` in the underlying [`StorageInner::data`]
/// buffer.  Views, transposes, and slices create new `Layout`s that share the
/// same storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    /// Size of each dimension.
    shape: Vec<usize>,
    /// Number of elements to step in flat storage when advancing one position
    /// along each dimension.
    strides: Vec<usize>,
    /// Element offset into the storage buffer where this view begins.
    offset: usize,
}

impl Layout {
    /// Build a contiguous, row-major (C-order) layout for the given `shape`.
    ///
    /// Strides are computed right-to-left so that the last dimension is
    /// contiguous (stride 1).  Offset is 0.
    pub fn contiguous(shape: Vec<usize>) -> Self {
        let ndim = shape.len();
        let mut strides = vec![0usize; ndim];
        if ndim > 0 {
            strides[ndim - 1] = 1;
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        Self {
            shape,
            strides,
            offset: 0,
        }
    }

    /// Number of dimensions (rank).
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements described by this layout.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns `true` if the layout describes a dense, row-major block with
    /// no gaps and no overlap — i.e., a simple `&[f32]` slice starting at
    /// `offset` with length `numel()` covers exactly the right elements.
    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }
        let mut expected_stride = 1usize;
        for i in (0..self.ndim()).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i];
        }
        true
    }

    /// Borrow the shape slice.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Borrow the strides slice.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Element offset into storage.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Return a new layout with axes `dim0` and `dim1` swapped.
    ///
    /// This is a pure metadata operation — shape and strides are swapped,
    /// offset is preserved, no data is moved.  The resulting layout is
    /// almost always non-contiguous (unless a swapped dimension has size 1).
    ///
    /// # Panics
    ///
    /// Panics if either dimension index is out of range.
    pub fn transposed(&self, dim0: usize, dim1: usize) -> Self {
        assert!(
            dim0 < self.ndim() && dim1 < self.ndim(),
            "transpose dims ({}, {}) out of range for {}-D tensor",
            dim0,
            dim1,
            self.ndim(),
        );
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.swap(dim0, dim1);
        strides.swap(dim0, dim1);
        Self {
            shape,
            strides,
            offset: self.offset,
        }
    }

    /// Attempt a zero-copy reshape.
    ///
    /// Returns `Some(new_layout)` if the current layout is contiguous (so a
    /// fresh set of row-major strides correctly maps the same flat data).
    /// Returns `None` if the layout is strided — the caller must materialise
    /// a contiguous copy first.
    ///
    /// # Panics
    ///
    /// Panics if `new_shape` has a different element count than the current
    /// shape.
    pub fn reshaped(&self, new_shape: Vec<usize>) -> Option<Self> {
        let old_numel: usize = self.shape.iter().product();
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(
            old_numel, new_numel,
            "cannot reshape {} elements into shape {:?} ({} elements)",
            old_numel, new_shape, new_numel,
        );
        if !self.is_contiguous() {
            return None;
        }
        Some(Self::contiguous_with_offset(new_shape, self.offset))
    }

    /// Build a contiguous (row-major) layout at an arbitrary storage offset.
    fn contiguous_with_offset(shape: Vec<usize>, offset: usize) -> Self {
        let ndim = shape.len();
        let mut strides = vec![0usize; ndim];
        if ndim > 0 {
            strides[ndim - 1] = 1;
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        Self {
            shape,
            strides,
            offset,
        }
    }
}

// ---------------------------------------------------------------------------
// Autograd layer
// ---------------------------------------------------------------------------

/// Gradient-tracking metadata attached to tensors that participate in the
/// computational graph.
///
/// Wrapped in `Arc` inside [`AutogradState::Tracked`] so that multiple
/// `Tensor` handles referencing the same logical tensor (e.g. `y = x + x`)
/// share a single `TensorMeta`.  This is essential for Tenet #3 (Strict
/// Graph Edge Counting): when `x` is used as both inputs to `add`, both
/// usages increment the **same** `total_grads` counter.
///
/// Inference-mode tensors pay no heap cost — the `Arc<TensorMeta>` only
/// exists in the `Tracked` variant.
pub struct TensorMeta {
    /// Whether this tensor requires gradient computation.
    pub requires_grad: bool,
    /// Handle into the gradient accumulation buffer registry.
    pub grad_id: Option<GradId>,
    /// The operation that produced this tensor (`None` for leaf tensors).
    pub creator: Option<OpId>,
    /// `true` if the user created this tensor directly (not via an op).
    pub is_leaf: bool,
    /// `true` if the gradient should be retained after `backward()` even for
    /// non-leaf tensors (mirrors `torch.Tensor.retain_grad()`).
    pub retains_grad: bool,
    /// Number of **incoming gradient edges** — i.e., how many downstream
    /// operations consume this tensor as an input.
    ///
    /// This is an [`AtomicUsize`] because multiple forward ops may reference
    /// the same tensor (via cloned `Arc<TensorMeta>`) and each must
    /// increment the counter through a shared `&` reference.
    ///
    /// ## Ordering: `Relaxed`
    ///
    /// The forward pass is single-threaded per tape (one thread-local tape
    /// per thread).  The backward pass reads this value only after the
    /// forward pass has completed and the tape has been taken from the
    /// thread-local — the take provides the happens-before edge.
    /// `Relaxed` is therefore sufficient.
    pub total_grads: AtomicUsize,
}

impl TensorMeta {
    /// Construct metadata for a user-created leaf tensor.
    ///
    /// - `creator` is `None` (no parent op).
    /// - `is_leaf` is `true`.
    /// - `total_grads` starts at 0 and is incremented as downstream ops
    ///   record this tensor as an input.
    pub fn leaf(requires_grad: bool) -> Self {
        Self {
            requires_grad,
            grad_id: None,
            creator: None,
            is_leaf: true,
            retains_grad: false,
            total_grads: AtomicUsize::new(0),
        }
    }
}

impl fmt::Debug for TensorMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorMeta")
            .field("requires_grad", &self.requires_grad)
            .field("grad_id", &self.grad_id)
            .field("creator", &self.creator)
            .field("is_leaf", &self.is_leaf)
            .field("retains_grad", &self.retains_grad)
            .field("total_grads", &self.total_grads.load(Ordering::Relaxed))
            .finish()
    }
}

/// Controls whether a [`Tensor`] participates in automatic differentiation.
///
/// This is an enum — not an `Option` — so the two states are named and
/// self-documenting at every match site.
///
/// # Zero-allocation inference
///
/// When `AutogradState` is `None`, the tensor carries **no** heap allocation
/// for gradient metadata.  The `Arc<TensorMeta>` only exists in the `Tracked`
/// variant, so switching an entire model to inference mode eliminates all
/// autograd overhead without touching the storage or layout layers.
///
/// # Shared metadata via `Arc`
///
/// `Tracked` holds an `Arc<TensorMeta>` (not `Box`), so cloning a tracked
/// tensor shares the metadata rather than duplicating it.  This is critical
/// for correct edge counting in graphs like `y = x + x`, where both usages
/// of `x` must increment the same `total_grads` counter.
#[derive(Debug, Clone)]
pub enum AutogradState {
    /// Inference mode — no gradient tracking, no extra allocation.
    None,
    /// Training mode — gradient metadata is shared via `Arc`.
    Tracked(Arc<TensorMeta>),
}

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------

/// The top-level tensor type composing storage, layout, and autograd state.
///
/// `Tensor` is the primary user-facing handle.  Cloning a `Tensor` is cheap:
/// it bumps the `Arc` refcount on storage, clones the layout vectors, and
/// clones the autograd state.
#[derive(Clone)]
pub struct Tensor {
    /// Shared, reference-counted raw memory.
    pub(crate) storage: StorageHandle,
    /// Multidimensional view into storage.
    pub(crate) layout: Layout,
    /// Gradient tracking; `AutogradState::None` in inference mode.
    pub(crate) state: AutogradState,
}

impl Tensor {
    /// Create a new contiguous, inference-mode tensor from raw data.
    ///
    /// The `shape` must be compatible with `data.len()`.
    ///
    /// # Panics
    ///
    /// Panics if the product of `shape` does not equal `data.len()`.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            numel,
            data.len(),
            "shape {:?} expects {} elements but got {}",
            shape,
            numel,
            data.len(),
        );
        let layout = Layout::contiguous(shape);
        Self {
            storage: StorageHandle::new(data),
            layout,
            state: AutogradState::None,
        }
    }

    /// Shape of the tensor as a slice.
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    /// Strides of the tensor as a slice.
    pub fn strides(&self) -> &[usize] {
        self.layout.strides()
    }

    /// Number of dimensions (rank).
    pub fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.layout.numel()
    }

    /// Whether this tensor's layout is contiguous in row-major order.
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// Whether this tensor requires gradient computation.
    pub fn requires_grad(&self) -> bool {
        match &self.state {
            AutogradState::None => false,
            AutogradState::Tracked(meta) => meta.requires_grad,
        }
    }

    /// Current storage version counter (see [`StorageInner::version`]).
    pub fn version(&self) -> usize {
        self.storage.version()
    }

    /// Read-only access to the underlying data buffer.
    pub fn data(&self) -> &[f32] {
        self.storage.data()
    }

    /// Enable gradient tracking on this tensor, making it a leaf in the
    /// computational graph.
    ///
    /// Creates a fresh `Arc<TensorMeta>` with a new [`GradId`] from the
    /// thread-local autograd context.  If the tensor is already tracked,
    /// this replaces the metadata entirely (a new Arc, new GradId).
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        match &self.state {
            AutogradState::None if requires_grad => {
                let grad_id = crate::autograd::context::next_grad_id();
                let mut meta = TensorMeta::leaf(true);
                meta.grad_id = Some(grad_id);
                self.state = AutogradState::Tracked(Arc::new(meta));
            }
            AutogradState::Tracked(_) if !requires_grad => {
                self.state = AutogradState::None;
            }
            // Already in the requested state — no-op.
            _ => {}
        }
    }

    /// Access the shared [`TensorMeta`], if this tensor is tracked.
    pub(crate) fn meta(&self) -> Option<&Arc<TensorMeta>> {
        match &self.state {
            AutogradState::Tracked(meta) => Some(meta),
            AutogradState::None => None,
        }
    }

    /// Get this tensor's [`GradId`], if tracked and assigned.
    pub fn grad_id(&self) -> Option<GradId> {
        match &self.state {
            AutogradState::Tracked(meta) => meta.grad_id,
            AutogradState::None => None,
        }
    }

    /// Construct a tensor from raw components, with no autograd state.
    ///
    /// Used by the backward engine to reconstruct saved input tensors
    /// from their `StorageHandle` + `Layout` without attaching any
    /// gradient tracking metadata.
    pub(crate) fn from_storage_and_layout(
        storage: StorageHandle,
        layout: Layout,
    ) -> Tensor {
        Tensor {
            storage,
            layout,
            state: AutogradState::None,
        }
    }
}

// ---------------------------------------------------------------------------
// Parameter
// ---------------------------------------------------------------------------

/// A learnable parameter: a [`Tensor`] paired with a globally unique
/// [`ParamId`].
///
/// `Parameter` is the unit of registration in a module's parameter list and
/// the unit the optimizer iterates over.  The `ParamId` allows the optimizer
/// to maintain per-parameter state (momentum buffers, etc.) across calls.
pub struct Parameter {
    /// The underlying tensor holding the parameter's value.
    pub tensor: Tensor,
    /// Unique identifier for this parameter.
    pub id: ParamId,
}

impl Parameter {
    /// Wrap an existing tensor as a named parameter.
    ///
    /// `ParamId` allocation strategy is deferred to the module/optimizer
    /// milestone — for now the caller provides the id.
    pub fn new(tensor: Tensor, id: ParamId) -> Self {
        Self { tensor, id }
    }
}
