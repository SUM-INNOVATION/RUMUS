// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Core tensor data model for RUMUS.
//!
//! The tensor is partitioned into three strict layers:
//!
//! 1. **Storage** (`StorageHandle` / `StorageInner`) — owns raw memory, tracks mutations.
//! 2. **Layout** — maps multidimensional indices onto flat storage.
//! 3. **Autograd** (`AutogradState` / `TensorMeta`) — gradient tracking metadata.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::RwLock;

// ---------------------------------------------------------------------------
// Newtypes — zero-cost, strongly-typed identifiers
// ---------------------------------------------------------------------------

/// Opaque handle for a WGPU fence used to synchronize GPU work.
///
/// The inner value is **not** publicly constructible.  Construction goes
/// through [`FenceId::new`], which enforces that `usize::MAX` (the
/// [`NO_FENCE`] sentinel) is never used as a valid fence id.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FenceId(usize);

impl FenceId {
    pub(crate) const fn new(raw: usize) -> Self {
        debug_assert!(raw != NO_FENCE, "FenceId cannot be usize::MAX (reserved sentinel)");
        Self(raw)
    }

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
// DType — element type tag
// ---------------------------------------------------------------------------

/// Element data type for tensor storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    /// Symmetric block-quantized INT8.  Each block of `block_size` elements
    /// has a 4-byte header (f16 scale in lower 16 bits, 2 bytes padding)
    /// followed by `block_size` i8 values.
    Q8 { block_size: usize },
}

impl DType {
    /// Size in bytes of a single element (not meaningful for Q8 — use
    /// `gpu_buf_size` instead).
    pub fn byte_size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::Q8 { .. } => 1, // logical element is i8, but use gpu_buf_size for real sizing
        }
    }

    /// Compute the GPU buffer byte size for `numel` elements, aligned to
    /// WebGPU's `COPY_BUFFER_ALIGNMENT` (4 bytes).
    ///
    /// F32: `numel * 4` (always aligned).
    /// F16: `(numel * 2 + 3) & !3` (round up to next 4-byte boundary).
    /// Q8:  `num_blocks * (4 + block_size) + padding` — 4-byte header per block.
    pub fn gpu_buf_size(self, numel: usize) -> u64 {
        let raw = match self {
            DType::F32 => numel * 4,
            DType::F16 => numel * 2,
            DType::Q8 { block_size } => {
                let num_blocks = (numel + block_size - 1) / block_size;
                // 4-byte header (f16 scale + 2B pad) + block_size i8 values per block
                num_blocks * (4 + block_size)
            }
        };
        ((raw + 3) & !3) as u64
    }

    /// Stride in bytes for one Q8 block (4-byte header + block_size data).
    /// Panics if not Q8.
    pub fn q8_block_stride(self) -> usize {
        match self {
            DType::Q8 { block_size } => 4 + block_size,
            _ => panic!("q8_block_stride called on non-Q8 dtype"),
        }
    }

    /// Returns true if this is a quantized type.
    pub fn is_quantized(self) -> bool {
        matches!(self, DType::Q8 { .. })
    }
}

// ---------------------------------------------------------------------------
// Storage layer
// ---------------------------------------------------------------------------

const NO_FENCE: usize = usize::MAX;

// --- Public type aliases for the mapped lock guards -------------------------
// These insulate external code from knowing about `parking_lot` directly.

/// Read guard for tensor data.  Derefs to `&[f32]`.
pub type DataReadGuard<'a> = parking_lot::MappedRwLockReadGuard<'a, [f32]>;

/// Write guard for tensor data.  Derefs to `&mut [f32]`.
pub type DataWriteGuard<'a> = parking_lot::MappedRwLockWriteGuard<'a, [f32]>;

// --- StorageData enum -------------------------------------------------------

/// Tracks which side of a CPU/GPU pair was most recently written.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Gpu variant prepared for Chunk 2
pub(crate) enum DirtySide {
    /// CPU has the latest data; GPU copy is stale.
    Cpu,
    /// GPU has the latest data; CPU copy is stale.
    Gpu,
    /// Both sides are in sync.
    Clean,
}

/// Where the tensor's data physically lives.
///
/// When the `gpu` feature is disabled, this enum has a single `Cpu` variant
/// and the compiler optimises away all match overhead.
#[allow(dead_code)] // Gpu/Both/Transferring variants prepared for Chunk 2
pub(crate) enum StorageData {
    /// CPU-resident data.
    Cpu(Vec<f32>),

    /// GPU-resident data.
    #[cfg(feature = "gpu")]
    Gpu {
        buffer: wgpu::Buffer,
        len: usize,
    },

    /// Data exists on both CPU and GPU; `dirty` tracks which is stale.
    #[cfg(feature = "gpu")]
    Both {
        cpu: Vec<f32>,
        gpu: wgpu::Buffer,
        dirty: DirtySide,
    },

    /// Transient state during a device transfer.
    ///
    /// The data has been extracted by `ensure_cpu`/`ensure_gpu` and the
    /// write lock has been released while the blocking transfer runs.
    /// Any thread that encounters this variant should panic — concurrent
    /// access to a tensor that is actively being transferred is an
    /// anti-pattern in our eager execution model.
    #[cfg(feature = "gpu")]
    Transferring,

    /// JIT deferred: storage will be populated when the fusion block flushes.
    #[cfg(feature = "jit")]
    Deferred {
        var_id: usize,
    },
}

impl fmt::Debug for StorageData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageData::Cpu(v) => f.debug_tuple("Cpu").field(&v.len()).finish(),
            #[cfg(feature = "gpu")]
            StorageData::Gpu { len, .. } => {
                f.debug_struct("Gpu").field("len", len).finish()
            }
            #[cfg(feature = "gpu")]
            StorageData::Both { cpu, dirty, .. } => f
                .debug_struct("Both")
                .field("len", &cpu.len())
                .field("dirty", dirty)
                .finish(),
            #[cfg(feature = "gpu")]
            StorageData::Transferring => write!(f, "Transferring"),
            #[cfg(feature = "jit")]
            StorageData::Deferred { var_id } => {
                f.debug_struct("Deferred").field("var_id", var_id).finish()
            }
        }
    }
}

// --- StorageInner -----------------------------------------------------------

/// The inner, heap-allocated storage that one or more [`StorageHandle`]s share.
///
/// # Thread Safety
///
/// - `data` is a `parking_lot::RwLock<StorageData>`.  Concurrent reads are
///   allowed; exclusive writes are used by the optimizer and device transfers.
/// - `version` and `fence` are atomics — safe from any thread.
pub struct StorageInner {
    data: RwLock<StorageData>,
    /// Element count.  Immutable after construction.
    len: usize,
    /// Element data type.  Immutable after construction.
    dtype: DType,
    /// Which GPU device this storage lives on (0 = primary, default).
    pub(crate) device_index: usize,
    /// Monotonically increasing mutation counter for autograd version checking.
    version: AtomicUsize,
    /// Per-resource WGPU fence (AtomicUsize with `usize::MAX` sentinel).
    fence: AtomicUsize,
}

impl fmt::Debug for StorageInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StorageInner")
            .field("len", &self.len)
            .field("version", &self.version.load(Ordering::Relaxed))
            .field("fence", &self.fence.load(Ordering::Relaxed))
            .finish()
    }
}

/// When the last `Arc<StorageInner>` is dropped, return any GPU buffer
/// to the [`BufferPool`] for reuse instead of letting `wgpu` deallocate it.
///
/// This is the critical resource lifecycle hook that makes the pool work:
/// intermediate tensors from forward/backward passes release their buffers
/// back to the pool, which are then recycled by `pool.acquire()` on the
/// next iteration — zero GPU allocation churn.
#[cfg(feature = "gpu")]
impl Drop for StorageInner {
    fn drop(&mut self) {
        // Extract the StorageData — we own it since we're being dropped.
        let data = self.data.get_mut();
        let placeholder = StorageData::Cpu(Vec::new());
        let owned = std::mem::replace(data, placeholder);

        match owned {
            StorageData::Gpu { buffer, .. } => {
                if let Some(ctx) = crate::backend::gpu::context::GpuContext::get() {
                    ctx.pool.release(buffer);
                }
                // If no GPU context (shutdown), buffer drops normally.
            }
            StorageData::Both { gpu, .. } => {
                if let Some(ctx) = crate::backend::gpu::context::GpuContext::get() {
                    ctx.pool.release(gpu);
                }
            }
            // Cpu / Transferring: nothing to recycle.
            _ => {}
        }
    }
}

// --- StorageHandle ----------------------------------------------------------

/// Thread-safe, reference-counted handle to a shared [`StorageInner`].
///
/// Multiple [`Tensor`]s (e.g. a tensor and its view) may hold handles to the
/// same inner storage.
#[derive(Clone, Debug)]
pub struct StorageHandle {
    inner: Arc<StorageInner>,
}

impl StorageHandle {
    /// Allocate new CPU storage owning the given `data` buffer (F32).
    pub fn new(data: Vec<f32>) -> Self {
        let len = data.len();
        Self {
            inner: Arc::new(StorageInner {
                data: RwLock::new(StorageData::Cpu(data)),
                len,
                dtype: DType::F32,
                device_index: 0,
                version: AtomicUsize::new(0),
                fence: AtomicUsize::new(NO_FENCE),
            }),
        }
    }

    /// Create new GPU-resident storage wrapping a `wgpu::Buffer`.
    #[cfg(feature = "gpu")]
    pub fn new_gpu(buffer: wgpu::Buffer, len: usize) -> Self {
        Self {
            inner: Arc::new(StorageInner {
                data: RwLock::new(StorageData::Gpu { buffer, len }),
                len,
                dtype: DType::F32,
                device_index: 0,
                version: AtomicUsize::new(0),
                fence: AtomicUsize::new(NO_FENCE),
            }),
        }
    }

    /// Create GPU-resident F16 storage wrapping a `wgpu::Buffer`.
    #[cfg(feature = "gpu")]
    pub fn new_gpu_f16(buffer: wgpu::Buffer, len: usize) -> Self {
        Self {
            inner: Arc::new(StorageInner {
                data: RwLock::new(StorageData::Gpu { buffer, len }),
                len,
                dtype: DType::F16,
                device_index: 0,
                version: AtomicUsize::new(0),
                fence: AtomicUsize::new(NO_FENCE),
            }),
        }
    }

    /// Create GPU-resident Q8 storage wrapping a `wgpu::Buffer`.
    ///
    /// `len` is the logical element count (number of original values, not bytes).
    #[cfg(feature = "gpu")]
    pub fn new_gpu_q8(buffer: wgpu::Buffer, len: usize, block_size: usize) -> Self {
        Self {
            inner: Arc::new(StorageInner {
                data: RwLock::new(StorageData::Gpu { buffer, len }),
                len,
                dtype: DType::Q8 { block_size },
                device_index: 0,
                version: AtomicUsize::new(0),
                fence: AtomicUsize::new(NO_FENCE),
            }),
        }
    }

    /// Create a deferred (JIT placeholder) storage.
    ///
    /// The actual GPU buffer is populated when the JIT fusion block flushes.
    #[cfg(feature = "jit")]
    pub fn new_deferred(var_id: usize, len: usize, dtype: DType) -> Self {
        Self {
            inner: Arc::new(StorageInner {
                data: RwLock::new(StorageData::Deferred { var_id }),
                len,
                dtype,
                device_index: 0,
                version: AtomicUsize::new(0),
                fence: AtomicUsize::new(NO_FENCE),
            }),
        }
    }

    /// Replace the internal StorageData (used by JIT flush to materialize deferred tensors).
    #[cfg(feature = "jit")]
    pub fn materialize_gpu(&self, buffer: wgpu::Buffer) {
        let mut guard = self.inner.data.write();
        *guard = StorageData::Gpu { buffer, len: self.inner.len };
    }

    /// Number of elements in this storage.
    pub fn len(&self) -> usize {
        self.inner.len
    }

    /// GPU device index this storage lives on.
    pub fn device_index(&self) -> usize {
        self.inner.device_index
    }

    /// Set the device index (only valid on freshly created storage with refcount 1).
    pub(crate) fn set_device_index(&mut self, idx: usize) {
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            inner.device_index = idx;
        }
    }

    /// Element data type.
    pub fn dtype(&self) -> DType {
        self.inner.dtype
    }

    /// Unique identity based on the storage pointer address.
    /// Used by the ONNX tracer to track value names across ops.
    pub fn ptr_id(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    /// Download the raw GPU buffer bytes without any dtype conversion.
    ///
    /// Returns the raw bytes as they exist on the GPU.  For F16 tensors,
    /// this returns f16 bytes (2 per element).  For F32, f32 bytes (4 per element).
    /// Used by the ONNX exporter to preserve native F16 weight data.
    #[cfg(feature = "gpu")]
    pub fn download_raw_bytes(&self) -> Vec<u8> {
        self.ensure_gpu();
        let guard = self.inner.data.read();
        let buffer = match &*guard {
            StorageData::Gpu { buffer, .. } => buffer,
            StorageData::Both { gpu, .. } => gpu,
            _ => unreachable!("ensure_gpu guarantees GPU data"),
        };
        let byte_size = self.inner.dtype.gpu_buf_size(self.inner.len);
        let ctx = self.device_ctx();
        ctx.download_raw_bytes(buffer, byte_size)
    }

    /// Get the GPU context for this storage's device.
    ///
    /// When `multi_gpu` is active, returns the context for the specific
    /// `device_index`.  Otherwise returns the primary context.
    #[cfg(feature = "gpu")]
    fn device_ctx(&self) -> &'static crate::backend::gpu::context::GpuContext {
        #[cfg(feature = "multi_gpu")]
        {
            let mgpu = crate::backend::gpu::context::MultiGpuContext::get()
                .expect("GPU context required");
            return mgpu.device(self.inner.device_index);
        }
        #[cfg(not(feature = "multi_gpu"))]
        {
            crate::backend::gpu::context::GpuContext::get()
                .expect("GPU context required")
        }
    }

    /// Returns `true` if the data is (at least partially) on the GPU.
    #[cfg(feature = "gpu")]
    pub fn is_gpu(&self) -> bool {
        matches!(
            &*self.inner.data.read(),
            StorageData::Gpu { .. } | StorageData::Both { .. }
        )
    }

    /// Mark the GPU copy as having the latest data (CPU copy is stale).
    ///
    /// Called after a GPU compute kernel writes to the buffer (e.g.,
    /// optimizer step).  The `wgpu::Buffer` handle is written to by the
    /// GPU asynchronously — this method records that the CPU side needs
    /// a D2H transfer before it can be read.
    #[cfg(feature = "gpu")]
    pub fn mark_gpu_dirty(&self) {
        let mut guard = self.inner.data.write();
        if let StorageData::Both { dirty, .. } = &mut *guard {
            *dirty = DirtySide::Gpu;
        }
    }

    /// Read the current version counter (`Acquire` ordering).
    pub fn version(&self) -> usize {
        self.inner.version.load(Ordering::Acquire)
    }

    /// Atomically increment the version counter (`AcqRel`).  Returns the
    /// **previous** version.
    pub fn bump_version(&self) -> usize {
        self.inner.version.fetch_add(1, Ordering::AcqRel)
    }

    /// Return the currently stored [`FenceId`], or `None` if no GPU work
    /// is pending.
    pub fn fence(&self) -> Option<FenceId> {
        let raw = self.inner.fence.load(Ordering::Acquire);
        if raw == NO_FENCE {
            None
        } else {
            Some(FenceId::new(raw))
        }
    }

    /// Record a pending GPU fence on this storage.
    pub fn set_fence(&self, fence: FenceId) {
        self.inner.fence.store(fence.get(), Ordering::Release);
    }

    /// Clear the pending fence, signalling that GPU work has completed.
    pub fn clear_fence(&self) {
        self.inner.fence.store(NO_FENCE, Ordering::Release);
    }

    /// Acquire a **read** lock on the CPU data.
    ///
    /// Returns a [`DataReadGuard`] that derefs to `&[f32]`.
    /// If the data is GPU-only (feature `gpu`), triggers a blocking
    /// device-to-host transfer first.
    ///
    /// **F16 tensors:** If this storage is `DType::F16`, the data is
    /// cast to F32 on the GPU, downloaded, and the CPU mirror replaced.
    /// This enables inspection/printing of F16 tensors without panicking.
    pub fn data(&self) -> DataReadGuard<'_> {
        #[cfg(feature = "gpu")]
        match self.inner.dtype {
            DType::F16 => self.ensure_cpu_f16_as_f32(),
            DType::Q8 { .. } => self.ensure_cpu_q8_as_f32(),
            _ => self.ensure_cpu(),
        }

        #[cfg(not(feature = "gpu"))]
        assert_eq!(self.inner.dtype, DType::F32, "F16/Q8 tensors require GPU feature");

        parking_lot::RwLockReadGuard::map(self.inner.data.read(), |sd| match sd {
            StorageData::Cpu(v) => v.as_slice(),
            #[cfg(feature = "gpu")]
            StorageData::Both { cpu, .. } => cpu.as_slice(),
            #[cfg(feature = "gpu")]
            StorageData::Gpu { .. } => unreachable!("ensure_cpu guarantees CPU data"),
            #[cfg(feature = "gpu")]
            StorageData::Transferring => {
                panic!("cannot read tensor data while a device transfer is in progress")
            }
            #[cfg(feature = "jit")]
            StorageData::Deferred { .. } => {
                panic!("cannot read JIT-deferred tensor — flush the JIT block first")
            }
        })
    }

    /// For F16 GPU tensors: cast to F32 on GPU, download as Vec<f32>.
    /// Stores the result as a CPU F32 mirror so data() can return &[f32].
    #[cfg(feature = "gpu")]
    fn ensure_cpu_f16_as_f32(&self) {
        // Fast path: already has a CPU mirror.
        {
            let guard = self.inner.data.read();
            match &*guard {
                StorageData::Cpu(_) => return,
                StorageData::Both {
                    dirty: DirtySide::Clean | DirtySide::Cpu,
                    ..
                } => return,
                _ => {}
            }
        }

        // Need to cast F16 → F32 on GPU, then download.
        let ctx = self.device_ctx();

        // Extract the GPU buffer.
        let extracted = {
            let mut guard = self.inner.data.write();
            match &*guard {
                StorageData::Cpu(_) => return,
                StorageData::Both {
                    dirty: DirtySide::Clean | DirtySide::Cpu,
                    ..
                } => return,
                _ => {}
            }
            std::mem::replace(&mut *guard, StorageData::Transferring)
        };

        let new_state = match extracted {
            StorageData::Gpu { buffer, len } => {
                let f32_buf = ctx.pool.acquire(
                    &ctx.device,
                    (len * 4) as u64,
                    crate::backend::gpu::context::STORAGE_USAGE,
                );
                crate::backend::gpu::compute::cast_f16_to_f32_dispatch(
                    ctx, &buffer, &f32_buf, len as u32,
                );
                let cpu_data = ctx.download(&f32_buf, len);
                ctx.pool.release(f32_buf);
                StorageData::Both {
                    cpu: cpu_data,
                    gpu: buffer,
                    dirty: DirtySide::Clean,
                }
            }
            StorageData::Both {
                gpu,
                dirty: DirtySide::Gpu,
                ..
            } => {
                let len = self.inner.len;
                let f32_buf = ctx.pool.acquire(
                    &ctx.device,
                    (len * 4) as u64,
                    crate::backend::gpu::context::STORAGE_USAGE,
                );
                crate::backend::gpu::compute::cast_f16_to_f32_dispatch(
                    ctx, &gpu, &f32_buf, len as u32,
                );
                let cpu_data = ctx.download(&f32_buf, len);
                ctx.pool.release(f32_buf);
                StorageData::Both {
                    cpu: cpu_data,
                    gpu,
                    dirty: DirtySide::Clean,
                }
            }
            other => other,
        };

        *self.inner.data.write() = new_state;
    }

    /// For Q8 GPU tensors: dequantize to F32 on GPU, download as Vec<f32>.
    #[cfg(feature = "gpu")]
    fn ensure_cpu_q8_as_f32(&self) {
        {
            let guard = self.inner.data.read();
            match &*guard {
                StorageData::Cpu(_) => return,
                StorageData::Both {
                    dirty: DirtySide::Clean | DirtySide::Cpu,
                    ..
                } => return,
                _ => {}
            }
        }

        let ctx = self.device_ctx();

        let extracted = {
            let mut guard = self.inner.data.write();
            match &*guard {
                StorageData::Cpu(_) => return,
                StorageData::Both {
                    dirty: DirtySide::Clean | DirtySide::Cpu,
                    ..
                } => return,
                _ => {}
            }
            std::mem::replace(&mut *guard, StorageData::Transferring)
        };

        let block_size = match self.inner.dtype {
            DType::Q8 { block_size } => block_size,
            _ => unreachable!(),
        };

        let new_state = match extracted {
            StorageData::Gpu { buffer, len } => {
                let f32_buf = ctx.pool.acquire(
                    &ctx.device,
                    (len * 4) as u64,
                    crate::backend::gpu::context::STORAGE_USAGE,
                );
                crate::backend::gpu::compute::dequantize_dispatch(
                    ctx, &buffer, &f32_buf, len as u32, block_size as u32,
                );
                let cpu_data = ctx.download(&f32_buf, len);
                ctx.pool.release(f32_buf);
                StorageData::Both {
                    cpu: cpu_data,
                    gpu: buffer,
                    dirty: DirtySide::Clean,
                }
            }
            StorageData::Both {
                gpu,
                dirty: DirtySide::Gpu,
                ..
            } => {
                let len = self.inner.len;
                let f32_buf = ctx.pool.acquire(
                    &ctx.device,
                    (len * 4) as u64,
                    crate::backend::gpu::context::STORAGE_USAGE,
                );
                crate::backend::gpu::compute::dequantize_dispatch(
                    ctx, &gpu, &f32_buf, len as u32, block_size as u32,
                );
                let cpu_data = ctx.download(&f32_buf, len);
                ctx.pool.release(f32_buf);
                StorageData::Both {
                    cpu: cpu_data,
                    gpu,
                    dirty: DirtySide::Clean,
                }
            }
            other => other,
        };

        *self.inner.data.write() = new_state;
    }

    /// Acquire an exclusive **write** lock on the CPU data.
    ///
    /// Returns a [`DataWriteGuard`] that derefs to `&mut [f32]`.
    /// If a GPU copy exists (`Both` state), it is marked stale.
    /// Caller **must** call [`bump_version`](Self::bump_version) after
    /// writing and dropping the guard.
    pub fn data_write(&self) -> DataWriteGuard<'_> {
        #[cfg(feature = "gpu")]
        self.ensure_cpu();

        parking_lot::RwLockWriteGuard::map(self.inner.data.write(), |sd| match sd {
            StorageData::Cpu(v) => v.as_mut_slice(),
            #[cfg(feature = "gpu")]
            StorageData::Both { cpu, dirty, .. } => {
                *dirty = DirtySide::Cpu;
                cpu.as_mut_slice()
            }
            #[cfg(feature = "gpu")]
            StorageData::Gpu { .. } => unreachable!("ensure_cpu guarantees CPU data"),
            #[cfg(feature = "gpu")]
            StorageData::Transferring => {
                panic!("cannot write tensor data while a device transfer is in progress")
            }
            #[cfg(feature = "jit")]
            StorageData::Deferred { .. } => {
                panic!("cannot write JIT-deferred tensor — flush the JIT block first")
            }
        })
    }

    /// Create a [`WeakStorageHandle`] that does not keep the storage alive.
    pub fn downgrade(&self) -> WeakStorageHandle {
        WeakStorageHandle {
            inner: Arc::downgrade(&self.inner),
        }
    }

    // --- GPU transfer methods (feature-gated) -------------------------------

    /// Acquire a read lock on the GPU buffer.
    ///
    /// Triggers a host-to-device transfer if the data is CPU-only.
    #[cfg(feature = "gpu")]
    pub fn gpu_buffer(
        &self,
    ) -> parking_lot::MappedRwLockReadGuard<'_, wgpu::Buffer> {
        self.ensure_gpu();
        parking_lot::RwLockReadGuard::map(self.inner.data.read(), |sd| match sd {
            StorageData::Gpu { buffer, .. } => buffer,
            StorageData::Both { gpu, .. } => gpu,
            StorageData::Cpu(_) => unreachable!("ensure_gpu guarantees GPU data"),
            StorageData::Transferring => {
                panic!("cannot access GPU buffer while a device transfer is in progress")
            }
            #[cfg(feature = "jit")]
            StorageData::Deferred { .. } => {
                panic!("cannot access GPU buffer of JIT-deferred tensor — flush the JIT block first")
            }
        })
    }

    /// Ensure the CPU copy exists and is up-to-date.
    ///
    /// - `Cpu` / `Both { dirty != Gpu }` → no-op.
    /// - `Gpu` → download GPU→CPU, transition to `Both { dirty: Clean }`.
    /// - `Both { dirty: Gpu }` → re-download, set `Clean`.
    ///
    /// # Lock protocol
    ///
    /// The write lock is held **only** to swap the state to `Transferring`
    /// and to install the result.  The blocking GPU download runs with
    /// **no lock held**, preventing stalls on unrelated readers/writers.
    #[cfg(feature = "gpu")]
    fn ensure_cpu(&self) {
        // Fast path: read lock to check if already good.
        {
            let guard = self.inner.data.read();
            match &*guard {
                StorageData::Cpu(_) => return,
                StorageData::Both {
                    dirty: DirtySide::Clean | DirtySide::Cpu,
                    ..
                } => return,
                _ => {}
            }
        }

        // Slow path: extract state under write lock, then transfer unlocked.
        let extracted = {
            let mut guard = self.inner.data.write();
            // Re-check after acquiring write lock (another thread may have
            // completed the transfer between our read and write lock).
            match &*guard {
                StorageData::Cpu(_) => return,
                StorageData::Both {
                    dirty: DirtySide::Clean | DirtySide::Cpu,
                    ..
                } => return,
                _ => {}
            }
            // Swap to Transferring — any concurrent accessor will panic with
            // a clear message rather than seeing invalid state.
            std::mem::replace(&mut *guard, StorageData::Transferring)
        };
        // Write lock dropped here — blocking IO runs lock-free.

        let ctx = self.device_ctx();

        let new_state = match extracted {
            StorageData::Gpu { buffer, len } => {
                let cpu_data = ctx.download(&buffer, len);
                StorageData::Both {
                    cpu: cpu_data,
                    gpu: buffer,
                    dirty: DirtySide::Clean,
                }
            }
            StorageData::Both {
                gpu,
                dirty: DirtySide::Gpu,
                ..
            } => {
                let len = self.inner.len;
                let cpu_data = ctx.download(&gpu, len);
                StorageData::Both {
                    cpu: cpu_data,
                    gpu,
                    dirty: DirtySide::Clean,
                }
            }
            other => other,
        };

        // Re-acquire write lock and install the result.
        *self.inner.data.write() = new_state;
    }

    /// Ensure the GPU copy exists and is up-to-date.
    ///
    /// - `Gpu` / `Both { dirty != Cpu }` → no-op.
    /// - `Cpu` → upload CPU→GPU, transition to `Both { dirty: Clean }`.
    /// - `Both { dirty: Cpu }` → re-upload, set `Clean`.
    ///
    /// # Lock protocol
    ///
    /// Same as [`ensure_cpu`] — write lock held only for state swap, not
    /// during the blocking upload.
    #[cfg(feature = "gpu")]
    pub fn ensure_gpu(&self) {
        {
            let guard = self.inner.data.read();
            match &*guard {
                StorageData::Gpu { .. } => return,
                StorageData::Both {
                    dirty: DirtySide::Clean | DirtySide::Gpu,
                    ..
                } => return,
                _ => {}
            }
        }

        let extracted = {
            let mut guard = self.inner.data.write();
            match &*guard {
                StorageData::Gpu { .. } => return,
                StorageData::Both {
                    dirty: DirtySide::Clean | DirtySide::Gpu,
                    ..
                } => return,
                _ => {}
            }
            std::mem::replace(&mut *guard, StorageData::Transferring)
        };

        let ctx = self.device_ctx();

        let new_state = match extracted {
            StorageData::Cpu(cpu_data) => {
                let buffer = ctx.upload(&cpu_data);
                StorageData::Both {
                    cpu: cpu_data,
                    gpu: buffer,
                    dirty: DirtySide::Clean,
                }
            }
            StorageData::Both {
                cpu,
                gpu,
                dirty: DirtySide::Cpu,
            } => {
                ctx.queue.write_buffer(&gpu, 0, bytemuck::cast_slice(&cpu));
                StorageData::Both {
                    cpu,
                    gpu,
                    dirty: DirtySide::Clean,
                }
            }
            other => other,
        };

        *self.inner.data.write() = new_state;
    }
}

// --- WeakStorageHandle ------------------------------------------------------

/// Non-owning handle to a [`StorageInner`].
///
/// Used by the autograd engine's [`crate::autograd::VersionSnapshot`] to
/// check version counters without keeping intermediate tensor memory alive.
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
    pub fn upgrade(&self) -> Option<StorageHandle> {
        self.inner.upgrade().map(|inner| StorageHandle { inner })
    }
}

// Send + Sync compile-time assertions.
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl Layout {
    pub fn contiguous(shape: Vec<usize>) -> Self {
        let ndim = shape.len();
        let mut strides = vec![0usize; ndim];
        if ndim > 0 {
            strides[ndim - 1] = 1;
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        Self { shape, strides, offset: 0 }
    }

    pub fn ndim(&self) -> usize { self.shape.len() }
    pub fn numel(&self) -> usize { self.shape.iter().product() }
    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn strides(&self) -> &[usize] { &self.strides }
    pub fn offset(&self) -> usize { self.offset }

    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() { return true; }
        let mut expected = 1usize;
        for i in (0..self.ndim()).rev() {
            if self.strides[i] != expected { return false; }
            expected *= self.shape[i];
        }
        true
    }

    pub fn transposed(&self, dim0: usize, dim1: usize) -> Self {
        assert!(
            dim0 < self.ndim() && dim1 < self.ndim(),
            "transpose dims ({}, {}) out of range for {}-D tensor",
            dim0, dim1, self.ndim(),
        );
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.swap(dim0, dim1);
        strides.swap(dim0, dim1);
        Self { shape, strides, offset: self.offset }
    }

    pub fn reshaped(&self, new_shape: Vec<usize>) -> Option<Self> {
        let old_numel: usize = self.shape.iter().product();
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(
            old_numel, new_numel,
            "cannot reshape {} elements into shape {:?} ({} elements)",
            old_numel, new_shape, new_numel,
        );
        if !self.is_contiguous() { return None; }
        Some(Self::contiguous_with_offset(new_shape, self.offset))
    }

    fn contiguous_with_offset(shape: Vec<usize>, offset: usize) -> Self {
        let ndim = shape.len();
        let mut strides = vec![0usize; ndim];
        if ndim > 0 {
            strides[ndim - 1] = 1;
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        Self { shape, strides, offset }
    }
}

// ---------------------------------------------------------------------------
// Autograd layer
// ---------------------------------------------------------------------------

/// Gradient-tracking metadata.  Wrapped in `Arc` inside
/// [`AutogradState::Tracked`] so cloned tensors share edge counts.
pub struct TensorMeta {
    pub requires_grad: bool,
    pub grad_id: Option<GradId>,
    pub creator: Option<OpId>,
    pub is_leaf: bool,
    pub retains_grad: bool,
    pub total_grads: AtomicUsize,
}

impl TensorMeta {
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
#[derive(Debug, Clone)]
pub enum AutogradState {
    None,
    Tracked(Arc<TensorMeta>),
}

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------

/// The top-level tensor type composing storage, layout, and autograd state.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub(crate) storage: StorageHandle,
    pub(crate) layout: Layout,
    pub(crate) state: AutogradState,
}

impl Tensor {
    /// Create a new contiguous, inference-mode tensor from raw data.
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
            shape, numel, data.len(),
        );
        let layout = Layout::contiguous(shape);
        Self {
            storage: StorageHandle::new(data),
            layout,
            state: AutogradState::None,
        }
    }

    pub fn shape(&self) -> &[usize] { self.layout.shape() }
    pub fn strides(&self) -> &[usize] { self.layout.strides() }
    pub fn ndim(&self) -> usize { self.layout.ndim() }
    pub fn numel(&self) -> usize { self.layout.numel() }
    pub fn is_contiguous(&self) -> bool { self.layout.is_contiguous() }
    pub fn dtype(&self) -> DType { self.storage.dtype() }

    pub fn requires_grad(&self) -> bool {
        match &self.state {
            AutogradState::None => false,
            AutogradState::Tracked(meta) => meta.requires_grad,
        }
    }

    pub fn version(&self) -> usize { self.storage.version() }

    /// Acquire a read lock on the underlying data.
    ///
    /// Returns a [`DataReadGuard`] that derefs to `&[f32]`.
    pub fn data(&self) -> DataReadGuard<'_> {
        self.storage.data()
    }

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
            _ => {}
        }
    }

    pub(crate) fn meta(&self) -> Option<&Arc<TensorMeta>> {
        match &self.state {
            AutogradState::Tracked(meta) => Some(meta),
            AutogradState::None => None,
        }
    }

    pub fn grad_id(&self) -> Option<GradId> {
        match &self.state {
            AutogradState::Tracked(meta) => meta.grad_id,
            AutogradState::None => None,
        }
    }

    pub(crate) fn from_storage_and_layout(
        storage: StorageHandle,
        layout: Layout,
    ) -> Tensor {
        Tensor { storage, layout, state: AutogradState::None }
    }

    /// Ensure this tensor's data is on the GPU.
    ///
    /// Triggers an H2D transfer if currently CPU-only.  No-op if the
    /// data is already GPU-resident.
    #[cfg(feature = "gpu")]
    pub fn to_gpu(&self) {
        self.storage.ensure_gpu();
    }

    /// Move this tensor to a specific GPU device.
    ///
    /// - Same device → clone (zero-copy).
    /// - CPU → target device: upload.
    /// - Device A → Device B: download to CPU, upload to B (WebGPU has
    ///   no peer-to-peer DMA).
    ///
    /// Returns a new untracked tensor on the target device.
    #[cfg(feature = "multi_gpu")]
    pub fn to_device(&self, target_device: usize) -> Tensor {
        use crate::backend::gpu::context::{MultiGpuContext, STORAGE_USAGE};

        if self.storage.device_index() == target_device {
            if self.storage.is_gpu() {
                return self.clone();
            }
        }

        let mgpu = MultiGpuContext::get().expect("MultiGpuContext required for to_device");
        assert!(target_device < mgpu.num_devices(), "device index out of range");
        let target_ctx = mgpu.device(target_device);

        // Get f32 data on CPU (triggers D2H if needed).
        let guard = self.storage.data();
        let cpu_data: &[f32] = &guard;

        // Upload to target device.
        let byte_size = self.dtype().gpu_buf_size(self.numel());
        let buffer = target_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("multi_gpu_transfer"),
            size: byte_size,
            usage: STORAGE_USAGE,
            mapped_at_creation: false,
        });
        target_ctx.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(cpu_data));
        drop(guard);

        let mut storage = StorageHandle::new_gpu(buffer, self.numel());
        storage.set_device_index(target_device);

        Tensor {
            storage,
            layout: crate::tensor::Layout::contiguous(self.shape().to_vec()),
            state: AutogradState::None,
        }
    }

    /// Returns the GPU device index this tensor lives on.
    pub fn device_index(&self) -> usize {
        self.storage.device_index()
    }
}
