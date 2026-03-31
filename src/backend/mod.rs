//! Hardware abstraction layer for compute operations.
//!
//! The [`Backend`] trait defines the contract that every compute backend
//! (CPU, WGPU, etc.) must satisfy.  All methods are **associated functions**
//! — no `&self` — because:
//!
//! - `CpuBackend` is a zero-sized type with no state.
//! - Future backends (e.g. `WgpuBackend`) will manage their device/queue
//!   state globally (via `OnceLock` or thread-local), not through an instance
//!   threaded into every tensor operation.  This keeps the [`crate::tensor::Tensor`]
//!   struct small and lifetime-free.

mod cpu;

pub use cpu::CpuBackend;

/// Hardware-agnostic compute contract.
///
/// Every method receives **contiguous** `&[f32]` inputs and writes into a
/// caller-allocated `&mut [f32]` output.  This "caller allocates" convention:
///
/// - Avoids hidden allocations inside the backend.
/// - Maps cleanly to GPU workflows where the output buffer may already live
///   in device memory.
/// - Lets the tensor layer own all allocation/lifetime decisions.
pub trait Backend {
    // ----- Memory allocation ------------------------------------------------

    /// Allocate a zero-filled buffer of `len` elements.
    fn zeros(len: usize) -> Vec<f32>;

    /// Allocate a buffer where every element is `value`.
    fn full(len: usize, value: f32) -> Vec<f32>;

    // ----- Element-wise math ------------------------------------------------
    // All slice arguments must have equal length.  The backend does NOT check
    // this — the caller (tensor ops layer) is responsible for validation.

    /// Element-wise addition: `dst[i] = lhs[i] + rhs[i]`.
    fn add(lhs: &[f32], rhs: &[f32], dst: &mut [f32]);

    /// Element-wise multiplication: `dst[i] = lhs[i] * rhs[i]`.
    fn mul(lhs: &[f32], rhs: &[f32], dst: &mut [f32]);

    // ----- Matrix math ------------------------------------------------------

    /// Dense matrix multiplication: `C = A @ B`.
    ///
    /// - `a` is row-major `(m × k)`.
    /// - `b` is row-major `(k × n)`.
    /// - `dst` is row-major `(m × n)` and **must be pre-zeroed** by the caller.
    ///
    /// The implementation may assume all three slices are contiguous and
    /// correctly sized; no bounds checking is performed.
    fn matmul(a: &[f32], b: &[f32], dst: &mut [f32], m: usize, k: usize, n: usize);
}
