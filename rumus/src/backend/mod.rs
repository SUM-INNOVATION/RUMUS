//! Hardware abstraction layer for compute operations.
//!
//! The [`Backend`] trait defines the contract that every compute backend
//! (CPU, WGPU, etc.) must satisfy.  All methods are **associated functions**
//! — no `&self` — because:
//!
//! - `CpuBackend` is a zero-sized type with no state.
//! - Future backends (e.g. `WgpuBackend`) will manage their device/queue
//!   state globally (via `OnceLock` or thread-local), not through an instance
//!   threaded into every tensor operation.

mod cpu;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use cpu::CpuBackend;

/// Hardware-agnostic compute contract.
///
/// Every method receives **contiguous** `&[f32]` inputs and writes into a
/// caller-allocated `&mut [f32]` output.
pub trait Backend {
    // ----- Memory allocation ------------------------------------------------

    /// Allocate a zero-filled buffer of `len` elements.
    fn zeros(len: usize) -> Vec<f32>;

    /// Allocate a buffer where every element is `value`.
    fn full(len: usize, value: f32) -> Vec<f32>;

    // ----- Element-wise math ------------------------------------------------

    /// `dst[i] = lhs[i] + rhs[i]`
    fn add(lhs: &[f32], rhs: &[f32], dst: &mut [f32]);

    /// `dst[i] = lhs[i] - rhs[i]`
    fn sub(lhs: &[f32], rhs: &[f32], dst: &mut [f32]);

    /// `dst[i] = lhs[i] * rhs[i]`
    fn mul(lhs: &[f32], rhs: &[f32], dst: &mut [f32]);

    /// `dst[i] = scalar * src[i]`
    fn scale(src: &[f32], dst: &mut [f32], scalar: f32);

    /// `dst[i] = max(0, src[i])`
    fn relu(src: &[f32], dst: &mut [f32]);

    /// `dst[i] = if input[i] > 0 { out_grad[i] } else { 0 }`
    fn relu_backward(input: &[f32], out_grad: &[f32], dst: &mut [f32]);

    // ----- Matrix math ------------------------------------------------------

    /// Dense matrix multiplication: `C = A @ B`.
    ///
    /// `a` is `(m x k)`, `b` is `(k x n)`, `dst` is `(m x n)` pre-zeroed.
    fn matmul(a: &[f32], b: &[f32], dst: &mut [f32], m: usize, k: usize, n: usize);

    // ----- Broadcast / reduction --------------------------------------------

    /// Add a 1-D bias to each row of a 2-D matrix.
    ///
    /// `matrix` is `(m x n)`, `bias` is `(n,)`, `dst` is `(m x n)`.
    fn add_bias(matrix: &[f32], bias: &[f32], dst: &mut [f32], m: usize, n: usize);

    /// Sum along rows: `dst[j] = sum_i(src[i*n + j])`.
    ///
    /// `src` is `(m x n)`, `dst` is `(n,)` pre-zeroed.
    fn sum_rows(src: &[f32], dst: &mut [f32], m: usize, n: usize);
}
