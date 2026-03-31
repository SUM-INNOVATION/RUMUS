//! Pure-Rust CPU backend — zero external dependencies.

use super::Backend;

/// Zero-sized CPU compute backend.
///
/// All operations run on the calling thread using standard Rust iterators
/// and loops.  No BLAS, no SIMD intrinsics — this is the correctness
/// baseline that every other backend is tested against.
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn zeros(len: usize) -> Vec<f32> {
        vec![0.0; len]
    }

    fn full(len: usize, value: f32) -> Vec<f32> {
        vec![value; len]
    }

    fn add(lhs: &[f32], rhs: &[f32], dst: &mut [f32]) {
        for ((d, &l), &r) in dst.iter_mut().zip(lhs).zip(rhs) {
            *d = l + r;
        }
    }

    fn mul(lhs: &[f32], rhs: &[f32], dst: &mut [f32]) {
        for ((d, &l), &r) in dst.iter_mut().zip(lhs).zip(rhs) {
            *d = l * r;
        }
    }

    /// Naive triple-loop matrix multiply using **ikj** iteration order.
    ///
    /// # Why ikj?
    ///
    /// The canonical `ijk` order computes one element of C at a time:
    ///
    /// ```text
    /// for i in 0..m:          // row of A / C
    ///     for j in 0..n:      // col of B / C
    ///         for k in 0..k:  // reduction axis
    ///             C[i,j] += A[i,k] * B[k,j]   // B access strides by n → cache miss
    /// ```
    ///
    /// The inner loop over `k` reads `B[k, j]` — stepping by `n` elements
    /// each iteration, which thrashes the L1 cache for large `n`.
    ///
    /// The `ikj` order restructures the loops so the **inner loop streams
    /// over contiguous memory** in both `B` and `C`:
    ///
    /// ```text
    /// for i in 0..m:          // row of A / C
    ///     for p in 0..k:      // reduction axis (renamed to avoid shadowing)
    ///         a_ip = A[i*k + p]
    ///         for j in 0..n:  // col of B / C
    ///             C[i*n + j] += a_ip * B[p*n + j]
    /// ```
    ///
    /// - `B[p*n + j]` with `j` varying by 1 → sequential read of row `p`.
    /// - `C[i*n + j]` with `j` varying by 1 → sequential read-modify-write
    ///   of row `i`.
    /// - `a_ip` is loop-invariant in the inner loop → sits in a register.
    ///
    /// This yields ~3-5× speedup over `ijk` on matrices that exceed L1 size,
    /// at zero code complexity cost.
    ///
    /// # Pre-conditions
    ///
    /// - `a.len() >= m * k`
    /// - `b.len() >= k * n`
    /// - `dst.len() >= m * n`
    /// - `dst` is pre-zeroed.
    fn matmul(a: &[f32], b: &[f32], dst: &mut [f32], m: usize, k: usize, n: usize) {
        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            let dst_row = &mut dst[i * n..(i + 1) * n];
            for p in 0..k {
                let a_ip = a_row[p];
                let b_row = &b[p * n..(p + 1) * n];
                // Inner loop iterates over pre-sliced rows of dst and B.
                // The compiler can see that both iterators have identical
                // length `n`, enabling it to elide all bounds checks and
                // auto-vectorise the fused multiply-add.
                for (d, &b_pj) in dst_row.iter_mut().zip(b_row) {
                    *d += a_ip * b_pj;
                }
            }
        }
    }
}
