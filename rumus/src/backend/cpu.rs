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

    fn sub(lhs: &[f32], rhs: &[f32], dst: &mut [f32]) {
        for ((d, &l), &r) in dst.iter_mut().zip(lhs).zip(rhs) {
            *d = l - r;
        }
    }

    fn mul(lhs: &[f32], rhs: &[f32], dst: &mut [f32]) {
        for ((d, &l), &r) in dst.iter_mut().zip(lhs).zip(rhs) {
            *d = l * r;
        }
    }

    fn scale(src: &[f32], dst: &mut [f32], scalar: f32) {
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = scalar * s;
        }
    }

    fn relu(src: &[f32], dst: &mut [f32]) {
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = if s > 0.0 { s } else { 0.0 };
        }
    }

    fn relu_backward(input: &[f32], out_grad: &[f32], dst: &mut [f32]) {
        for i in 0..dst.len() {
            dst[i] = if input[i] > 0.0 { out_grad[i] } else { 0.0 };
        }
    }

    /// ikj-ordered matrix multiply.  See module docs for rationale.
    fn matmul(a: &[f32], b: &[f32], dst: &mut [f32], m: usize, k: usize, n: usize) {
        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            let dst_row = &mut dst[i * n..(i + 1) * n];
            for p in 0..k {
                let a_ip = a_row[p];
                let b_row = &b[p * n..(p + 1) * n];
                for (d, &b_pj) in dst_row.iter_mut().zip(b_row) {
                    *d += a_ip * b_pj;
                }
            }
        }
    }

    fn add_bias(matrix: &[f32], bias: &[f32], dst: &mut [f32], m: usize, n: usize) {
        for i in 0..m {
            let row = &matrix[i * n..(i + 1) * n];
            let dst_row = &mut dst[i * n..(i + 1) * n];
            for (d, (&v, &b)) in dst_row.iter_mut().zip(row.iter().zip(bias)) {
                *d = v + b;
            }
        }
    }

    fn sum_rows(src: &[f32], dst: &mut [f32], m: usize, n: usize) {
        for i in 0..m {
            let row = &src[i * n..(i + 1) * n];
            for (d, &v) in dst.iter_mut().zip(row) {
                *d += v;
            }
        }
    }
}
