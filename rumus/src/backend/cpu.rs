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

    fn im2col(
        src: &[f32], dst: &mut [f32],
        c_in: usize, h: usize, w: usize,
        k: usize, stride: usize, pad: usize,
        out_h: usize, out_w: usize,
    ) {
        let num_patches = out_h * out_w;
        for c in 0..c_in {
            for kh in 0..k {
                for kw in 0..k {
                    let row = c * k * k + kh * k + kw;
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let col = oh * out_w + ow;
                            let ih = (oh * stride + kh) as isize - pad as isize;
                            let iw = (ow * stride + kw) as isize - pad as isize;
                            dst[row * num_patches + col] =
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    src[c * h * w + ih as usize * w + iw as usize]
                                } else {
                                    0.0
                                };
                        }
                    }
                }
            }
        }
    }

    fn col2im(
        src: &[f32], dst: &mut [f32],
        c_in: usize, h: usize, w: usize,
        k: usize, stride: usize, pad: usize,
        out_h: usize, out_w: usize,
    ) {
        let num_patches = out_h * out_w;
        for c in 0..c_in {
            for kh in 0..k {
                for kw in 0..k {
                    let row = c * k * k + kh * k + kw;
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let col = oh * out_w + ow;
                            let ih = (oh * stride + kh) as isize - pad as isize;
                            let iw = (ow * stride + kw) as isize - pad as isize;
                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                dst[c * h * w + ih as usize * w + iw as usize]
                                    += src[row * num_patches + col];
                            }
                        }
                    }
                }
            }
        }
    }

    fn add_channel_bias(
        src: &[f32], bias: &[f32], dst: &mut [f32],
        channels: usize, spatial: usize,
    ) {
        for c in 0..channels {
            for s in 0..spatial {
                dst[c * spatial + s] = src[c * spatial + s] + bias[c];
            }
        }
    }

    fn sum_channel_bias_grad(
        src: &[f32], dst: &mut [f32],
        channels: usize, spatial: usize,
    ) {
        for c in 0..channels {
            for s in 0..spatial {
                dst[c] += src[c * spatial + s];
            }
        }
    }

    fn max_pool2d(
        src: &[f32], dst: &mut [f32], indices: &mut [f32],
        channels: usize, h: usize, w: usize,
        k: usize, stride: usize,
        out_h: usize, out_w: usize,
    ) {
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let out_idx = c * out_h * out_w + oh * out_w + ow;
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_flat = 0usize;
                    for kh in 0..k {
                        for kw in 0..k {
                            let ih = oh * stride + kh;
                            let iw = ow * stride + kw;
                            let val = src[c * h * w + ih * w + iw];
                            if val > max_val {
                                max_val = val;
                                max_flat = ih * w + iw;
                            }
                        }
                    }
                    dst[out_idx] = max_val;
                    indices[out_idx] = max_flat as f32;
                }
            }
        }
    }

    fn max_pool2d_backward(
        out_grad: &[f32], indices: &[f32], dst: &mut [f32],
        channels: usize, h: usize, w: usize,
        out_h: usize, out_w: usize,
    ) {
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let out_idx = c * out_h * out_w + oh * out_w + ow;
                    let src_idx = indices[out_idx] as usize;
                    dst[c * h * w + src_idx] += out_grad[out_idx];
                }
            }
        }
    }

    fn dropout(
        src: &[f32], dst: &mut [f32], mask: &mut [f32],
        numel: usize, p: f32, step: u64,
    ) {
        let scale = 1.0 / (1.0 - p);
        let seed = step as u32;
        for i in 0..numel {
            let hash = pcg_hash_cpu(seed ^ (i as u32));
            // Map upper bits to [0, 1) and compare against p.
            let u = (hash >> 8) as f32 / (1u32 << 24) as f32;
            if u < p {
                dst[i] = 0.0;
                mask[i] = 0.0;
            } else {
                dst[i] = src[i] * scale;
                mask[i] = scale;
            }
        }
    }
}

/// PCG-style hash for CPU dropout PRNG.
fn pcg_hash_cpu(input: u32) -> u32 {
    let mut state = input;
    state = state.wrapping_mul(747796405).wrapping_add(2891336453);
    state = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
    (state >> 22) ^ state
}
