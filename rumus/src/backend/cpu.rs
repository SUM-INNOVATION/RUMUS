// SPDX-License-Identifier: Apache-2.0 OR MIT
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

    fn sigmoid(src: &[f32], dst: &mut [f32]) {
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = 1.0 / (1.0 + (-s).exp());
        }
    }

    fn sigmoid_backward(saved_out: &[f32], out_grad: &[f32], dst: &mut [f32]) {
        for i in 0..dst.len() {
            let s = saved_out[i];
            dst[i] = out_grad[i] * s * (1.0 - s);
        }
    }

    fn tanh_act(src: &[f32], dst: &mut [f32]) {
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = s.tanh();
        }
    }

    fn tanh_backward(saved_out: &[f32], out_grad: &[f32], dst: &mut [f32]) {
        for i in 0..dst.len() {
            let t = saved_out[i];
            dst[i] = out_grad[i] * (1.0 - t * t);
        }
    }

    fn gelu(src: &[f32], dst: &mut [f32]) {
        let c = (2.0f32 / std::f32::consts::PI).sqrt();
        for (d, &x) in dst.iter_mut().zip(src) {
            let inner = c * (x + 0.044715 * x * x * x);
            *d = 0.5 * x * (1.0 + inner.tanh());
        }
    }

    fn gelu_backward(saved_input: &[f32], out_grad: &[f32], dst: &mut [f32]) {
        let c = (2.0f32 / std::f32::consts::PI).sqrt();
        for i in 0..dst.len() {
            let x = saved_input[i];
            let inner = c * (x + 0.044715 * x * x * x);
            let t = inner.tanh();
            let sech2 = 1.0 - t * t;
            let d_inner = c * (1.0 + 3.0 * 0.044715 * x * x);
            let gelu_prime = 0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner;
            dst[i] = out_grad[i] * gelu_prime;
        }
    }

    fn leaky_relu(src: &[f32], dst: &mut [f32], alpha: f32) {
        for (d, &x) in dst.iter_mut().zip(src) {
            *d = if x > 0.0 { x } else { alpha * x };
        }
    }

    fn leaky_relu_backward(saved_input: &[f32], out_grad: &[f32], dst: &mut [f32], alpha: f32) {
        for i in 0..dst.len() {
            dst[i] = out_grad[i] * if saved_input[i] > 0.0 { 1.0 } else { alpha };
        }
    }

    fn batch_norm_forward(
        input: &[f32], weight: &[f32], bias: &[f32],
        running_mean: &mut [f32], running_var: &mut [f32],
        output: &mut [f32], save: &mut [f32],
        batch: usize, channels: usize, height: usize, width: usize,
        epsilon: f32, momentum: f32, is_training: bool,
    ) {
        let spatial = height * width;
        let n = batch * spatial;
        for c in 0..channels {
            let (mean, var) = if is_training {
                let mut sum = 0.0f32;
                for b in 0..batch {
                    for hw in 0..spatial {
                        sum += input[b * channels * spatial + c * spatial + hw];
                    }
                }
                let mean = sum / n as f32;
                let mut var_sum = 0.0f32;
                for b in 0..batch {
                    for hw in 0..spatial {
                        let d = input[b * channels * spatial + c * spatial + hw] - mean;
                        var_sum += d * d;
                    }
                }
                let var = var_sum / n as f32;
                save[c * 2] = mean;
                save[c * 2 + 1] = 1.0 / (var + epsilon).sqrt();
                running_mean[c] = (1.0 - momentum) * running_mean[c] + momentum * mean;
                running_var[c] = (1.0 - momentum) * running_var[c] + momentum * var;
                (mean, var)
            } else {
                (running_mean[c], running_var[c])
            };
            let invstd = 1.0 / (var + epsilon).sqrt();
            for b in 0..batch {
                for hw in 0..spatial {
                    let flat = b * channels * spatial + c * spatial + hw;
                    let x_hat = (input[flat] - mean) * invstd;
                    output[flat] = weight[c] * x_hat + bias[c];
                }
            }
        }
    }

    fn batch_norm_backward(
        grad_out: &[f32], input: &[f32], weight: &[f32], save: &[f32],
        grad_input: &mut [f32],
        batch: usize, channels: usize, height: usize, width: usize,
    ) {
        let spatial = height * width;
        let n = batch * spatial;
        for c in 0..channels {
            let mean = save[c * 2];
            let invstd = save[c * 2 + 1];
            let gamma = weight[c];
            let mut c1 = 0.0f32;
            let mut c2 = 0.0f32;
            for b in 0..batch {
                for hw in 0..spatial {
                    let flat = b * channels * spatial + c * spatial + hw;
                    let gn = grad_out[flat] * gamma;
                    let xh = (input[flat] - mean) * invstd;
                    c1 += gn;
                    c2 += gn * xh;
                }
            }
            c1 /= n as f32;
            c2 /= n as f32;
            for b in 0..batch {
                for hw in 0..spatial {
                    let flat = b * channels * spatial + c * spatial + hw;
                    let gn = grad_out[flat] * gamma;
                    let xh = (input[flat] - mean) * invstd;
                    grad_input[flat] = invstd * (gn - c1 - xh * c2);
                }
            }
        }
    }

    fn adaptive_avg_pool2d(
        input: &[f32], output: &mut [f32],
        batch: usize, channels: usize, h_in: usize, w_in: usize, h_out: usize, w_out: usize,
    ) {
        let in_spatial = h_in * w_in;
        let out_spatial = h_out * w_out;
        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let hs = oh * h_in / h_out;
                        let he = ((oh + 1) * h_in + h_out - 1) / h_out;
                        let ws = ow * w_in / w_out;
                        let we = ((ow + 1) * w_in + w_out - 1) / w_out;
                        let base = b * channels * in_spatial + c * in_spatial;
                        let count = (he - hs) * (we - ws);
                        let mut sum = 0.0f32;
                        for h in hs..he { for w in ws..we { sum += input[base + h * w_in + w]; } }
                        output[b * channels * out_spatial + c * out_spatial + oh * w_out + ow] = sum / count as f32;
                    }
                }
            }
        }
    }

    fn adaptive_avg_pool2d_backward(
        grad_out: &[f32], grad_in: &mut [f32],
        batch: usize, channels: usize, h_in: usize, w_in: usize, h_out: usize, w_out: usize,
    ) {
        let in_spatial = h_in * w_in;
        let out_spatial = h_out * w_out;
        for b in 0..batch {
            for c in 0..channels {
                for ih in 0..h_in {
                    for iw in 0..w_in {
                        let mut g = 0.0f32;
                        for oh in 0..h_out {
                            let hs = oh * h_in / h_out;
                            let he = ((oh + 1) * h_in + h_out - 1) / h_out;
                            if ih < hs || ih >= he { continue; }
                            for ow in 0..w_out {
                                let ws = ow * w_in / w_out;
                                let we = ((ow + 1) * w_in + w_out - 1) / w_out;
                                if iw < ws || iw >= we { continue; }
                                let count = (he - hs) * (we - ws);
                                g += grad_out[b * channels * out_spatial + c * out_spatial + oh * w_out + ow] / count as f32;
                            }
                        }
                        grad_in[b * channels * in_spatial + c * in_spatial + ih * w_in + iw] = g;
                    }
                }
            }
        }
    }

    fn bmm(a: &[f32], b: &[f32], dst: &mut [f32], batch: usize, m: usize, k: usize, n: usize) {
        for bi in 0..batch {
            let a_off = bi * m * k;
            let b_off = bi * k * n;
            let c_off = bi * m * n;
            for i in 0..m {
                for p in 0..k {
                    let a_ip = a[a_off + i * k + p];
                    for j in 0..n {
                        dst[c_off + i * n + j] += a_ip * b[b_off + p * n + j];
                    }
                }
            }
        }
    }

    fn softmax_forward(input: &[f32], output: &mut [f32], num_rows: usize, row_size: usize) {
        for r in 0..num_rows {
            let base = r * row_size;
            let row = &input[base..base + row_size];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
            for j in 0..row_size {
                output[base + j] = (input[base + j] - max_val).exp() / sum_exp;
            }
        }
    }

    fn softmax_backward(saved_out: &[f32], grad_out: &[f32], grad_in: &mut [f32], num_rows: usize, row_size: usize) {
        for r in 0..num_rows {
            let base = r * row_size;
            let mut dot = 0.0f32;
            for j in 0..row_size {
                dot += grad_out[base + j] * saved_out[base + j];
            }
            for j in 0..row_size {
                grad_in[base + j] = saved_out[base + j] * (grad_out[base + j] - dot);
            }
        }
    }

    fn layer_norm_forward(
        input: &[f32], weight: &[f32], bias: &[f32],
        output: &mut [f32], save: &mut [f32],
        num_instances: usize, norm_size: usize, epsilon: f32,
    ) {
        let d = norm_size;
        for i in 0..num_instances {
            let base = i * d;
            let row = &input[base..base + d];
            let mean: f32 = row.iter().sum::<f32>() / d as f32;
            let var: f32 = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / d as f32;
            let invstd = 1.0 / (var + epsilon).sqrt();
            save[i * 2] = mean;
            save[i * 2 + 1] = invstd;
            for j in 0..d {
                let x_hat = (input[base + j] - mean) * invstd;
                output[base + j] = weight[j] * x_hat + bias[j];
            }
        }
    }

    fn layer_norm_backward(
        grad_out: &[f32], input: &[f32], weight: &[f32],
        save: &[f32], grad_input: &mut [f32],
        num_instances: usize, norm_size: usize,
    ) {
        let d = norm_size;
        for i in 0..num_instances {
            let base = i * d;
            let mean = save[i * 2];
            let invstd = save[i * 2 + 1];
            let mut c1 = 0.0f32;
            let mut c2 = 0.0f32;
            for j in 0..d {
                let grad_norm_j = grad_out[base + j] * weight[j];
                let x_hat_j = (input[base + j] - mean) * invstd;
                c1 += grad_norm_j;
                c2 += grad_norm_j * x_hat_j;
            }
            c1 /= d as f32;
            c2 /= d as f32;
            for j in 0..d {
                let grad_norm_j = grad_out[base + j] * weight[j];
                let x_hat_j = (input[base + j] - mean) * invstd;
                grad_input[base + j] = invstd * (grad_norm_j - c1 - x_hat_j * c2);
            }
        }
    }

    fn embedding_forward(
        indices: &[f32], weight: &[f32], output: &mut [f32],
        total_lookups: usize, embed_dim: usize,
    ) {
        for i in 0..total_lookups {
            let token_id = indices[i] as usize;
            for d in 0..embed_dim {
                output[i * embed_dim + d] = weight[token_id * embed_dim + d];
            }
        }
    }

    fn embedding_backward(
        grad_out: &[f32], indices: &[f32], grad_weight: &mut [f32],
        total_lookups: usize, embed_dim: usize,
    ) {
        for i in 0..total_lookups {
            let token_id = indices[i] as usize;
            for d in 0..embed_dim {
                grad_weight[token_id * embed_dim + d] += grad_out[i * embed_dim + d];
            }
        }
    }

    fn cross_entropy_forward(
        logits: &[f32], targets: &[f32],
        grad: &mut [f32], loss_per_b: &mut [f32],
        batch: usize, num_classes: usize,
    ) {
        let inv_b = 1.0 / batch as f32;
        for b in 0..batch {
            let row = &logits[b * num_classes..(b + 1) * num_classes];
            let max_z = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = row.iter().map(|&z| (z - max_z).exp()).sum();
            let log_sum_exp = max_z + sum_exp.ln();
            let target_class = targets[b] as usize;
            loss_per_b[b] = (-row[target_class] + log_sum_exp) * inv_b;
            for c in 0..num_classes {
                let softmax_c = (row[c] - max_z).exp() / sum_exp;
                let one_hot = if c == target_class { 1.0 } else { 0.0 };
                grad[b * num_classes + c] = (softmax_c - one_hot) * inv_b;
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
