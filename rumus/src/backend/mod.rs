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

    // ----- Convolution helpers ------------------------------------------------

    /// im2col: extract patches from `[C_in, H, W]` into `[C_in*K*K, out_h*out_w]`.
    fn im2col(
        src: &[f32], dst: &mut [f32],
        c_in: usize, h: usize, w: usize,
        k: usize, stride: usize, pad: usize,
        out_h: usize, out_w: usize,
    );

    /// col2im: scatter columns back to image, accumulating overlaps.
    /// `dst` must be pre-zeroed.
    fn col2im(
        src: &[f32], dst: &mut [f32],
        c_in: usize, h: usize, w: usize,
        k: usize, stride: usize, pad: usize,
        out_h: usize, out_w: usize,
    );

    /// Add channel bias: `dst[c*spatial+s] = src[c*spatial+s] + bias[c]`.
    fn add_channel_bias(
        src: &[f32], bias: &[f32], dst: &mut [f32],
        channels: usize, spatial: usize,
    );

    /// Sum over spatial for each channel: `dst[c] = sum_s(src[c*spatial+s])`.
    /// `dst` must be pre-zeroed.
    fn sum_channel_bias_grad(
        src: &[f32], dst: &mut [f32],
        channels: usize, spatial: usize,
    );

    // ----- Pooling ------------------------------------------------------------

    /// Max pool 2D forward.  Operates on one batch element: `[C, H, W]`.
    /// Writes pooled values to `dst` and argmax flat indices (as f32) to `indices`.
    fn max_pool2d(
        src: &[f32], dst: &mut [f32], indices: &mut [f32],
        channels: usize, h: usize, w: usize,
        k: usize, stride: usize,
        out_h: usize, out_w: usize,
    );

    /// Max pool 2D backward.  Scatters `out_grad` to positions recorded in
    /// `indices`.  `dst` must be pre-zeroed.  Safe when `stride >= k`.
    fn max_pool2d_backward(
        out_grad: &[f32], indices: &[f32], dst: &mut [f32],
        channels: usize, h: usize, w: usize,
        out_h: usize, out_w: usize,
    );

    // ----- Dropout -------------------------------------------------------------

    /// Generate a dropout mask and apply it to `src`.
    ///
    /// `mask[i]` is `0.0` (dropped) or `1.0 / (1.0 - p)` (kept, scaled).
    /// `dst[i] = src[i] * mask[i]`.
    ///
    /// `step` is a monotonically increasing counter for PRNG seeding.
    fn dropout(
        src: &[f32], dst: &mut [f32], mask: &mut [f32],
        numel: usize, p: f32, step: u64,
    );

    // ----- Advanced activations ------------------------------------------------

    fn sigmoid(src: &[f32], dst: &mut [f32]);
    fn sigmoid_backward(saved_out: &[f32], out_grad: &[f32], dst: &mut [f32]);
    fn tanh_act(src: &[f32], dst: &mut [f32]);
    fn tanh_backward(saved_out: &[f32], out_grad: &[f32], dst: &mut [f32]);
    fn gelu(src: &[f32], dst: &mut [f32]);
    fn gelu_backward(saved_input: &[f32], out_grad: &[f32], dst: &mut [f32]);
    fn leaky_relu(src: &[f32], dst: &mut [f32], alpha: f32);
    fn leaky_relu_backward(saved_input: &[f32], out_grad: &[f32], dst: &mut [f32], alpha: f32);

    // ----- Batched MatMul ------------------------------------------------------

    fn bmm(a: &[f32], b: &[f32], dst: &mut [f32], batch: usize, m: usize, k: usize, n: usize);

    // ----- Softmax -------------------------------------------------------------

    fn softmax_forward(input: &[f32], output: &mut [f32], num_rows: usize, row_size: usize);
    fn softmax_backward(saved_out: &[f32], grad_out: &[f32], grad_in: &mut [f32], num_rows: usize, row_size: usize);

    // ----- LayerNorm -----------------------------------------------------------

    fn layer_norm_forward(
        input: &[f32], weight: &[f32], bias: &[f32],
        output: &mut [f32], save_mean_invstd: &mut [f32],
        num_instances: usize, norm_size: usize, epsilon: f32,
    );

    fn layer_norm_backward(
        grad_out: &[f32], input: &[f32], weight: &[f32],
        save_mean_invstd: &[f32], grad_input: &mut [f32],
        num_instances: usize, norm_size: usize,
    );

    // ----- Embedding -----------------------------------------------------------

    fn embedding_forward(
        indices: &[f32], weight: &[f32], output: &mut [f32],
        total_lookups: usize, embed_dim: usize,
    );

    fn embedding_backward(
        grad_out: &[f32], indices: &[f32], grad_weight: &mut [f32],
        total_lookups: usize, embed_dim: usize,
    );

    // ----- Loss ----------------------------------------------------------------

    /// Cross-entropy loss forward: computes per-batch loss AND gradient in one pass.
    /// Uses the Log-Sum-Exp trick for numerical stability.
    fn cross_entropy_forward(
        logits: &[f32], targets: &[f32],
        grad: &mut [f32], loss_per_b: &mut [f32],
        batch: usize, num_classes: usize,
    );
}
