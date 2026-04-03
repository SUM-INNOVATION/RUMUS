//! Backward operation structs and the version-checking snapshot.
//!
//! Each struct captures the minimal data needed to compute gradients for
//! its corresponding forward op.  No opaque closures — every backward op
//! is a concrete, inspectable type that is `Send + Sync` by construction.

use crate::autograd::AutogradError;
use crate::tensor::{GradId, Layout, StorageHandle, WeakStorageHandle};

// ---------------------------------------------------------------------------
// VersionSnapshot — weak-reference version checker
// ---------------------------------------------------------------------------

/// Snapshot of a [`StorageHandle`]'s version counter at tape-record time.
///
/// Holds a [`WeakStorageHandle`] so recording does **not** keep intermediate
/// tensor memory alive.
///
/// - **Upgrade succeeds:** compare live version vs recorded.  Mismatch →
///   [`AutogradError::VersionMismatch`].
/// - **Upgrade fails:** dead tensor → provably unmutated → `Ok(())`.
#[derive(Debug, Clone)]
pub struct VersionSnapshot {
    pub grad_id: GradId,
    pub weak_storage: WeakStorageHandle,
    pub recorded_version: usize,
}

impl VersionSnapshot {
    pub fn new(grad_id: GradId, storage: &StorageHandle) -> Self {
        Self {
            grad_id,
            recorded_version: storage.version(),
            weak_storage: storage.downgrade(),
        }
    }

    pub fn check(&self) -> Result<(), AutogradError> {
        match self.weak_storage.upgrade() {
            Some(strong) => {
                let current = strong.version();
                if current != self.recorded_version {
                    Err(AutogradError::VersionMismatch {
                        grad_id: self.grad_id,
                        expected: self.recorded_version,
                        found: current,
                    })
                } else {
                    Ok(())
                }
            }
            None => Ok(()),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-op backward structs
// ---------------------------------------------------------------------------

/// Backward for `c = a + b`.
///
/// `∂L/∂a = ∂L/∂c`,  `∂L/∂b = ∂L/∂c`  (identity).
#[derive(Debug)]
pub struct AddBackward {
    pub lhs_version: VersionSnapshot,
    pub rhs_version: VersionSnapshot,
}

/// Backward for `c = a - b`.
///
/// `∂L/∂a = ∂L/∂c`,  `∂L/∂b = -∂L/∂c`.
#[derive(Debug)]
pub struct SubBackward {
    pub lhs_version: VersionSnapshot,
    pub rhs_version: VersionSnapshot,
}

/// Backward for `c = a * b` (element-wise).
///
/// `∂L/∂a = ∂L/∂c ⊙ b`,  `∂L/∂b = ∂L/∂c ⊙ a`.
#[derive(Debug)]
pub struct MulBackward {
    pub lhs_storage: StorageHandle,
    pub lhs_layout: Layout,
    pub lhs_version: VersionSnapshot,
    pub rhs_storage: StorageHandle,
    pub rhs_layout: Layout,
    pub rhs_version: VersionSnapshot,
}

/// Backward for `C = A @ B`.
///
/// `∂L/∂A = ∂L/∂C @ Bᵀ`,  `∂L/∂B = Aᵀ @ ∂L/∂C`.
#[derive(Debug)]
pub struct MatmulBackward {
    pub lhs_storage: StorageHandle,
    pub lhs_layout: Layout,
    pub lhs_version: VersionSnapshot,
    pub rhs_storage: StorageHandle,
    pub rhs_layout: Layout,
    pub rhs_version: VersionSnapshot,
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

/// Backward for `y = relu(x)`.
///
/// `∂L/∂x[i] = ∂L/∂y[i]  if x[i] > 0,  else 0`.
#[derive(Debug)]
pub struct ReluBackward {
    pub input_storage: StorageHandle,
    pub input_layout: Layout,
    pub input_version: VersionSnapshot,
}

/// Backward for `loss = mse_loss(pred, target)` (fused).
///
/// `∂L/∂pred[i] = out_grad_scalar * 2 * (pred[i] - target[i]) / N`.
///
/// Only `pred` receives a gradient; `target` is treated as a constant.
#[derive(Debug)]
pub struct MseLossBackward {
    pub pred_storage: StorageHandle,
    pub pred_layout: Layout,
    pub pred_version: VersionSnapshot,
    pub target_storage: StorageHandle,
    pub target_layout: Layout,
    pub target_version: VersionSnapshot,
    pub numel: usize,
}

/// Backward for `y = add_bias(matrix, bias)`.
///
/// `∂L/∂matrix = ∂L/∂y`  (identity, same shape `[m,n]`).
/// `∂L/∂bias = sum_rows(∂L/∂y)`  (reduce `[m,n]` → `[n]`).
#[derive(Debug)]
pub struct AddBiasBackward {
    pub input_version: VersionSnapshot,
    pub bias_version: VersionSnapshot,
    pub m: usize,
    pub n: usize,
}

/// Backward for `slice_batch(input, index)`.
///
/// `∂L/∂input` is a zero tensor matching the original batched input shape,
/// with `∂L/∂output` placed at the `index`-th batch slot.
#[derive(Debug)]
pub struct SliceBatchBackward {
    pub input_version: VersionSnapshot,
    /// Shape of the original batched input (e.g. `[batch, C, H, W]`).
    pub original_shape: Vec<usize>,
    /// Which batch element was sliced.
    pub index: usize,
}

/// Backward for `im2col(input)`.
///
/// `∂L/∂input = col2im(∂L/∂output)`.
#[derive(Debug)]
pub struct Im2ColBackward {
    pub input_version: VersionSnapshot,
    pub c_in: usize,
    pub h: usize,
    pub w: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub out_h: usize,
    pub out_w: usize,
}

/// Backward for `stack([t0, t1, ...], axis=0)`.
///
/// `∂L/∂t_i = slice(∂L/∂output, i)` along axis 0.
#[derive(Debug)]
pub struct StackBackward {
    /// Number of tensors that were stacked.
    pub count: usize,
    /// Shape of each individual tensor (all must match).
    pub each_shape: Vec<usize>,
    /// Version snapshots for each input.
    pub versions: Vec<VersionSnapshot>,
}

/// Backward for `add_channel_bias(src, bias)`.
///
/// `∂L/∂src = ∂L/∂out`  (identity, same shape `[batch*C, spatial]`)
/// `∂L/∂bias = sum over spatial of ∂L/∂out` per channel.
#[derive(Debug)]
pub struct AddChannelBiasBackward {
    pub input_version: VersionSnapshot,
    pub bias_version: VersionSnapshot,
    pub channels: usize,
    pub spatial: usize,
}

/// Backward for `max_pool2d(input)`.
///
/// Scatters `∂L/∂output` to the argmax positions saved during forward.
#[derive(Debug)]
pub struct MaxPool2dBackward {
    pub input_version: VersionSnapshot,
    /// Saved argmax indices (flat spatial offsets stored as f32).
    pub indices_storage: StorageHandle,
    pub indices_layout: Layout,
    pub channels: usize,
    pub h: usize,
    pub w: usize,
    pub out_h: usize,
    pub out_w: usize,
}

/// Backward for `reshape_tracked(input, new_shape)`.
///
/// `∂L/∂input = reshape(∂L/∂output, original_shape)` — zero-copy.
#[derive(Debug)]
pub struct ReshapeBackward {
    pub input_version: VersionSnapshot,
    pub original_shape: Vec<usize>,
}

/// Backward for `flatten(input)`.
///
/// `∂L/∂input = reshape(∂L/∂output, original_shape)` — zero-copy.
#[derive(Debug)]
pub struct FlattenBackward {
    pub input_version: VersionSnapshot,
    pub original_shape: Vec<usize>,
}

/// Backward for `cross_entropy_loss(logits, targets)`.
///
/// The gradient was pre-computed during the forward pass (softmax - one_hot,
/// scaled by 1/B).  Backward simply scales by the incoming `out_grad` scalar.
#[derive(Debug)]
pub struct CrossEntropyBackward {
    pub input_version: VersionSnapshot,
    /// Pre-computed gradient [B, C], saved during forward.
    pub grad_storage: StorageHandle,
    pub grad_layout: Layout,
}

/// Backward for `dropout(input, p)`.
///
/// `∂L/∂input = ∂L/∂output * saved_mask`.
/// Reuses the existing `mul` dispatch (auto CPU/GPU).
#[derive(Debug)]
pub struct DropoutBackward {
    pub input_version: VersionSnapshot,
    pub mask_storage: StorageHandle,
    pub mask_layout: Layout,
}

/// Backward for tracked `transpose(dim0, dim1)`.
/// `grad_input = transpose(grad_output, dim0, dim1)` — reverse the swap.
#[derive(Debug)]
pub struct TransposeBackward {
    pub input_version: VersionSnapshot,
    pub dim0: usize,
    pub dim1: usize,
}

/// Backward for `bmm(A, B)`.
/// `grad_A = bmm(grad_C, B^T)`, `grad_B = bmm(A^T, grad_C)`.
#[derive(Debug)]
pub struct BmmBackward {
    pub lhs_storage: StorageHandle,
    pub lhs_layout: Layout,
    pub lhs_version: VersionSnapshot,
    pub rhs_storage: StorageHandle,
    pub rhs_layout: Layout,
    pub rhs_version: VersionSnapshot,
    pub batch: usize,
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

/// Backward for `softmax(input)`.  Saves **output**.
/// `grad_input = saved * (grad_out - dot)` where `dot = Σ grad_out * saved`.
#[derive(Debug)]
pub struct SoftmaxBackward {
    pub output_storage: StorageHandle,
    pub output_layout: Layout,
    pub input_version: VersionSnapshot,
    pub num_rows: usize,
    pub row_size: usize,
}

/// Backward for `layer_norm`.
///
/// Kernel 1: per-instance grad_input via c1/c2 reductions.
/// Kernel 2: grad_weight = reduce(grad_out * x_hat), grad_bias = reduce(grad_out).
#[derive(Debug)]
pub struct LayerNormBackward {
    pub input_storage: StorageHandle,
    pub input_layout: Layout,
    pub input_version: VersionSnapshot,
    pub weight_storage: StorageHandle,
    pub weight_layout: Layout,
    pub weight_version: VersionSnapshot,
    pub save_storage: StorageHandle,  // [num_instances, 2]: mean + invstd
    pub save_layout: Layout,
    pub num_instances: usize,
    pub norm_size: usize,
}

/// Backward for `embedding(indices)`.
///
/// Sparse scatter: grad_weight[token_id] += grad_output[lookup].
/// CPU-only backward (no f32 atomics in WGSL).
#[derive(Debug)]
pub struct EmbeddingBackward {
    pub input_version: VersionSnapshot,
    pub indices_storage: StorageHandle,
    pub indices_layout: Layout,
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub total_lookups: usize,
}

/// Backward for `sigmoid(input)`.  Saves **output**.
/// `grad = out_grad * saved_out * (1 - saved_out)`
#[derive(Debug)]
pub struct SigmoidBackward {
    pub output_storage: StorageHandle,
    pub output_layout: Layout,
    pub input_version: VersionSnapshot,
}

/// Backward for `tanh(input)`.  Saves **output**.
/// `grad = out_grad * (1 - saved_out^2)`
#[derive(Debug)]
pub struct TanhBackward {
    pub output_storage: StorageHandle,
    pub output_layout: Layout,
    pub input_version: VersionSnapshot,
}

/// Backward for `gelu(input)` (tanh approx).  Saves **input**.
#[derive(Debug)]
pub struct GeluBackward {
    pub input_storage: StorageHandle,
    pub input_layout: Layout,
    pub input_version: VersionSnapshot,
}

/// Backward for `leaky_relu(input, alpha)`.  Saves **input**.
#[derive(Debug)]
pub struct LeakyReluBackward {
    pub input_storage: StorageHandle,
    pub input_layout: Layout,
    pub input_version: VersionSnapshot,
    pub alpha: f32,
}

/// Backward for a broadcasted binary op.
///
/// If an operand was broadcast, its gradient must be summed (reduced)
/// along the broadcast dimensions.
#[derive(Debug)]
pub struct BroadcastAddBackward {
    pub lhs_version: VersionSnapshot,
    pub rhs_version: VersionSnapshot,
    pub lhs_broadcast: Option<crate::tensor::broadcast::BroadcastInfo>,
    pub rhs_broadcast: Option<crate::tensor::broadcast::BroadcastInfo>,
    pub output_shape: Vec<usize>,
}

#[derive(Debug)]
pub struct BroadcastSubBackward {
    pub lhs_version: VersionSnapshot,
    pub rhs_version: VersionSnapshot,
    pub lhs_broadcast: Option<crate::tensor::broadcast::BroadcastInfo>,
    pub rhs_broadcast: Option<crate::tensor::broadcast::BroadcastInfo>,
    pub output_shape: Vec<usize>,
}

#[derive(Debug)]
pub struct BroadcastMulBackward {
    pub lhs_storage: StorageHandle,
    pub lhs_layout: Layout,
    pub lhs_version: VersionSnapshot,
    pub rhs_storage: StorageHandle,
    pub rhs_layout: Layout,
    pub rhs_version: VersionSnapshot,
    pub lhs_broadcast: Option<crate::tensor::broadcast::BroadcastInfo>,
    pub rhs_broadcast: Option<crate::tensor::broadcast::BroadcastInfo>,
    pub output_shape: Vec<usize>,
}

// ---------------------------------------------------------------------------
// BackwardOp enum
// ---------------------------------------------------------------------------

/// Discriminated union of all backward operation types.
///
/// No closures, no trait objects — `Send + Sync` and inspectable.
#[derive(Debug)]
pub enum BackwardOp {
    Add(AddBackward),
    Sub(SubBackward),
    Mul(MulBackward),
    Matmul(MatmulBackward),
    Relu(ReluBackward),
    MseLoss(MseLossBackward),
    AddBias(AddBiasBackward),
    Im2Col(Im2ColBackward),
    Stack(StackBackward),
    AddChannelBias(AddChannelBiasBackward),
    SliceBatch(SliceBatchBackward),
    MaxPool2d(MaxPool2dBackward),
    Flatten(FlattenBackward),
    Reshape(ReshapeBackward),
    Dropout(DropoutBackward),
    CrossEntropy(CrossEntropyBackward),
    Sigmoid(SigmoidBackward),
    Tanh(TanhBackward),
    Gelu(GeluBackward),
    LeakyRelu(LeakyReluBackward),
    Transpose(TransposeBackward),
    Bmm(BmmBackward),
    Softmax(SoftmaxBackward),
    LayerNorm(LayerNormBackward),
    Embedding(EmbeddingBackward),
    BroadcastAdd(BroadcastAddBackward),
    BroadcastSub(BroadcastSubBackward),
    BroadcastMul(BroadcastMulBackward),
}

const _: () = {
    fn _assert_send<T: Send>() {}
    fn _assert_sync<T: Sync>() {}
    fn _assertions() {
        _assert_send::<BackwardOp>();
        _assert_sync::<BackwardOp>();
    }
};
