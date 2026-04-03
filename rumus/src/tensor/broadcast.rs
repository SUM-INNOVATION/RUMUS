//! N-dimensional broadcasting utilities (PyTorch/NumPy semantics).

/// Compute the output shape when broadcasting two tensors.
///
/// Returns `None` if the shapes are not broadcastable.
/// Aligns from the right; dimensions must be equal, or one must be 1.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let ndim = a.len().max(b.len());
    let mut result = vec![0usize; ndim];
    for i in 0..ndim {
        let da = if i < ndim - a.len() { 1 } else { a[i - (ndim - a.len())] };
        let db = if i < ndim - b.len() { 1 } else { b[i - (ndim - b.len())] };
        if da == db {
            result[i] = da;
        } else if da == 1 {
            result[i] = db;
        } else if db == 1 {
            result[i] = da;
        } else {
            return None;
        }
    }
    Some(result)
}

/// Compute broadcast strides for an operand relative to the output shape.
///
/// A dimension that was broadcast (size 1 in the operand, > 1 in output)
/// gets stride 0 — all output indices along that dimension map to the
/// same operand element.  Prepended dimensions (missing in the operand)
/// also get stride 0.
pub fn broadcast_strides(operand_shape: &[usize], output_shape: &[usize]) -> Vec<usize> {
    let ndim = output_shape.len();
    let offset = ndim - operand_shape.len();
    let mut strides = vec![0usize; ndim];
    let mut s = 1usize;
    for d in (0..operand_shape.len()).rev() {
        if operand_shape[d] == 1 {
            strides[offset + d] = 0;
        } else {
            strides[offset + d] = s;
            s *= operand_shape[d];
        }
    }
    strides
}

/// Information about which dimensions were broadcast for a given operand.
/// Used by the backward pass to know which axes to reduce.
#[derive(Debug, Clone)]
pub struct BroadcastInfo {
    /// Original shape of the operand before broadcasting.
    pub original_shape: Vec<usize>,
    /// Which dimensions of the output were broadcast (had size 1 or missing
    /// in the operand).
    pub reduced_dims: Vec<usize>,
}

impl BroadcastInfo {
    /// Build broadcast info for an operand relative to the output shape.
    pub fn new(operand_shape: &[usize], output_shape: &[usize]) -> Option<Self> {
        let ndim = output_shape.len();
        let offset = ndim - operand_shape.len();
        let mut reduced_dims = Vec::new();

        // Leading dims (prepended) are always broadcast.
        for d in 0..offset {
            reduced_dims.push(d);
        }
        // Trailing dims: broadcast if operand has size 1 but output > 1.
        for d in 0..operand_shape.len() {
            if operand_shape[d] == 1 && output_shape[offset + d] > 1 {
                reduced_dims.push(offset + d);
            }
        }

        if reduced_dims.is_empty() {
            None // no broadcasting happened
        } else {
            Some(Self {
                original_shape: operand_shape.to_vec(),
                reduced_dims,
            })
        }
    }
}

/// Suffix products for a shape (used by WGSL kernels for index decomposition).
pub fn suffix_products(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut suffix = vec![1usize; ndim];
    for d in (0..ndim.saturating_sub(1)).rev() {
        suffix[d] = suffix[d + 1] * shape[d + 1];
    }
    suffix
}

/// CPU broadcast binary op: applies `op` element-wise with broadcasting.
pub fn cpu_broadcast_binary(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    output_shape: &[usize],
    a_strides: &[usize],
    b_strides: &[usize],
    op: fn(f32, f32) -> f32,
) {
    let numel: usize = output_shape.iter().product();
    let ndim = output_shape.len();
    let suffix = suffix_products(output_shape);

    for i in 0..numel {
        let mut a_idx = 0usize;
        let mut b_idx = 0usize;
        let mut remainder = i;
        for d in 0..ndim {
            let coord = remainder / suffix[d];
            remainder %= suffix[d];
            a_idx += coord * a_strides[d];
            b_idx += coord * b_strides[d];
        }
        out[i] = op(a[a_idx], b[b_idx]);
    }
}

/// CPU reduce_sum: sum `src` along the specified dimensions.
pub fn cpu_reduce_sum(
    src: &[f32],
    dst: &mut [f32],
    input_shape: &[usize],
    reduced_dims: &[usize],
) {
    let ndim = input_shape.len();
    let in_numel: usize = input_shape.iter().product();
    let suffix = suffix_products(input_shape);

    // Compute output shape (reduced dims become 1).
    let mut out_shape = input_shape.to_vec();
    for &d in reduced_dims {
        out_shape[d] = 1;
    }
    let out_suffix = suffix_products(&out_shape);
    let out_numel: usize = out_shape.iter().product();

    // Zero output.
    for v in dst.iter_mut().take(out_numel) {
        *v = 0.0;
    }

    // Accumulate.
    for i in 0..in_numel {
        let mut out_idx = 0usize;
        let mut remainder = i;
        for d in 0..ndim {
            let coord = remainder / suffix[d];
            remainder %= suffix[d];
            if !reduced_dims.contains(&d) {
                out_idx += coord * out_suffix[d];
            }
        }
        dst[out_idx] += src[i];
    }
}
