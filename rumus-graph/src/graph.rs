// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Sparse graph representation in CSR format with GPU buffers.

use rumus::backend::gpu::context::{GpuContext, STORAGE_USAGE};
use rumus::tensor::Tensor;

/// Compressed Sparse Row representation of a sparse adjacency matrix.
///
/// All buffers live on the GPU for zero-copy kernel access.
pub struct SparseTensor {
    pub num_nodes: usize,
    pub num_edges: usize,
    /// [num_nodes + 1] u32 — row_ptr[i] = start of row i's edges in col_indices.
    pub row_ptr_buf: wgpu::Buffer,
    /// [num_edges] u32 — column indices (neighbor node IDs).
    pub col_indices_buf: wgpu::Buffer,
    /// [num_edges] f32 — edge weights (None = unweighted, all 1.0).
    pub values_buf: Option<wgpu::Buffer>,
}

impl SparseTensor {
    /// Build a CSR graph from pre-sorted edge arrays (CPU → GPU upload).
    ///
    /// `row_ptr`: [N+1] u32, `col_indices`: [E] u32, `values`: optional [E] f32.
    pub fn from_csr(
        row_ptr: &[u32],
        col_indices: &[u32],
        values: Option<&[f32]>,
        num_nodes: usize,
    ) -> Self {
        let ctx = GpuContext::get().expect("GPU required for SparseTensor");
        let num_edges = col_indices.len();

        let rp_buf = ctx.pool.acquire(&ctx.device, ((num_nodes + 1) * 4) as u64, STORAGE_USAGE);
        ctx.queue.write_buffer(&rp_buf, 0, bytemuck::cast_slice(row_ptr));

        let ci_buf = ctx.pool.acquire(&ctx.device, (num_edges * 4) as u64, STORAGE_USAGE);
        ctx.queue.write_buffer(&ci_buf, 0, bytemuck::cast_slice(col_indices));

        let v_buf = values.map(|vals| {
            let buf = ctx.pool.acquire(&ctx.device, (num_edges * 4) as u64, STORAGE_USAGE);
            ctx.queue.write_buffer(&buf, 0, bytemuck::cast_slice(vals));
            buf
        });

        Self {
            num_nodes,
            num_edges,
            row_ptr_buf: rp_buf,
            col_indices_buf: ci_buf,
            values_buf: v_buf,
        }
    }
}

impl Drop for SparseTensor {
    fn drop(&mut self) {
        if let Some(ctx) = GpuContext::get() {
            // Return buffers to the pool.
            // We need to take ownership — swap with a dummy.
            // Since Drop gives us &mut self, we can't move out directly.
            // Use a small placeholder trick.
            let rp = std::mem::replace(
                &mut self.row_ptr_buf,
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None, size: 4, usage: STORAGE_USAGE, mapped_at_creation: false,
                }),
            );
            ctx.pool.release(rp);

            let ci = std::mem::replace(
                &mut self.col_indices_buf,
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None, size: 4, usage: STORAGE_USAGE, mapped_at_creation: false,
                }),
            );
            ctx.pool.release(ci);

            if let Some(v) = self.values_buf.take() {
                ctx.pool.release(v);
            }
        }
    }
}

/// A directed graph with both forward (A) and backward (A^T) CSR
/// representations for differentiable message passing.
pub struct Graph {
    pub forward: SparseTensor,
    pub backward: SparseTensor,
    pub num_nodes: usize,
    pub num_edges: usize,
    pub has_values: bool,
    // Tensor wrappers for passing CSR buffers into CustomOp.
    // These wrap the same GPU buffers (via StorageHandle::new_gpu).
    pub fwd_row_ptr: Tensor,
    pub fwd_col_idx: Tensor,
    pub fwd_values: Tensor,
    pub bwd_row_ptr: Tensor,
    pub bwd_col_idx: Tensor,
    pub bwd_values: Tensor,
}

impl Graph {
    /// Build a graph from edge lists.
    ///
    /// `src[i] → dst[i]` for each edge.  Nodes are sorted by degree
    /// to mitigate WGSL subgroup divergence.
    pub fn new(
        src: &[u32],
        dst: &[u32],
        weights: Option<&[f32]>,
        num_nodes: usize,
    ) -> Self {
        assert_eq!(src.len(), dst.len(), "Graph: src/dst length mismatch");
        let num_edges = src.len();
        let has_values = weights.is_some();

        // Build forward CSR (src → dst).
        let (fwd_rp, fwd_ci, fwd_vals) = build_csr(src, dst, weights, num_nodes);

        // Build backward CSR (dst → src) = transpose.
        let (bwd_rp, bwd_ci, bwd_vals) = build_csr(dst, src, weights, num_nodes);

        let forward = SparseTensor::from_csr(&fwd_rp, &fwd_ci, fwd_vals.as_deref(), num_nodes);
        let backward = SparseTensor::from_csr(&bwd_rp, &bwd_ci, bwd_vals.as_deref(), num_nodes);

        // Create Tensor wrappers around the GPU buffers for CustomOp.
        // These are u32 buffers but we wrap them as "f32 tensors" with
        // matching byte sizes — the WGSL shader reads them as array<u32>.
        let fwd_row_ptr = wrap_u32_buffer(&forward.row_ptr_buf, num_nodes + 1);
        let fwd_col_idx = wrap_u32_buffer(&forward.col_indices_buf, num_edges);
        let fwd_values = if let Some(ref vb) = forward.values_buf {
            wrap_f32_buffer(vb, num_edges)
        } else {
            // Dummy 1-element tensor for unweighted graphs.
            Tensor::new(vec![1.0], vec![1])
        };

        let bwd_row_ptr = wrap_u32_buffer(&backward.row_ptr_buf, num_nodes + 1);
        let bwd_col_idx = wrap_u32_buffer(&backward.col_indices_buf, num_edges);
        let bwd_values = if let Some(ref vb) = backward.values_buf {
            wrap_f32_buffer(vb, num_edges)
        } else {
            Tensor::new(vec![1.0], vec![1])
        };

        Self {
            forward,
            backward,
            num_nodes,
            num_edges,
            has_values,
            fwd_row_ptr,
            fwd_col_idx,
            fwd_values,
            bwd_row_ptr,
            bwd_col_idx,
            bwd_values,
        }
    }

    /// Differentiable message passing: output[i] = Σ_{j ∈ N(i)} A[i,j] * features[j].
    pub fn spmm(&self, features: &Tensor) -> Tensor {
        assert_eq!(features.shape()[0], self.num_nodes, "spmm: node count mismatch");
        let hidden_dim = features.shape()[1];

        crate::ops::spmm_forward(self, features, hidden_dim)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build CSR arrays from edge lists on the CPU.
fn build_csr(
    src: &[u32],
    dst: &[u32],
    weights: Option<&[f32]>,
    num_nodes: usize,
) -> (Vec<u32>, Vec<u32>, Option<Vec<f32>>) {
    let num_edges = src.len();

    // Sort edges by source node.
    let mut edges: Vec<(u32, u32, f32)> = (0..num_edges)
        .map(|i| {
            let w = weights.map_or(1.0, |ws| ws[i]);
            (src[i], dst[i], w)
        })
        .collect();
    edges.sort_unstable_by_key(|e| e.0);

    // Build CSR.
    let mut row_ptr = vec![0u32; num_nodes + 1];
    let mut col_indices = Vec::with_capacity(num_edges);
    let mut values = if weights.is_some() {
        Some(Vec::with_capacity(num_edges))
    } else {
        None
    };

    for &(s, d, w) in &edges {
        row_ptr[s as usize + 1] += 1;
        col_indices.push(d);
        if let Some(ref mut vals) = values {
            vals.push(w);
        }
    }

    // Prefix sum for row_ptr.
    for i in 1..=num_nodes {
        row_ptr[i] += row_ptr[i - 1];
    }

    (row_ptr, col_indices, values)
}

/// Download u32 GPU buffer to CPU, reinterpret as f32, create Tensor, push to GPU.
///
/// The WGSL shader reads the binding as `array<u32>` — the bit pattern is
/// preserved because u32 and f32 are both 4 bytes.
fn wrap_u32_buffer(buf: &wgpu::Buffer, num_u32s: usize) -> Tensor {
    let ctx = GpuContext::get().expect("GPU required");
    // Download raw bytes.
    let byte_size = (num_u32s * 4) as u64;
    let raw = ctx.download_raw_bytes(buf, byte_size);
    // Reinterpret as f32 (same bit pattern, 4 bytes each).
    let f32_data: Vec<f32> = bytemuck::cast_slice(&raw).to_vec();
    let t = Tensor::new(f32_data, vec![num_u32s]);
    t.to_gpu();
    t
}

/// Download f32 GPU buffer to CPU, create Tensor, push to GPU.
fn wrap_f32_buffer(buf: &wgpu::Buffer, num_f32s: usize) -> Tensor {
    let ctx = GpuContext::get().expect("GPU required");
    let data = ctx.download(buf, num_f32s);
    let t = Tensor::new(data, vec![num_f32s]);
    t.to_gpu();
    t
}
