// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Fully Sharded Data Parallelism (FSDP).
//!
//! Shards model parameters across GPUs along dimension 0.  Each rank
//! stores only 1/N of every parameter.  Weights are re-gathered on-demand
//! for computation and dropped immediately after — true FSDP memory profile.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::autograd::context;
use crate::autograd::{
    BackwardOp, FsdpLinearBackward, FsdpSync, TapeEntry, VersionSnapshot,
};
use crate::backend::gpu::context::MultiGpuContext;
use crate::nn::{Module, Parameter};
use crate::tensor::{AutogradState, Layout, StorageHandle, Tensor, TensorMeta};

// ---------------------------------------------------------------------------
// ShardInfo
// ---------------------------------------------------------------------------

/// Metadata describing how a parameter is sharded across ranks.
#[derive(Debug, Clone)]
pub struct ShardInfo {
    /// Original unsharded parameter shape.
    pub full_shape: Vec<usize>,
    /// Number of rows (dim 0) in each shard.
    pub shard_size: usize,
    /// Start index in the full tensor for this rank's shard.
    pub shard_offset: usize,
}

// ---------------------------------------------------------------------------
// FSDP
// ---------------------------------------------------------------------------

/// Fully Sharded Data Parallelism wrapper.
///
/// Each rank stores only its local parameter shards (1/N of every parameter).
/// Weights are re-gathered from all ranks for computation and dropped
/// immediately after.
pub struct FSDP {
    pub rank: usize,
    pub world_size: usize,
    pub device_index: usize,
    /// This rank's local parameter shards (name → shard tensor).
    pub local_shards: HashMap<String, Parameter>,
    /// Per-parameter sharding metadata.
    pub shard_map: HashMap<String, ShardInfo>,
    /// Shared cross-rank sync barriers (one per layer/parameter pair).
    /// Key = layer name (e.g., "weight").  All ranks share the same Arc.
    pub sync_map: HashMap<String, Arc<FsdpSync>>,
}

impl FSDP {
    /// Create a new FSDP wrapper for the given rank.
    ///
    /// Slices every parameter from `state_dict` along dim 0 and pushes
    /// this rank's shard to `devices[rank]`.  The original full parameters
    /// are not retained.
    pub fn new<M: Module>(model: &M, device_ids: &[usize], rank: usize) -> Self {
        let world_size = device_ids.len();
        assert!(rank < world_size, "FSDP: rank {} >= world_size {}", rank, world_size);
        let _mgpu = MultiGpuContext::get().expect("MultiGpuContext required for FSDP");

        let state_dict = model.state_dict("");
        let device_index = device_ids[rank];

        let mut local_shards = HashMap::new();
        let mut shard_map = HashMap::new();

        for (name, tensor) in &state_dict {
            let full_shape = tensor.shape().to_vec();
            let dim0 = full_shape[0];
            let shard_size = (dim0 + world_size - 1) / world_size;
            let shard_offset = rank * shard_size;
            let shard_end = (shard_offset + shard_size).min(dim0);
            let actual_shard_size = shard_end - shard_offset;

            // Slice this rank's portion.
            let shard_tensor = if shard_offset < dim0 {
                tensor.slice_range(0, shard_offset, shard_end)
            } else {
                // Edge case: rank has no data (more ranks than rows).
                let numel_per_row: usize = full_shape[1..].iter().product();
                Tensor::new(vec![0.0; actual_shard_size * numel_per_row], {
                    let mut s = full_shape.clone();
                    s[0] = 0;
                    s
                })
            };

            // Push to this rank's device.
            let shard_gpu = shard_tensor.to_device(device_index);
            let param = Parameter::new(shard_gpu);

            shard_map.insert(name.clone(), ShardInfo {
                full_shape,
                shard_size: actual_shard_size,
                shard_offset,
            });
            local_shards.insert(name.clone(), param);
        }

        Self {
            rank,
            world_size,
            device_index,
            local_shards,
            shard_map,
            sync_map: HashMap::new(), // populated via set_sync_map
        }
    }

    /// Create a sync map that can be shared across all ranks.
    ///
    /// Call this once, then pass the result to `set_sync_map` on every rank.
    pub fn create_sync_map<M: Module>(model: &M, world_size: usize) -> HashMap<String, Arc<FsdpSync>> {
        let state_dict = model.state_dict("");
        let mut map = HashMap::new();
        for name in state_dict.keys() {
            map.insert(name.clone(), Arc::new(FsdpSync::new(world_size)));
        }
        map
    }

    /// Set the shared sync barriers (must be the same Arc instances across all ranks).
    pub fn set_sync_map(&mut self, sync_map: HashMap<String, Arc<FsdpSync>>) {
        self.sync_map = sync_map;
    }

    /// All-Gather a parameter: reconstruct the full tensor from all ranks' shards.
    ///
    /// Downloads each rank's shard to CPU, concatenates, uploads to this
    /// rank's device.  The returned tensor is temporary — caller must drop
    /// it after computation.
    pub fn all_gather_param(&self, name: &str, all_ranks: &[&FSDP]) -> Tensor {
        let info = &self.shard_map[name];

        // Download all shards to CPU (concurrently).
        let cpu_shards: Vec<Vec<f32>> = std::thread::scope(|s| {
            let handles: Vec<_> = all_ranks.iter().map(|fsdp| {
                s.spawn(|| {
                    let shard = &fsdp.local_shards[name];
                    let guard = shard.tensor.storage.data();
                    guard.to_vec()
                })
            }).collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        // Concatenate on CPU.
        let mut full_data = Vec::new();
        for shard in &cpu_shards {
            full_data.extend_from_slice(shard);
        }

        // Truncate to exact full_shape (handles padding from uneven sharding).
        let full_numel: usize = info.full_shape.iter().product();
        full_data.truncate(full_numel);

        // Upload to this rank's device.
        let full_tensor = Tensor::new(full_data, info.full_shape.clone());
        full_tensor.to_device(self.device_index)
    }

    /// Collect shard StorageHandles from all ranks for a parameter.
    fn collect_shard_storages(name: &str, all_ranks: &[&FSDP]) -> (Vec<StorageHandle>, Vec<Layout>) {
        let mut storages = Vec::with_capacity(all_ranks.len());
        let mut layouts = Vec::with_capacity(all_ranks.len());
        for fsdp in all_ranks {
            let param = &fsdp.local_shards[name];
            storages.push(param.tensor.storage.clone());
            layouts.push(param.tensor.layout.clone());
        }
        (storages, layouts)
    }

    /// Forward pass for a sharded linear layer.
    ///
    /// All-gathers the full weight (and bias), computes `Y = X @ W^T + b`
    /// inside a `no_grad` scope (to prevent the eager tape from capturing
    /// references to the gathered weight), drops the gathered tensors
    /// immediately, and records a custom `FsdpLinearBackward` on the tape.
    pub fn forward_linear(
        &self,
        x: &Tensor,
        weight_name: &str,
        bias_name: Option<&str>,
        all_ranks: &[&FSDP],
    ) -> Tensor {
        let w_info = &self.shard_map[weight_name];

        // All-Gather W and b (temporary — dropped after matmul).
        let full_w = self.all_gather_param(weight_name, all_ranks);
        let full_b = bias_name.map(|bn| self.all_gather_param(bn, all_ranks));

        // Compute Y = X @ W^T + b inside no_grad so the eager matmul/add_bias
        // don't record MatmulBackward/AddBiasBackward that would hold
        // references to full_w and keep it alive in VRAM.
        let y_storage;
        let y_layout;
        let y_shape;
        {
            let _guard = crate::autograd::context::no_grad();
            let w_t = full_w.transpose(0, 1);
            let mut y = x.matmul(&w_t);
            if let Some(ref b) = full_b {
                y = y.add_bias(b);
            }
            y_storage = y.storage.clone();
            y_layout = y.layout.clone();
            y_shape = y.shape().to_vec();
        }

        // Drop gathered tensors NOW — their refcount hits 0.
        drop(full_w);
        drop(full_b);

        // Record our custom FsdpLinearBackward (which re-gathers during backward).
        let has_grad = x.requires_grad()
            || self.local_shards[weight_name].tensor.requires_grad();
        if has_grad && !context::is_no_grad() {
            let out_grad_id = context::next_grad_id();
            let x_gid = x.grad_id().unwrap_or_else(context::next_grad_id);
            let w_shard_gid = self.local_shards[weight_name].grad_id();

            if let Some(m) = x.meta() { m.total_grads.fetch_add(1, Ordering::Relaxed); }
            if let Some(m) = self.local_shards[weight_name].tensor.meta() {
                m.total_grads.fetch_add(1, Ordering::Relaxed);
            }

            let (w_storages, w_layouts) = Self::collect_shard_storages(weight_name, all_ranks);

            let has_bias = bias_name.is_some();
            let (b_storages, b_gid, bias_shard_offset, bias_shard_size) = if let Some(bn) = bias_name {
                let (bs, _bl) = Self::collect_shard_storages(bn, all_ranks);
                let bg = self.local_shards[bn].grad_id();
                if let Some(m) = self.local_shards[bn].tensor.meta() {
                    m.total_grads.fetch_add(1, Ordering::Relaxed);
                }
                let bi = &self.shard_map[bn];
                (bs, Some(bg), bi.shard_offset, bi.shard_size)
            } else {
                (vec![], None, 0, 0)
            };

            let bias_full_shape = bias_name
                .map(|bn| self.shard_map[bn].full_shape.clone())
                .unwrap_or_default();

            let mut inputs = vec![x_gid, w_shard_gid];
            if let Some(bg) = b_gid {
                inputs.push(bg);
            }

            let op_id = context::with_tape(|tape| {
                tape.push(TapeEntry {
                    op: BackwardOp::FsdpLinear(FsdpLinearBackward {
                        input_version: VersionSnapshot::new(x_gid, &x.storage),
                        input_storage: x.storage.clone(),
                        input_layout: x.layout.clone(),
                        weight_shard_storages: w_storages,
                        weight_shard_layouts: w_layouts,
                        full_weight_shape: w_info.full_shape.clone(),
                        shard_size: w_info.shard_size,
                        weight_shard_offset: w_info.shard_offset,
                        rank: self.rank,
                        world_size: self.world_size,
                        device_index: self.device_index,
                        has_bias,
                        bias_shard_storages: b_storages,
                        full_bias_shape: bias_full_shape,
                        bias_shard_offset,
                        bias_shard_size,
                        sync: self.sync_map.get(weight_name)
                            .expect("FSDP: sync_map missing for parameter — call set_sync_map first")
                            .clone(),
                    }),
                    inputs,
                    outputs: vec![out_grad_id],
                })
            });

            return Tensor {
                storage: y_storage,
                layout: y_layout,
                state: AutogradState::Tracked(Arc::new(TensorMeta {
                    requires_grad: true,
                    grad_id: Some(out_grad_id),
                    creator: op_id,
                    is_leaf: false,
                    retains_grad: false,
                    total_grads: AtomicUsize::new(0),
                })),
            };
        }

        Tensor {
            storage: y_storage,
            layout: Layout::contiguous(y_shape),
            state: AutogradState::None,
        }
    }

    /// Return all local shard parameters for the optimizer.
    pub fn parameters(&self) -> Vec<Parameter> {
        self.local_shards.values().cloned().collect()
    }
}
