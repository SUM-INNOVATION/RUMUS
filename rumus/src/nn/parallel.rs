// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Data-parallel multi-GPU wrapper and gradient synchronization.
//!
//! `DataParallel<M>` scatters input batches across GPUs, runs forward
//! passes concurrently via `std::thread::scope`, and gathers outputs.
//! `AllReduceSync` averages gradients across devices via CPU staging.

use std::collections::HashMap;

use crate::autograd::GradientStore;
use crate::backend::gpu::context::MultiGpuContext;
use crate::nn::{Module, Parameter};
use crate::tensor::{self, Tensor};

// ---------------------------------------------------------------------------
// DataParallel
// ---------------------------------------------------------------------------

/// Wraps a `Module` for data-parallel execution across multiple GPUs.
///
/// On construction, master weights (device 0) are broadcast to all replicas.
/// `forward()` scatters the input batch, runs the closure concurrently on
/// each device, and gathers the outputs back to device 0.
pub struct DataParallel<M: Module> {
    pub module: M,
    pub device_ids: Vec<usize>,
    /// Per-device copies of the state dict (device 0 = master, not duplicated).
    replicas: Vec<HashMap<String, Tensor>>,
}

impl<M: Module> DataParallel<M> {
    /// Create a new DataParallel wrapper.
    ///
    /// Broadcasts the module's weights from device 0 to all specified devices.
    pub fn new(module: M, device_ids: Vec<usize>) -> Self {
        assert!(!device_ids.is_empty(), "DataParallel: need at least one device");
        let mgpu = MultiGpuContext::get().expect("MultiGpuContext required for DataParallel");
        for &id in &device_ids {
            assert!(id < mgpu.num_devices(), "device {} out of range", id);
        }

        let master_dict = module.state_dict("");
        let mut replicas = Vec::with_capacity(device_ids.len());

        // Device 0's weights are the master — store directly.
        replicas.push(master_dict.clone());

        // Broadcast to other devices.
        for &dev in &device_ids[1..] {
            let mut replica_dict = HashMap::new();
            for (name, tensor) in &master_dict {
                replica_dict.insert(name.clone(), tensor.to_device(dev));
            }
            replicas.push(replica_dict);
        }

        Self { module, device_ids, replicas }
    }

    /// Scatter the input, run forward on all devices, gather the output.
    ///
    /// The `fwd` closure receives a state dict reference and the input chunk,
    /// and must return the model output.  It runs on a separate thread per
    /// device.
    pub fn forward<F>(&self, input: &Tensor, fwd: F) -> Tensor
    where
        F: Fn(&HashMap<String, Tensor>, &Tensor) -> Tensor + Send + Sync,
    {
        let n = self.device_ids.len();
        let batch = input.shape()[0];
        assert!(batch % n == 0, "DataParallel: batch {} not divisible by {} devices", batch, n);
        let chunk_size = batch / n;

        // Scatter: split input along dim 0 and move chunks to each device.
        let chunks: Vec<Tensor> = (0..n)
            .map(|i| {
                input
                    .slice_range(0, i * chunk_size, (i + 1) * chunk_size)
                    .to_device(self.device_ids[i])
            })
            .collect();

        // Forward: run concurrently on all devices.
        let outputs: Vec<Tensor> = std::thread::scope(|s| {
            let handles: Vec<_> = chunks
                .iter()
                .enumerate()
                .map(|(i, chunk)| {
                    let replica = &self.replicas[i];
                    let fwd_ref = &fwd;
                    s.spawn(move || fwd_ref(replica, chunk))
                })
                .collect();

            handles
                .into_iter()
                .map(|h| h.join().expect("DataParallel: forward thread panicked"))
                .collect()
        });

        // Gather: move all outputs to device 0 and concatenate.
        let gathered: Vec<Tensor> = outputs
            .into_iter()
            .map(|t| t.to_device(self.device_ids[0]))
            .collect();

        tensor::cat(&gathered, 0)
    }

    /// Re-broadcast master weights (device 0) to all replicas.
    ///
    /// Call this after `optimizer.step()` to synchronize updated weights.
    pub fn broadcast_weights(&mut self) {
        let master_dict = self.module.state_dict("");

        for (i, &dev) in self.device_ids[1..].iter().enumerate() {
            for (name, tensor) in &master_dict {
                self.replicas[i + 1].insert(name.clone(), tensor.to_device(dev));
            }
        }
        // Update device 0 replica too.
        self.replicas[0] = master_dict;
    }
}

// ---------------------------------------------------------------------------
// AllReduceSync
// ---------------------------------------------------------------------------

/// CPU-staged gradient averaging across multiple devices.
///
/// Downloads gradients from all devices, averages element-wise on the CPU,
/// and returns a single `GradientStore` with the averaged gradients.
pub struct AllReduceSync {
    device_ids: Vec<usize>,
}

impl AllReduceSync {
    pub fn new(device_ids: Vec<usize>) -> Self {
        Self { device_ids }
    }

    /// Average gradients across all devices using the proper WebGPU
    /// async map lifecycle.
    ///
    /// 4-phase pipeline:
    ///   1. Submit: copy gradient buffers → staging buffers on all devices.
    ///   2. Map: request async mapping of all staging buffers.
    ///   3. Poll: `device.poll(Wait)` per device — forces GPU copies to complete.
    ///   4. Average: read mapped views, compute element-wise mean, build result.
    pub fn all_reduce(
        &self,
        grad_stores: &mut [GradientStore],
        params: &[Parameter],
    ) -> GradientStore {
        let mgpu = MultiGpuContext::get().expect("MultiGpuContext required for AllReduce");
        let n = self.device_ids.len() as f32;
        let mut averaged = GradientStore::new();

        // For each parameter, collect gradients from all devices.
        for param in params {
            let gid = param.grad_id();
            let numel = param.tensor.numel();
            let byte_size = (numel * 4) as u64; // F32 gradients

            // Phase 1 + 2: Submit copies and request mappings for all devices.
            let mut staging_buffers = Vec::with_capacity(self.device_ids.len());
            let mut map_receivers = Vec::with_capacity(self.device_ids.len());

            for (store_idx, &dev_id) in self.device_ids.iter().enumerate() {
                let ctx = mgpu.device(dev_id);

                if let Some(grad) = grad_stores[store_idx].get(gid) {
                    // Ensure the gradient is on the GPU for this device.
                    grad.storage.ensure_gpu();
                    let grad_buf = grad.storage.gpu_buffer();

                    // Phase 1: create staging buffer + encode copy.
                    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("allreduce_staging"),
                        size: byte_size,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let mut encoder = ctx.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor::default(),
                    );
                    encoder.copy_buffer_to_buffer(&grad_buf, 0, &staging, 0, byte_size);
                    ctx.queue.submit(std::iter::once(encoder.finish()));
                    drop(grad_buf);

                    // Phase 2: request async mapping.
                    let (tx, rx) = std::sync::mpsc::sync_channel(1);
                    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                        let _ = tx.send(r);
                    });

                    staging_buffers.push(Some(staging));
                    map_receivers.push(Some(rx));
                } else {
                    staging_buffers.push(None);
                    map_receivers.push(None);
                }
            }

            // Phase 3: poll all devices to force copies to complete.
            for &dev_id in &self.device_ids {
                mgpu.device(dev_id).device.poll(wgpu::Maintain::Wait);
            }

            // Phase 4: read mapped views and compute element-wise average.
            let mut sum = vec![0.0f32; numel];
            let mut count = 0usize;

            for (staging, rx) in staging_buffers.iter().zip(map_receivers.iter()) {
                if let (Some(staging_buf), Some(receiver)) = (staging, rx) {
                    // Wait for the map callback.
                    receiver
                        .recv()
                        .expect("map callback not called")
                        .expect("buffer map failed");

                    let view = staging_buf.slice(..).get_mapped_range();
                    let f32_data: &[f32] = bytemuck::cast_slice(&view);
                    for (i, &v) in f32_data.iter().enumerate() {
                        if i < numel {
                            sum[i] += v;
                        }
                    }
                    drop(view);
                    staging_buf.unmap();
                    count += 1;
                }
            }

            // Average.
            if count > 0 {
                let divisor = count as f32;
                for v in &mut sum {
                    *v /= divisor;
                }
            }

            averaged
                .accumulate(gid, Tensor::new(sum, param.tensor.shape().to_vec()))
                .expect("AllReduce: accumulate failed");
        }

        averaged
    }
}
