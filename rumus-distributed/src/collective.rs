// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Async collective operations: AllReduce via dedicated comm threads.

use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};

use rumus::tensor::Tensor;

// ---------------------------------------------------------------------------
// CollectiveBarrier — reusable cross-rank synchronization
// ---------------------------------------------------------------------------

/// Cross-rank barrier for summing f32 vectors.
///
/// Shared by TP AllReduce, FSDP Reduce-Scatter, and PP gradient exchange.
pub struct CollectiveBarrier {
    pub world_size: usize,
    state: Mutex<BarrierState>,
    cvar: Condvar,
}

struct BarrierState {
    buffers: Vec<Vec<f32>>,
    result: Option<Vec<f32>>,
    read_count: usize,
}

impl CollectiveBarrier {
    pub fn new(world_size: usize) -> Self {
        Self {
            world_size,
            state: Mutex::new(BarrierState {
                buffers: Vec::new(),
                result: None,
                read_count: 0,
            }),
            cvar: Condvar::new(),
        }
    }

    /// Push local data, wait for all ranks, return the reduced (summed + averaged) result.
    pub fn reduce(&self, local: Vec<f32>) -> Vec<f32> {
        let mut state = self.state.lock().unwrap();

        state.buffers.push(local);

        if state.buffers.len() == self.world_size {
            // Last arrival: sum all buffers.
            let len = state.buffers[0].len();
            let mut summed = vec![0.0f32; len];
            for buf in &state.buffers {
                for (s, &v) in summed.iter_mut().zip(buf.iter()) {
                    *s += v;
                }
            }
            let n = self.world_size as f32;
            for v in &mut summed {
                *v /= n;
            }
            state.result = Some(summed);
            state.read_count = 0;
            self.cvar.notify_all();
        } else {
            state = self.cvar
                .wait_while(state, |s| s.result.is_none())
                .unwrap();
        }

        let result = state.result.as_ref().unwrap().clone();
        state.read_count += 1;
        if state.read_count == self.world_size {
            state.buffers.clear();
            state.result = None;
            state.read_count = 0;
        }

        result
    }
}

// ---------------------------------------------------------------------------
// CommThread — dedicated background thread for async collectives
// ---------------------------------------------------------------------------

/// Request from the compute thread to the comm thread.
pub struct CommRequest {
    pub staging_buf: wgpu::Buffer,
    pub dst_buf: wgpu::Buffer,
    pub byte_size: u64,
    pub barrier: Arc<CollectiveBarrier>,
    pub response_tx: mpsc::SyncSender<()>,
}

/// Dedicated communication thread for async AllReduce.
///
/// Owns `Arc<Device>` and `Arc<Queue>` for its GPU — can `poll` and `write_buffer`.
pub struct CommThread {
    tx: mpsc::SyncSender<CommRequest>,
    _handle: std::thread::JoinHandle<()>,
}

impl CommThread {
    /// Spawn a comm thread for the given device/queue.
    pub fn spawn(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Self {
        let (tx, rx) = mpsc::sync_channel::<CommRequest>(16);

        let handle = std::thread::spawn(move || {
            while let Ok(req) = rx.recv() {
                // Map the staging buffer (blocks this thread only, not compute).
                let slice = req.staging_buf.slice(..);
                let (map_tx, map_rx) = mpsc::sync_channel(1);
                slice.map_async(wgpu::MapMode::Read, move |r| {
                    let _ = map_tx.send(r);
                });
                device.poll(wgpu::Maintain::Wait);
                map_rx.recv().unwrap().unwrap();

                // Read the data.
                let view = slice.get_mapped_range();
                let local: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
                drop(view);
                req.staging_buf.unmap();

                // Barrier: sum with all ranks.
                let reduced = req.barrier.reduce(local);

                // Upload reduced result to the destination buffer.
                queue.write_buffer(&req.dst_buf, 0, bytemuck::cast_slice(&reduced));

                // Signal completion.
                let _ = req.response_tx.send(());
            }
        });

        Self { tx, _handle: handle }
    }

    /// Submit a non-blocking AllReduce request.
    pub fn submit(&self, req: CommRequest) {
        self.tx.send(req).expect("comm thread dead");
    }
}

// ---------------------------------------------------------------------------
// AsyncAllReduce — high-level non-blocking API
// ---------------------------------------------------------------------------

/// Non-blocking AllReduce handle.
pub struct AllReduceHandle {
    rx: mpsc::Receiver<()>,
}

impl AllReduceHandle {
    /// Block until the AllReduce result is available in the destination buffer.
    pub fn wait(self) {
        let _ = self.rx.recv();
    }
}

/// Submit a non-blocking AllReduce via the comm thread.
///
/// 1. Encodes copy from `src_buf` to a staging buffer.
/// 2. Submits the copy command.
/// 3. Sends the staging + dst to the comm thread.
/// 4. Returns a handle the compute thread can `.wait()` on later.
pub fn async_allreduce(
    comm: &CommThread,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src_buf: &wgpu::Buffer,
    dst_buf: wgpu::Buffer,
    byte_size: u64,
    barrier: &Arc<CollectiveBarrier>,
) -> AllReduceHandle {
    // Create staging buffer.
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("allreduce_staging"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Encode + submit the GPU copy.
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(src_buf, 0, &staging, 0, byte_size);
    queue.submit(std::iter::once(enc.finish()));

    // Send to comm thread (non-blocking from compute thread's perspective).
    let (resp_tx, resp_rx) = mpsc::sync_channel(1);
    comm.submit(CommRequest {
        staging_buf: staging,
        dst_buf,
        byte_size,
        barrier: Arc::clone(barrier),
        response_tx: resp_tx,
    });

    AllReduceHandle { rx: resp_rx }
}
