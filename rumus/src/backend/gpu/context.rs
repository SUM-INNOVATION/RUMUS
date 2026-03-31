//! Process-global GPU device context.
//!
//! Initialized lazily on first access via [`GpuContext::get()`].
//! Uses `OnceLock` — thread-safe, zero overhead after init.

use std::sync::OnceLock;

/// Process-global GPU context holding the WGPU device and queue.
///
/// The GPU device/queue is a process-wide resource (like PyTorch's
/// `torch.cuda` context).  A single `GpuContext` is shared across all
/// threads via `OnceLock`.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

static GPU_CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

impl GpuContext {
    /// Get the global GPU context, initializing it if needed.
    ///
    /// Uses `pollster::block_on` to synchronously request the adapter and
    /// device.  This blocks exactly once on the first call; all subsequent
    /// calls are a no-op atomic load.
    ///
    /// Returns `None` if no compatible GPU adapter is available — never
    /// panics on missing hardware.
    pub fn get() -> Option<&'static GpuContext> {
        GPU_CONTEXT
            .get_or_init(|| {
                let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
                let adapter = pollster::block_on(instance.request_adapter(
                    &wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        ..Default::default()
                    },
                ))?;
                let (device, queue) = pollster::block_on(
                    adapter.request_device(&wgpu::DeviceDescriptor::default(), None),
                )
                .ok()?;
                Some(GpuContext { device, queue })
            })
            .as_ref()
    }

    /// Returns `true` if a compatible GPU is available.
    pub fn is_available() -> bool {
        Self::get().is_some()
    }

    /// Download GPU buffer data to a CPU `Vec<f32>`.
    ///
    /// Blocking: submits a copy command, maps a staging buffer, and polls
    /// the device until the data is ready.
    pub fn download(&self, source: &wgpu::Buffer, len: usize) -> Vec<f32> {
        let byte_size = (len * std::mem::size_of::<f32>()) as u64;

        // Create a staging buffer for readback.
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_readback"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy source → staging.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, byte_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map the staging buffer and block until ready.
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("map callback not called")
            .expect("buffer map failed");

        // Read the data.
        let view = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        staging.unmap();

        result
    }

    /// Upload CPU data to a new GPU buffer.
    ///
    /// The returned buffer has `STORAGE | COPY_SRC | COPY_DST` usage flags,
    /// suitable for compute shader input/output and staging copies.
    pub fn upload(&self, data: &[f32]) -> wgpu::Buffer {
        let byte_size = (data.len() * std::mem::size_of::<f32>()) as u64;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_storage"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(data));
        buffer
    }
}
