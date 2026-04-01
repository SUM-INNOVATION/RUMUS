//! Process-global GPU device context with pre-compiled pipeline cache.

use std::sync::OnceLock;

use super::pool::BufferPool;

/// Standard buffer usage for tensors: storage read/write + staging copies.
pub const STORAGE_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

/// Pre-compiled compute pipelines for all RUMUS operations.
///
/// Compiled once at `GpuContext` init time, then immutable.
/// Named struct fields give compile-time guarantees that every op has a
/// pipeline — no `HashMap` lookup that can fail at runtime.
pub struct PipelineCache {
    // Binary element-wise ops (add, sub, mul, relu_backward)
    pub binary_layout: wgpu::BindGroupLayout,
    pub add_pipeline: wgpu::ComputePipeline,
    pub sub_pipeline: wgpu::ComputePipeline,
    pub mul_pipeline: wgpu::ComputePipeline,
    pub relu_bw_pipeline: wgpu::ComputePipeline,

    // Unary element-wise ops (relu, scale)
    pub unary_layout: wgpu::BindGroupLayout,
    pub relu_pipeline: wgpu::ComputePipeline,
    pub scale_pipeline: wgpu::ComputePipeline,

    // Matrix multiply
    pub matmul_layout: wgpu::BindGroupLayout,
    pub matmul_pipeline: wgpu::ComputePipeline,

    // Bias ops (add_bias, sum_rows)
    pub bias_layout: wgpu::BindGroupLayout,
    pub add_bias_pipeline: wgpu::ComputePipeline,
    pub sum_rows_pipeline: wgpu::ComputePipeline,

    // Optimizer ops
    pub sgd_layout: wgpu::BindGroupLayout,
    pub sgd_pipeline: wgpu::ComputePipeline,
    pub adam_layout: wgpu::BindGroupLayout,
    pub adam_pipeline: wgpu::ComputePipeline,

    // Pool ops
    pub pool_layout: wgpu::BindGroupLayout,
    pub max_pool2d_pipeline: wgpu::ComputePipeline,
    pub pool_bw_layout: wgpu::BindGroupLayout,
    pub max_pool2d_bw_pipeline: wgpu::ComputePipeline,

    // Conv ops (im2col, col2im share unary_layout; channel bias uses bias_layout)
    pub im2col_pipeline: wgpu::ComputePipeline,
    pub col2im_pipeline: wgpu::ComputePipeline,
    pub add_channel_bias_pipeline: wgpu::ComputePipeline,
    pub sum_channel_bias_grad_pipeline: wgpu::ComputePipeline,
}

impl PipelineCache {
    fn new(device: &wgpu::Device) -> Self {
        // ---- Bind group layouts ------------------------------------------------

        let binary_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("binary_layout"),
            entries: &[
                bgl_storage(0, true),
                bgl_storage(1, true),
                bgl_storage_rw(2),
                bgl_uniform(3),
            ],
        });

        let unary_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("unary_layout"),
            entries: &[
                bgl_storage(0, true),
                bgl_storage_rw(1),
                bgl_uniform(2),
            ],
        });

        // matmul uses the same shape as binary: 2 read, 1 rw, 1 uniform
        let matmul_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matmul_layout"),
            entries: &[
                bgl_storage(0, true),
                bgl_storage(1, true),
                bgl_storage_rw(2),
                bgl_uniform(3),
            ],
        });

        // bias: matrix(read) + bias(read) + out(rw) + uniform
        let bias_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bias_layout"),
            entries: &[
                bgl_storage(0, true),
                bgl_storage(1, true),
                bgl_storage_rw(2),
                bgl_uniform(3),
            ],
        });

        // ---- Shader modules ----------------------------------------------------

        let ew_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("elementwise"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/elementwise.wgsl").into(),
            ),
        });

        let unary_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("unary"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/unary.wgsl").into(),
            ),
        });

        let mm_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/matmul.wgsl").into(),
            ),
        });

        let bias_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bias"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/bias.wgsl").into(),
            ),
        });

        // Pool forward: input(read) + output(rw) + indices(rw) + uniform
        let pool_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pool_layout"),
            entries: &[
                bgl_storage(0, true),
                bgl_storage_rw(1),
                bgl_storage_rw(2),
                bgl_uniform(3),
            ],
        });

        // Pool backward: out_grad(read) + indices(read) + grad_input(rw) + uniform
        let pool_bw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pool_bw_layout"),
            entries: &[
                bgl_storage(0, true),
                bgl_storage(1, true),
                bgl_storage_rw(2),
                bgl_uniform(3),
            ],
        });

        let optim_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("optim"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/optim.wgsl").into(),
            ),
        });

        let pool_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pool"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/pool.wgsl").into(),
            ),
        });

        let conv_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("conv"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/conv.wgsl").into(),
            ),
        });

        // SGD: grad(read) + vel(rw) + param(rw) + uniform
        let sgd_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sgd_layout"),
            entries: &[
                bgl_storage(0, true),
                bgl_storage_rw(1),
                bgl_storage_rw(2),
                bgl_uniform(3),
            ],
        });

        // Adam: grad(read) + m(rw) + v(rw) + param(rw) + uniform
        let adam_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("adam_layout"),
            entries: &[
                bgl_storage(0, true),
                bgl_storage_rw(1),
                bgl_storage_rw(2),
                bgl_storage_rw(3),
                bgl_uniform(4),
            ],
        });

        // ---- Pipelines ---------------------------------------------------------

        let make_pipeline = |layout: &wgpu::BindGroupLayout,
                             module: &wgpu::ShaderModule,
                             entry: &str,
                             label: &str|
         -> wgpu::ComputePipeline {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[layout],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        // Build all pipelines before moving layouts into Self.
        let add_pipeline = make_pipeline(&binary_layout, &ew_module, "add_kernel", "add");
        let sub_pipeline = make_pipeline(&binary_layout, &ew_module, "sub_kernel", "sub");
        let mul_pipeline = make_pipeline(&binary_layout, &ew_module, "mul_kernel", "mul");
        let relu_bw_pipeline = make_pipeline(&binary_layout, &ew_module, "relu_backward_kernel", "relu_bw");

        let relu_pipeline = make_pipeline(&unary_layout, &unary_module, "relu_kernel", "relu");
        let scale_pipeline = make_pipeline(&unary_layout, &unary_module, "scale_kernel", "scale");
        let im2col_pipeline = make_pipeline(&unary_layout, &conv_module, "im2col_kernel", "im2col");
        let col2im_pipeline = make_pipeline(&unary_layout, &conv_module, "col2im_kernel", "col2im");

        let matmul_pipeline = make_pipeline(&matmul_layout, &mm_module, "matmul_kernel", "matmul");

        let add_bias_pipeline = make_pipeline(&bias_layout, &bias_module, "add_bias_kernel", "add_bias");
        let sum_rows_pipeline = make_pipeline(&bias_layout, &bias_module, "sum_rows_kernel", "sum_rows");
        let add_channel_bias_pipeline = make_pipeline(&bias_layout, &conv_module, "add_channel_bias_kernel", "add_ch_bias");
        let sum_channel_bias_grad_pipeline = make_pipeline(&bias_layout, &conv_module, "sum_channel_bias_grad_kernel", "sum_ch_bias");

        let sgd_pipeline = make_pipeline(&sgd_layout, &optim_module, "sgd_step", "sgd");
        let adam_pipeline = make_pipeline(&adam_layout, &optim_module, "adam_step", "adam");

        let max_pool2d_pipeline = make_pipeline(&pool_layout, &pool_module, "max_pool2d_kernel", "max_pool2d");
        let max_pool2d_bw_pipeline = make_pipeline(&pool_bw_layout, &pool_module, "max_pool2d_backward_kernel", "max_pool2d_bw");

        Self {
            binary_layout, add_pipeline, sub_pipeline, mul_pipeline, relu_bw_pipeline,
            unary_layout, relu_pipeline, scale_pipeline,
            matmul_layout, matmul_pipeline,
            bias_layout, add_bias_pipeline, sum_rows_pipeline,
            sgd_layout, sgd_pipeline,
            adam_layout, adam_pipeline,
            pool_layout, max_pool2d_pipeline,
            pool_bw_layout, max_pool2d_bw_pipeline,
            im2col_pipeline, col2im_pipeline,
            add_channel_bias_pipeline, sum_channel_bias_grad_pipeline,
        }
    }
}

// --- Helper functions for BindGroupLayoutEntry construction ------------------

fn bgl_storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    bgl_storage(binding, false)
}

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// --- GpuContext --------------------------------------------------------------

/// Process-global GPU context holding device, queue, pipeline cache, and
/// buffer pool.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipelines: PipelineCache,
    pub pool: BufferPool,
}

static GPU_CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

impl GpuContext {
    /// Get the global GPU context, initializing it if needed.
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
                let pipelines = PipelineCache::new(&device);
                let pool = BufferPool::new();
                Some(GpuContext {
                    device,
                    queue,
                    pipelines,
                    pool,
                })
            })
            .as_ref()
    }

    /// Returns `true` if a compatible GPU is available.
    pub fn is_available() -> bool {
        Self::get().is_some()
    }

    /// Download GPU buffer data to a CPU `Vec<f32>`.
    pub fn download(&self, source: &wgpu::Buffer, len: usize) -> Vec<f32> {
        let byte_size = (len * std::mem::size_of::<f32>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_readback"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, byte_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("map callback not called")
            .expect("buffer map failed");

        let view = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        staging.unmap();
        result
    }

    /// Upload CPU data to a new GPU buffer.
    pub fn upload(&self, data: &[f32]) -> wgpu::Buffer {
        let byte_size = (data.len() * std::mem::size_of::<f32>()) as u64;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_storage"),
            size: byte_size,
            usage: STORAGE_USAGE,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(data));
        buffer
    }
}
