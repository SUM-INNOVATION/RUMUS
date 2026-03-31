//! WGPU-based GPU backend.
//!
//! Provides the [`GpuContext`] device singleton and [`BufferPool`] for
//! efficient GPU memory management.  Compute shaders will be added in
//! Milestone 4, Chunk 2.

pub mod context;
pub mod pool;
