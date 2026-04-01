//! WGPU-based GPU backend.
//!
//! Provides the [`GpuContext`] device singleton, [`BufferPool`] for
//! efficient GPU memory management, and [`compute`] dispatch functions
//! for all tensor operations.

pub mod compute;
pub mod context;
pub mod pool;
