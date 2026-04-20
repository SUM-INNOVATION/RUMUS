// SPDX-License-Identifier: Apache-2.0 OR MIT
//! `rumus-distributed` — 3D parallelism for RUMUS.
//!
//! Tensor Parallelism (ColumnParallel + RowParallel), Pipeline Parallelism
//! (1F1B micro-batch schedule), and async collective operations.

pub mod collective;
pub mod pipeline;
pub mod tensor_parallel;

pub use collective::{AllReduceHandle, CollectiveBarrier, CommThread};
pub use pipeline::{PipelineExecutor, PipelineStage};
pub use tensor_parallel::{ColumnParallelLinear, RowParallelLinear};
