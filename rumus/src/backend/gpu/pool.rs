// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Thread-safe GPU buffer pool.
//!
//! Caches and reuses `wgpu::Buffer` allocations, keyed by size (rounded
//! up to the nearest power of 2) and usage flags.  This eliminates the
//! per-operation GPU allocation overhead in training loops.

use std::collections::HashMap;

use parking_lot::Mutex;

/// Round `size` up to the nearest power of two.
fn round_up_pow2(size: u64) -> u64 {
    if size <= 1 {
        return 1;
    }
    1u64 << (64 - (size - 1).leading_zeros())
}

/// Cache key: rounded size + usage flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PoolKey {
    size: u64,
    usage: wgpu::BufferUsages,
}

/// Thread-safe GPU buffer pool.
///
/// Buffers are returned to the pool via [`release`](BufferPool::release)
/// instead of being dropped.  When a new buffer is requested,
/// [`acquire`](BufferPool::acquire) checks for a cached buffer of
/// matching size and usage before allocating from the device.
pub struct BufferPool {
    cache: Mutex<HashMap<PoolKey, Vec<wgpu::Buffer>>>,
}

impl BufferPool {
    /// Create an empty buffer pool.
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Acquire a buffer of at least `size_bytes` with the given `usage`.
    ///
    /// Returns a cached buffer if available; otherwise allocates a new one.
    /// The actual buffer size may be larger than requested (rounded to
    /// power of 2) to improve cache hit rates.
    pub fn acquire(
        &self,
        device: &wgpu::Device,
        size_bytes: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let rounded = round_up_pow2(size_bytes);
        let key = PoolKey {
            size: rounded,
            usage,
        };

        // Try to reuse a cached buffer.
        if let Some(buf) = self.cache.lock().get_mut(&key).and_then(|v| v.pop()) {
            return buf;
        }

        // Allocate a fresh buffer from the device.
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pooled_buffer"),
            size: rounded,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Return a buffer to the pool for future reuse.
    ///
    /// The buffer's actual size (which may have been rounded up) and
    /// usage flags are used as the cache key.
    pub fn release(&self, buffer: wgpu::Buffer) {
        let key = PoolKey {
            size: buffer.size(),
            usage: buffer.usage(),
        };
        self.cache.lock().entry(key).or_default().push(buffer);
    }

    /// Drop all cached buffers, freeing GPU memory.
    pub fn clear(&self) {
        self.cache.lock().clear();
    }

    /// Number of buffers currently cached (across all size buckets).
    ///
    /// Used for leak detection: after a training loop stabilizes, this
    /// should be bounded (one per unique size bucket, not growing with
    /// epoch count).
    pub fn cached_count(&self) -> usize {
        self.cache.lock().values().map(|v| v.len()).sum()
    }
}
