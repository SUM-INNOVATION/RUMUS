//! Multithreaded data loading with bounded prefetching.

use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

use crate::data::dataset::{DataItem, Dataset};
use crate::tensor;

/// Batched data loader with optional multithreaded prefetching.
///
/// Yields `DataItem` batches (input stacked along dim 0, target stacked
/// along dim 0).  Worker threads fetch and collate items in the background;
/// bounded channels prevent runaway memory usage.
pub struct DataLoader {
    dataset: Arc<dyn Dataset>,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    num_workers: usize,
    prefetch_factor: usize,
}

impl DataLoader {
    /// Create a new `DataLoader`.
    ///
    /// - `num_workers = 0`: main-thread loading (no spawned threads).
    /// - `prefetch_factor`: max completed batches buffered ahead per worker
    ///   (total channel capacity = `prefetch_factor * num_workers`).
    pub fn new(
        dataset: Arc<dyn Dataset>,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
        num_workers: usize,
        prefetch_factor: usize,
    ) -> Self {
        assert!(batch_size > 0, "DataLoader: batch_size must be > 0");
        assert!(prefetch_factor > 0, "DataLoader: prefetch_factor must be > 0");
        Self {
            dataset,
            batch_size,
            shuffle,
            drop_last,
            num_workers,
            prefetch_factor,
        }
    }

    /// Create an iterator that yields one epoch of batches.
    ///
    /// Each call generates a fresh index permutation (if shuffling) and
    /// spawns worker threads (if `num_workers > 0`).  Threads are joined
    /// when the iterator is dropped.
    pub fn iter(&self) -> DataLoaderIter {
        let n = self.dataset.len();
        let mut indices: Vec<usize> = (0..n).collect();

        if self.shuffle {
            fisher_yates_shuffle(&mut indices);
        }

        // Chunk indices into batch-sized groups.
        let mut batch_groups: Vec<Vec<usize>> = indices
            .chunks(self.batch_size)
            .map(|c| c.to_vec())
            .collect();

        // Drop the last incomplete batch if requested.
        if self.drop_last {
            if let Some(last) = batch_groups.last() {
                if last.len() < self.batch_size {
                    batch_groups.pop();
                }
            }
        }

        if self.num_workers == 0 {
            DataLoaderIter::SingleThread {
                dataset: Arc::clone(&self.dataset),
                batch_groups,
                cursor: 0,
            }
        } else {
            // Bounded channel for index groups: workers pull on demand.
            let (index_tx, index_rx) = mpsc::sync_channel::<Vec<usize>>(self.num_workers);

            // Bounded channel for completed batches: backpressure mechanism.
            // Capacity = prefetch_factor * num_workers so every worker can
            // have `prefetch_factor` batches in-flight before blocking.
            let channel_cap = self.prefetch_factor * self.num_workers;
            let (batch_tx, batch_rx) = mpsc::sync_channel::<DataItem>(channel_cap);

            // Share the index receiver across workers.
            let index_rx = Arc::new(parking_lot::Mutex::new(index_rx));

            let mut handles = Vec::with_capacity(self.num_workers);
            for _ in 0..self.num_workers {
                let ds = Arc::clone(&self.dataset);
                let irx = Arc::clone(&index_rx);
                let btx = batch_tx.clone();

                let handle = thread::spawn(move || {
                    loop {
                        // Pull next batch of indices (blocks if no work available).
                        let batch_indices = {
                            let rx = irx.lock();
                            match rx.recv() {
                                Ok(indices) => indices,
                                Err(_) => break, // channel closed — epoch done
                            }
                        };

                        // Fetch items and collate into a batch.
                        let items: Vec<DataItem> = batch_indices
                            .iter()
                            .map(|&i| ds.get(i))
                            .collect();
                        let batch = collate(items);

                        // Send to consumer. If receiver dropped (early break),
                        // gracefully exit — do NOT panic.
                        if btx.send(batch).is_err() {
                            break;
                        }
                    }
                    // btx drops here → signals we're done sending.
                });
                handles.push(handle);
            }

            // Drop the extra clone — only workers hold senders now.
            drop(batch_tx);

            // Feed all batch groups into the index channel.
            // This runs on a dedicated thread; bounded channel means it may
            // block if workers are slower than submission (natural pacing).
            let feeder = thread::spawn(move || {
                for group in batch_groups {
                    if index_tx.send(group).is_err() {
                        break; // workers gone (early teardown)
                    }
                }
                // index_tx drops here → workers see recv() error → exit
            });

            DataLoaderIter::MultiThread {
                batch_rx: Some(batch_rx),
                feeder: Some(feeder),
                workers: handles,
            }
        }
    }
}

/// Iterator over one epoch of batches.
///
/// Not `Send` — must be consumed on the thread that created it.
///
/// Channels are wrapped in `Option` so that `Drop` can `take()` them
/// before joining threads — preventing deadlocks where a worker is
/// blocked on `send()` while the receiver is still alive.
pub enum DataLoaderIter {
    SingleThread {
        dataset: Arc<dyn Dataset>,
        batch_groups: Vec<Vec<usize>>,
        cursor: usize,
    },
    MultiThread {
        batch_rx: Option<mpsc::Receiver<DataItem>>,
        feeder: Option<thread::JoinHandle<()>>,
        workers: Vec<thread::JoinHandle<()>>,
    },
}

impl Iterator for DataLoaderIter {
    type Item = DataItem;

    fn next(&mut self) -> Option<DataItem> {
        match self {
            DataLoaderIter::SingleThread {
                dataset,
                batch_groups,
                cursor,
            } => {
                if *cursor >= batch_groups.len() {
                    return None;
                }
                let indices = &batch_groups[*cursor];
                *cursor += 1;
                let items: Vec<DataItem> = indices.iter().map(|&i| dataset.get(i)).collect();
                Some(collate(items))
            }
            DataLoaderIter::MultiThread { batch_rx, .. } => {
                batch_rx.as_ref().and_then(|rx| rx.recv().ok())
            }
        }
    }
}

impl Drop for DataLoaderIter {
    fn drop(&mut self) {
        if let DataLoaderIter::MultiThread {
            batch_rx,
            feeder,
            workers,
        } = self
        {
            // 1. Drop the batch receiver FIRST.  Any worker blocked on
            //    `btx.send(batch)` will immediately get `Err(SendError)`
            //    and break out of its loop.  Without this, joining workers
            //    would deadlock.
            drop(batch_rx.take());

            // 2. Join all worker threads.  They are now guaranteed to
            //    terminate: either they finished naturally, or the send
            //    error from step 1 caused them to break.
            for handle in workers.drain(..) {
                let _ = handle.join();
            }

            // 3. Join the feeder thread.  With all workers gone, the
            //    feeder's `index_tx.send()` will error out (no receivers)
            //    and it will break its loop.
            if let Some(f) = feeder.take() {
                let _ = f.join();
            }
        }
    }
}

/// Collate a `Vec<DataItem>` into a single batched `DataItem`.
///
/// Stacks all inputs along a new leading batch dimension, and likewise
/// for targets.
fn collate(items: Vec<DataItem>) -> DataItem {
    let inputs: Vec<_> = items.iter().map(|item| item.input.clone()).collect();
    let targets: Vec<_> = items.iter().map(|item| item.target.clone()).collect();

    DataItem {
        input: tensor::stack(&inputs),
        target: tensor::stack(&targets),
    }
}

// ---------------------------------------------------------------------------
// Fisher-Yates shuffle with zero-dep LCG PRNG
// ---------------------------------------------------------------------------

use std::cell::Cell;

thread_local! {
    static SHUFFLE_RNG: Cell<u64> = Cell::new(0xDEADBEEFCAFE1234);
}

fn lcg_next() -> u64 {
    SHUFFLE_RNG.with(|state| {
        let s = state
            .get()
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state.set(s);
        s
    })
}

fn fisher_yates_shuffle(indices: &mut [usize]) {
    let n = indices.len();
    for i in (1..n).rev() {
        let j = (lcg_next() as usize) % (i + 1);
        indices.swap(i, j);
    }
}
