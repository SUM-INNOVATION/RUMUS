//! Dataset trait and data item types.

use crate::tensor::Tensor;

/// A single data item: input-target pair.
pub struct DataItem {
    pub input: Tensor,
    pub target: Tensor,
}

/// Trait for indexable datasets.
///
/// Implementations must be `Send + Sync` because `DataLoader` worker threads
/// call `get()` concurrently via `Arc<dyn Dataset>`.
pub trait Dataset: Send + Sync {
    /// Total number of items in the dataset.
    fn len(&self) -> usize;

    /// Retrieve the item at `index`.
    ///
    /// # Panics
    ///
    /// May panic if `index >= len()`.
    fn get(&self, index: usize) -> DataItem;

    /// Whether the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
