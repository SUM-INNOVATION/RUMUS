//! Learnable parameter with globally unique identity.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::autograd::AutogradError;
use crate::nn::Module;
use crate::tensor::{GradId, ParamId, Tensor};

/// Global monotonically increasing `ParamId` counter.
///
/// Thread-safe — multiple modules can be constructed concurrently.
/// `Relaxed` ordering is sufficient: uniqueness is guaranteed by the
/// atomic RMW itself.
static NEXT_PARAM_ID: AtomicUsize = AtomicUsize::new(0);

fn alloc_param_id() -> ParamId {
    ParamId(NEXT_PARAM_ID.fetch_add(1, Ordering::Relaxed))
}

/// A learnable parameter: a [`Tensor`] with persistent identity.
///
/// On construction, the tensor is automatically set to `requires_grad = true`
/// and assigned a fresh [`GradId`].  The [`ParamId`] is globally unique and
/// stable across forward passes.
///
/// Cloning is cheap: `Arc` refcount bumps on storage and metadata.
#[derive(Clone, Debug)]
pub struct Parameter {
    /// Globally unique identifier for this parameter.
    pub id: ParamId,
    /// The underlying tensor holding the parameter's value.
    pub tensor: Tensor,
}

impl Parameter {
    /// Create a new parameter from a tensor.
    ///
    /// Automatically allocates a [`ParamId`] and sets `requires_grad = true`.
    pub fn new(mut tensor: Tensor) -> Self {
        tensor.set_requires_grad(true);
        Self {
            id: alloc_param_id(),
            tensor,
        }
    }

    /// The [`GradId`] of this parameter's tensor.
    pub fn grad_id(&self) -> GradId {
        self.tensor
            .grad_id()
            .expect("Parameter tensor must have a GradId")
    }
}

/// `Module` impl for `Parameter` — enables uniform field iteration in
/// the `#[derive(Module)]` macro.
impl Module for Parameter {
    fn parameters(&self) -> Vec<Parameter> {
        vec![self.clone()]
    }

    fn state_dict(&self, prefix: &str) -> HashMap<String, Tensor> {
        // prefix arrives as e.g. "linear1.weight." — strip trailing '.'
        let key = prefix.trim_end_matches('.').to_string();
        let mut map = HashMap::new();
        map.insert(key, self.tensor.clone());
        map
    }

    fn load_state_dict(
        &mut self,
        dict: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), AutogradError> {
        let key = prefix.trim_end_matches('.');
        if let Some(loaded) = dict.get(key) {
            if self.tensor.shape() != loaded.shape() {
                return Err(AutogradError::StateError {
                    key: key.to_string(),
                    message: format!(
                        "shape mismatch: expected {:?}, got {:?}",
                        self.tensor.shape(),
                        loaded.shape(),
                    ),
                });
            }
            // Overwrite data via RwLock write guard.
            // src and dst are separate Arc<StorageInner> — no deadlock.
            let src_guard = loaded.storage.data();
            let mut dst_guard = self.tensor.storage.data_write();
            dst_guard.copy_from_slice(&src_guard);
            drop(dst_guard);
            drop(src_guard);
            self.tensor.storage.bump_version();
        }
        // Missing key is not an error — allows partial loading.
        Ok(())
    }
}
