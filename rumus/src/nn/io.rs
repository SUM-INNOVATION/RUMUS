//! Safetensors serialization and deserialization for state dictionaries.
//!
//! Uses the industry-standard `safetensors` format for secure, portable
//! model persistence.  No `unsafe` — `bytemuck::cast_slice` handles the
//! `f32 → u8` reinterpret safely.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use safetensors::tensor::SafeTensors;
use safetensors::Dtype;

use crate::autograd::AutogradError;
use crate::tensor::Tensor;

/// Save a state dictionary to a safetensors file.
///
/// Each tensor is stored as F32 in the safetensors format.  The `RwLock`
/// read guard on each tensor's storage is held only for the duration of
/// the byte conversion.
///
/// # Errors
///
/// Returns [`AutogradError::StateError`] on IO failure.
pub fn save_safetensors(
    dict: &HashMap<String, Tensor>,
    path: &Path,
) -> Result<(), AutogradError> {
    // Build the safetensors data map: name → (Vec<u8>, shape, dtype).
    // We need to collect all data upfront because safetensors::serialize
    // requires all data to be available simultaneously.
    let mut data_map: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::with_capacity(dict.len());

    for (name, tensor) in dict {
        let guard = tensor.storage.data();
        // Safe f32 → u8 reinterpretation via bytemuck.
        let bytes: &[u8] = bytemuck::cast_slice(guard.as_slice());
        data_map.push((
            name.clone(),
            bytes.to_vec(),
            tensor.shape().to_vec(),
        ));
    }

    // Build TensorView references and serialize.
    // safetensors::serialize expects IntoIterator<Item = (impl AsRef<str>, TensorView)>.
    let views: Vec<(&str, safetensors::tensor::TensorView<'_>)> = data_map
        .iter()
        .map(|(name, bytes, shape)| {
            (
                name.as_str(),
                safetensors::tensor::TensorView::new(Dtype::F32, shape.clone(), bytes)
                    .expect("invalid tensor view"),
            )
        })
        .collect();

    let serialized = safetensors::tensor::serialize(views, &None).map_err(|e| {
        AutogradError::StateError {
            key: String::new(),
            message: format!("safetensors serialize error: {}", e),
        }
    })?;

    fs::write(path, &serialized).map_err(|e| AutogradError::StateError {
        key: String::new(),
        message: format!("IO write error: {}", e),
    })?;

    Ok(())
}

/// Load a state dictionary from a safetensors file.
///
/// Returns a `HashMap<String, Tensor>` with inference-mode tensors
/// (`AutogradState::None`).
///
/// # Errors
///
/// Returns [`AutogradError::StateError`] on IO failure, parse failure,
/// or unsupported dtype (only F32 is supported).
pub fn load_safetensors(
    path: &Path,
) -> Result<HashMap<String, Tensor>, AutogradError> {
    let bytes = fs::read(path).map_err(|e| AutogradError::StateError {
        key: String::new(),
        message: format!("IO read error: {}", e),
    })?;

    let tensors = SafeTensors::deserialize(&bytes).map_err(|e| {
        AutogradError::StateError {
            key: String::new(),
            message: format!("safetensors parse error: {}", e),
        }
    })?;

    let mut dict = HashMap::new();

    for (name, view) in tensors.tensors() {
        if view.dtype() != Dtype::F32 {
            return Err(AutogradError::StateError {
                key: name.to_string(),
                message: format!("unsupported dtype {:?}, expected F32", view.dtype()),
            });
        }

        let data_bytes = view.data();
        let shape: Vec<usize> = view.shape().to_vec();

        // Defensive: reject trailing bytes that don't form a complete f32.
        if data_bytes.len() % 4 != 0 {
            return Err(AutogradError::StateError {
                key: name.to_string(),
                message: format!(
                    "data length {} is not a multiple of 4 (F32 size)",
                    data_bytes.len(),
                ),
            });
        }

        // Safe u8 → f32 conversion via from_le_bytes.
        let floats: Vec<f32> = data_bytes
            .chunks_exact(4)
            .map(|chunk| {
                let arr: [u8; 4] = chunk.try_into().expect("chunk must be 4 bytes");
                f32::from_le_bytes(arr)
            })
            .collect();

        dict.insert(name.to_string(), Tensor::new(floats, shape));
    }

    Ok(dict)
}
