//! 2D convolutional layer via im2col + matmul.

use std::cell::Cell;
use std::collections::HashMap;

use crate::autograd::AutogradError;
use crate::nn::{Module, Parameter};
use crate::tensor::{self, Tensor};

// Reuse the LCG from linear.rs — same thread-local state.
thread_local! {
    static CONV_RNG: Cell<u64> = Cell::new(123456789);
}

fn lcg_uniform(bound: f32) -> f32 {
    CONV_RNG.with(|state| {
        let s = state
            .get()
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state.set(s);
        let u = (s >> 33) as f32 / (1u64 << 31) as f32;
        (2.0 * u - 1.0) * bound
    })
}

/// 2D convolutional layer.
///
/// Uses the im2col algorithm: reshape input patches into columns, then
/// compute the convolution as a single `matmul` per batch element.
///
/// Weight shape: `[out_channels, in_channels * kernel_size * kernel_size]`.
/// Bias shape: `[out_channels]` (optional).
///
/// Forward: for each batch element:
///   1. `x_col = input[b].im2col(k, stride, padding)` → `[col_height, num_patches]`
///   2. `out_b = weight @ x_col` → `[out_channels, num_patches]`
///   3. Stack all `out_b` → `[batch, out_channels, num_patches]`
///   4. Add channel bias if present.
///   5. Reshape to `[batch, out_channels, out_h, out_w]`.
///
/// All ops are individually tape-recorded — the autograd engine handles
/// the entire backward pass via Kahn's traversal.
pub struct Conv2d {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

impl Conv2d {
    /// Create a new Conv2d layer with Kaiming Uniform initialization.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        with_bias: bool,
    ) -> Self {
        let fan_in = in_channels * kernel_size * kernel_size;
        let bound = (1.0 / fan_in as f32).sqrt();

        let weight_data: Vec<f32> = (0..out_channels * fan_in)
            .map(|_| lcg_uniform(bound))
            .collect();
        let weight = Parameter::new(Tensor::new(
            weight_data,
            vec![out_channels, fan_in],
        ));

        let bias = if with_bias {
            let bias_data: Vec<f32> = (0..out_channels)
                .map(|_| lcg_uniform(bound))
                .collect();
            Some(Parameter::new(Tensor::new(bias_data, vec![out_channels])))
        } else {
            None
        };

        Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        }
    }

    /// Forward pass.
    ///
    /// `input` shape: `[batch, in_channels, height, width]`.
    /// Output shape: `[batch, out_channels, out_h, out_w]`.
    ///
    /// Each step is a tracked op so the autograd tape records the full
    /// graph:  `slice_batch → im2col → matmul → add_channel_bias → stack`.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(input.ndim(), 4, "Conv2d: input must be 4-D [B, C, H, W]");
        let batch = input.shape()[0];
        let c_in = input.shape()[1];
        let h = input.shape()[2];
        let w = input.shape()[3];
        assert_eq!(c_in, self.in_channels, "Conv2d: channel mismatch");

        let out_h = (h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let out_w = (w + 2 * self.padding - self.kernel_size) / self.stride + 1;

        let mut batch_outputs: Vec<Tensor> = Vec::with_capacity(batch);

        for b in 0..batch {
            // Tracked slice: [batch, C_in, H, W] → [C_in, H, W]
            // Gradients flow back to the original batched input.
            let x_b = input.slice_batch(b);

            // im2col: [C_in, H, W] → [col_height, num_patches]
            let x_col = x_b.im2col(self.kernel_size, self.stride, self.padding);

            // matmul: [C_out, col_height] @ [col_height, num_patches]
            //       → [C_out, num_patches]
            let mut out_b = self.weight.tensor.matmul(&x_col);

            // Add channel bias inside the loop — out_b shape is
            // [out_channels, num_patches], which matches bias [out_channels].
            if let Some(ref bias) = self.bias {
                out_b = out_b.add_channel_bias(&bias.tensor);
            }

            batch_outputs.push(out_b);
        }

        // stack: Vec<[C_out, num_patches]> → [batch, C_out, num_patches]
        let stacked = tensor::stack(&batch_outputs);

        // Reshape to final [batch, out_channels, out_h, out_w].
        stacked.reshape(vec![batch, self.out_channels, out_h, out_w])
    }
}

impl Module for Conv2d {
    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn state_dict(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut dict = self.weight.state_dict(&format!("{}weight.", prefix));
        if let Some(ref bias) = self.bias {
            dict.extend(bias.state_dict(&format!("{}bias.", prefix)));
        }
        dict
    }

    fn load_state_dict(
        &mut self,
        dict: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<(), AutogradError> {
        self.weight
            .load_state_dict(dict, &format!("{}weight.", prefix))?;
        if let Some(ref mut bias) = self.bias {
            bias.load_state_dict(dict, &format!("{}bias.", prefix))?;
        }
        Ok(())
    }
}
