//! Transposed 2D convolution (deconvolution) via matmul + col2im.

use std::cell::Cell;
use std::collections::HashMap;

use crate::autograd::AutogradError;
use crate::backend::{Backend, CpuBackend};
use crate::nn::{Module, Parameter};
use crate::tensor::{self, Tensor};

thread_local! {
    static CONVT_RNG: Cell<u64> = Cell::new(987654321);
}

fn lcg_uniform(bound: f32) -> f32 {
    CONVT_RNG.with(|state| {
        let s = state
            .get()
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state.set(s);
        let u = (s >> 33) as f32 / (1u64 << 31) as f32;
        (2.0 * u - 1.0) * bound
    })
}

/// Transposed 2D convolution (sometimes called "deconvolution").
///
/// Upsamples spatial dimensions.  Implemented as a composition of
/// tracked ops so the autograd engine handles backward automatically.
///
/// Forward algorithm per batch element:
///   1. `cols = weight^T @ x_flat`:  `[col_height, num_patches]`
///   2. `image = col2im(cols)`:  `[C_out, H_out, W_out]`
///   3. Add channel bias (if present).
///
/// Weight shape: `[in_channels, out_channels * kernel_size * kernel_size]`.
///
/// The transpose convolution with `stride > 1` produces an output of size:
///   `H_out = (H_in - 1) * stride - 2 * padding + kernel_size`
pub struct ConvTranspose2d {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

impl ConvTranspose2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        with_bias: bool,
    ) -> Self {
        let fan_in = in_channels;
        let bound = (1.0 / fan_in as f32).sqrt();

        let col_height = out_channels * kernel_size * kernel_size;
        let weight_data: Vec<f32> = (0..in_channels * col_height)
            .map(|_| lcg_uniform(bound))
            .collect();
        // Weight shape: [in_channels, out_channels * K * K]
        let weight = Parameter::new(Tensor::new(
            weight_data,
            vec![in_channels, col_height],
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
    /// `input` shape: `[B, in_channels, H_in, W_in]`.
    /// Output shape: `[B, out_channels, H_out, W_out]`.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(input.ndim(), 4, "ConvTranspose2d: input must be 4-D [B, C, H, W]");
        let batch = input.shape()[0];
        let c_in = input.shape()[1];
        let h_in = input.shape()[2];
        let w_in = input.shape()[3];
        assert_eq!(c_in, self.in_channels, "ConvTranspose2d: channel mismatch");

        let h_out = (h_in - 1) * self.stride - 2 * self.padding + self.kernel_size;
        let w_out = (w_in - 1) * self.stride - 2 * self.padding + self.kernel_size;
        let num_patches = h_in * w_in;

        // weight^T: [col_height, in_channels]
        let w_t = self.weight.tensor.transpose_tracked(0, 1);

        let mut batch_outputs: Vec<Tensor> = Vec::with_capacity(batch);

        for b in 0..batch {
            // Slice out batch element: [C_in, H_in, W_in]
            let x_b = input.slice_batch(b);

            // Reshape to [C_in, H_in * W_in] for matmul
            let x_flat = x_b.reshape_tracked(vec![c_in, num_patches]);

            // cols = W^T @ x_flat: [col_height, num_patches]
            let cols = w_t.matmul(&x_flat);

            // col2im: [col_height, num_patches] → [C_out, H_out, W_out]
            // col2im is the inverse of im2col — scatters columns back into image.
            // We need to call it on the contiguous column data.
            let cols_c = cols.contiguous_tracked();
            let cols_guard = cols_c.storage.data();
            let mut img = CpuBackend::zeros(self.out_channels * h_out * w_out);
            CpuBackend::col2im(
                &cols_guard, &mut img,
                self.out_channels, h_out, w_out,
                self.kernel_size, self.stride, self.padding,
                h_in, w_in,
            );
            drop(cols_guard);

            let mut out_b = Tensor::new(img, vec![self.out_channels, h_out * w_out]);

            // Add channel bias
            if let Some(ref bias) = self.bias {
                out_b = out_b.add_channel_bias(&bias.tensor);
            }

            batch_outputs.push(out_b);
        }

        // Stack: Vec<[C_out, H_out*W_out]> → [B, C_out, H_out*W_out]
        let stacked = tensor::stack(&batch_outputs);

        // Reshape to [B, C_out, H_out, W_out]
        stacked.reshape_tracked(vec![batch, self.out_channels, h_out, w_out])
    }
}

impl Module for ConvTranspose2d {
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
        self.weight.load_state_dict(dict, &format!("{}weight.", prefix))?;
        if let Some(ref mut bias) = self.bias {
            bias.load_state_dict(dict, &format!("{}bias.", prefix))?;
        }
        Ok(())
    }
}
