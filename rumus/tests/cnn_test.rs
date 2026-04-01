//! End-to-end CNN integration test: spatial pattern classification.
//!
//! Proves Conv2d, MaxPool2d, Flatten, and Linear work together with
//! full autograd and optimizer support.
//!
//! Architecture (Mini-LeNet):
//!   Conv2d(1→4, k=3, s=1, p=0)  → [B,4,6,6]
//!   ReLU
//!   MaxPool2d(k=2, s=2)         → [B,4,3,3]   (per batch element: [4,3,3])
//!   Flatten                      → [B,36]
//!   Linear(36→16) + ReLU
//!   Linear(16→1)
//!
//! Dataset: 4 synthetic 8×8 images (1 channel).
//!   Class 0: center 4×4 block is 0, border is 1.
//!   Class 1: center 4×4 block is 1, border is 0.

use rumus::autograd;
use rumus::nn::{self, Conv2d, Flatten, Linear, MaxPool2d, Module};
use rumus::optim::{Adam, Optimizer};
use rumus::tensor::{self, Tensor};

#[derive(Module)]
struct MiniCNN {
    conv: Conv2d,
    pool: MaxPool2d,
    flat: Flatten,
    fc1: Linear,
    fc2: Linear,
}

impl MiniCNN {
    fn new() -> Self {
        Self {
            conv: Conv2d::new(1, 4, 3, 1, 0, true),  // [B,1,8,8] → [B,4,6,6]
            pool: MaxPool2d::new(2, 2),
            flat: Flatten::new(),
            fc1: Linear::new(36, 16, true),
            fc2: Linear::new(16, 1, true),
        }
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let batch = input.shape()[0];

        // Conv2d handles batching internally via slice_batch.
        let x = self.conv.forward(input); // [B,4,6,6]

        // Pool + flatten operate per-batch (they expect 3-D input).
        let mut pooled_outputs: Vec<Tensor> = Vec::with_capacity(batch);
        for b in 0..batch {
            let x_b = x.slice_batch(b);          // [4,6,6]
            let x_b = nn::relu(&x_b);
            let x_b = self.pool.forward(&x_b);   // [4,3,3]
            pooled_outputs.push(x_b);
        }
        let pooled = tensor::stack(&pooled_outputs); // [B,4,3,3]

        let flat = self.flat.forward(&pooled);       // [B,36]
        let x = nn::relu(&self.fc1.forward(&flat));
        self.fc2.forward(&x)                          // [B,1]
    }
}

/// Generate a synthetic 8×8 image.
/// `center`: if true, center 4×4 block is 1.0; if false, border is 1.0.
fn make_image(center: bool) -> Vec<f32> {
    let mut img = vec![0.0f32; 64];
    for r in 0..8 {
        for c in 0..8 {
            let is_center = r >= 2 && r < 6 && c >= 2 && c < 6;
            if center && is_center {
                img[r * 8 + c] = 1.0;
            } else if !center && !is_center {
                img[r * 8 + c] = 1.0;
            }
        }
    }
    img
}

#[test]
fn test_cnn_spatial_classification() {
    // Dataset: 4 images, 2 center (class 1) + 2 border (class 0).
    let mut input_data = Vec::with_capacity(4 * 1 * 8 * 8);
    input_data.extend(make_image(false)); // border → 0
    input_data.extend(make_image(true));  // center → 1
    input_data.extend(make_image(false)); // border → 0
    input_data.extend(make_image(true));  // center → 1

    let inputs = Tensor::new(input_data, vec![4, 1, 8, 8]);
    let targets = Tensor::new(vec![0.0, 1.0, 0.0, 1.0], vec![4, 1]);

    let model = MiniCNN::new();
    let mut optimizer = Adam::new(model.parameters(), 0.01);

    let mut final_loss = f32::MAX;
    for _epoch in 0..100 {
        let pred = model.forward(&inputs);
        let loss = nn::mse_loss(&pred, &targets);

        {
            let g = loss.data();
            final_loss = g[0];
        }

        let mut grads = autograd::backward(&loss).expect("backward failed");
        optimizer.step(&mut grads).expect("optimizer step failed");
    }

    assert!(
        final_loss < 0.05,
        "CNN training did not converge: final loss = {:.6}",
        final_loss,
    );
}
