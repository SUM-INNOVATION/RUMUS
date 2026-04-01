//! End-to-end GPU integration test: XOR training with the entire
//! forward-backward-optimizer pipeline running on the GPU.
//!
//! Gated by `#[cfg(feature = "gpu")]` — skipped in CPU-only builds.
//! Also gracefully skips at runtime if no GPU adapter is available.

#![cfg(feature = "gpu")]

use rumus::autograd;
use rumus::backend::gpu::context::GpuContext;
use rumus::nn::{self, Linear, Module, ModuleToGpu};
use rumus::optim::{Adam, Optimizer};
use rumus::tensor::Tensor;

#[derive(Module)]
struct XorMLP {
    hidden: Linear,
    output: Linear,
}

impl XorMLP {
    fn new() -> Self {
        Self {
            hidden: Linear::new(2, 8, true),
            output: Linear::new(8, 1, true),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let h = nn::relu(&self.hidden.forward(x));
        self.output.forward(&h)
    }
}

#[test]
fn test_gpu_xor_training() {
    // Skip if no GPU available (CI-friendly).
    if !GpuContext::is_available() {
        eprintln!("Skipping GPU test: no compatible GPU adapter found");
        return;
    }

    // ---- XOR dataset ----
    let inputs = Tensor::new(
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        vec![4, 2],
    );
    let targets = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1]);

    // ---- Model ----
    let model = XorMLP::new();

    // Push everything to GPU.
    model.to_gpu();
    inputs.to_gpu();
    targets.to_gpu();

    let mut optimizer = Adam::new(model.parameters(), 0.01);

    // ---- Training loop ----
    let mut final_loss = f32::MAX;
    for _epoch in 0..200 {
        let pred = model.forward(&inputs);
        let loss = nn::mse_loss(&pred, &targets);

        // mse_loss returns a scalar — reading it triggers a 4-byte D2H.
        {
            let g = loss.data();
            final_loss = g[0];
        }

        let mut grads = autograd::backward(&loss).expect("backward failed");
        optimizer.step(&mut grads).expect("optimizer step failed");
    }

    // ---- Assert convergence ----
    assert!(
        final_loss < 0.01,
        "GPU XOR training did not converge: final loss = {:.6}",
        final_loss,
    );
}
