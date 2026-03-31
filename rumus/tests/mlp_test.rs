//! End-to-end integration test: XOR training + save/load roundtrip.
//!
//! This test proves the entire RUMUS stack works:
//! - `Linear` layer with bias broadcasting
//! - `relu` activation with correct backward
//! - `mse_loss` with fused gradient
//! - `backward()` (Kahn's algorithm with edge counting)
//! - `Adam` optimizer (moment buffers, bias correction, RwLock writes)
//! - `state_dict` / `load_state_dict` roundtrip via safetensors
//! - `#[derive(Module)]` proc macro

use rumus::autograd;
use rumus::nn::{self, Linear, Module};
use rumus::optim::{Adam, Optimizer};
use rumus::tensor::Tensor;

/// XOR MLP: 2 → 8 → 1 with ReLU activation.
///
/// XOR is a classic non-linearly-separable problem that requires a hidden
/// layer + nonlinearity — proving the entire autograd stack, not just a
/// single linear regression.
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
fn test_xor_training_and_save_load() {
    // ---- XOR dataset ----
    // [0,0]→0, [0,1]→1, [1,0]→1, [1,1]→0
    let inputs = Tensor::new(
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        vec![4, 2],
    );
    let targets = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1]);

    // ---- Model + Optimizer ----
    let model = XorMLP::new();
    let mut optimizer = Adam::new(model.parameters(), 0.01);

    // ---- Training loop ----
    let mut final_loss = f32::MAX;
    for _epoch in 0..200 {
        let pred = model.forward(&inputs);
        let loss = nn::mse_loss(&pred, &targets);

        // Read loss value.
        {
            let loss_guard = loss.data();
            final_loss = loss_guard[0];
        }

        // Backward pass.
        let mut grads = autograd::backward(&loss).expect("backward failed");

        // Optimizer step.
        optimizer.step(&mut grads).expect("optimizer step failed");
    }

    // ---- Assert convergence ----
    assert!(
        final_loss < 0.01,
        "XOR training did not converge: final loss = {:.6}",
        final_loss,
    );

    // ---- Save / Load roundtrip ----
    let dir = std::env::temp_dir().join("rumus_test_xor.safetensors");

    // Save.
    let state = model.state_dict("");
    nn::save_safetensors(&state, &dir).expect("save failed");

    // Verify keys are present.
    assert!(state.contains_key("hidden.weight"), "missing hidden.weight");
    assert!(state.contains_key("hidden.bias"), "missing hidden.bias");
    assert!(state.contains_key("output.weight"), "missing output.weight");
    assert!(state.contains_key("output.bias"), "missing output.bias");

    // Create a fresh model with different random weights.
    let mut model2 = XorMLP::new();

    // Load saved weights into fresh model.
    let loaded_state = nn::load_safetensors(&dir).expect("load failed");
    model2
        .load_state_dict(&loaded_state, "")
        .expect("load_state_dict failed");

    // ---- Verify outputs match exactly ----
    // Use no_grad to avoid polluting the tape.
    let _guard = autograd::context::no_grad();
    let out1 = model.forward(&inputs);
    let out2 = model2.forward(&inputs);
    let g1 = out1.data();
    let g2 = out2.data();
    assert_eq!(out1.numel(), out2.numel());
    for i in 0..out1.numel() {
        assert_eq!(
            g1[i], g2[i],
            "output mismatch at index {}: original={}, loaded={}",
            i, g1[i], g2[i],
        );
    }

    // Cleanup.
    std::fs::remove_file(&dir).ok();
}
