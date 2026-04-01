//! End-to-end training loop tests for M7b.
//!
//! Tests:
//! 1. CPU 3-class classification with AdamW + cross_entropy_loss + Trainer.
//! 2. GPU variant (feature-gated) with BufferPool leak detection.

use rumus::nn::{self, Linear, Module};
use rumus::optim::AdamW;
use rumus::tensor::Tensor;
use rumus::train::Trainer;

// ---------------------------------------------------------------------------
// Model: 2-layer MLP for 3-class classification
// ---------------------------------------------------------------------------

#[derive(Module)]
struct ClassifierMLP {
    fc1: Linear,
    fc2: Linear,
}

impl ClassifierMLP {
    fn new() -> Self {
        Self {
            fc1: Linear::new(4, 8, true),
            fc2: Linear::new(8, 3, true),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let h = nn::relu(&self.fc1.forward(x));
        self.fc2.forward(&h) // raw logits [B, 3]
    }
}

// ---------------------------------------------------------------------------
// Synthetic 3-class dataset (12 samples, 4 features)
// ---------------------------------------------------------------------------

/// Generate a deterministic 3-class dataset.
///
/// Class 0: feature 0 is dominant (1.0), others are 0.1.
/// Class 1: feature 1 is dominant.
/// Class 2: feature 2 is dominant.
///
/// 4 samples per class = 12 total.
fn make_dataset() -> (Tensor, Tensor) {
    let mut data = Vec::with_capacity(12 * 4);
    let mut targets = Vec::with_capacity(12);

    for class in 0..3u32 {
        for variant in 0..4u32 {
            let mut row = [0.1f32; 4];
            row[class as usize] = 1.0;
            // Small variation per variant to prevent degeneracy.
            row[3] = variant as f32 * 0.05;
            data.extend_from_slice(&row);
            targets.push(class as f32);
        }
    }

    (
        Tensor::new(data, vec![12, 4]),
        Tensor::new(targets, vec![12]),
    )
}

// ---------------------------------------------------------------------------
// Test 1: CPU training with Trainer + AdamW + cross_entropy_loss
// ---------------------------------------------------------------------------

#[test]
fn test_cpu_cross_entropy_training() {
    let (inputs, targets) = make_dataset();
    let model = ClassifierMLP::new();
    let optimizer = AdamW::new(model.parameters(), 0.01);
    let mut trainer = Trainer::new(optimizer);

    let mut final_loss = f32::MAX;
    for _epoch in 0..200 {
        trainer.reset_epoch();
        let loss_val = trainer
            .train_step(|| {
                let logits = model.forward(&inputs);
                nn::cross_entropy_loss(&logits, &targets)
            })
            .expect("train_step failed");
        final_loss = loss_val;
    }

    assert!(
        final_loss < 0.15,
        "CPU 3-class training did not converge: final loss = {:.6}",
        final_loss,
    );
}

// ---------------------------------------------------------------------------
// Test 2: GPU training with BufferPool leak detection
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_cross_entropy_training() {
    use rumus::backend::gpu::context::GpuContext;
    use rumus::nn::ModuleToGpu;

    if !GpuContext::is_available() {
        eprintln!("Skipping GPU training test: no GPU available");
        return;
    }

    let (inputs, targets) = make_dataset();
    let model = ClassifierMLP::new();

    model.to_gpu();
    inputs.to_gpu();
    targets.to_gpu();

    let optimizer = AdamW::new(model.parameters(), 0.01);
    let mut trainer = Trainer::new(optimizer);

    let ctx = GpuContext::get().unwrap();

    let mut final_loss = f32::MAX;
    for epoch in 0..200 {
        trainer.reset_epoch();
        let loss_val = trainer
            .train_step(|| {
                let logits = model.forward(&inputs);
                nn::cross_entropy_loss(&logits, &targets)
            })
            .expect("train_step failed");
        final_loss = loss_val;

        // After the first full iteration, intermediate tensors from the
        // forward/backward/optimizer cycle should have been dropped,
        // returning their GPU buffers to the pool.
        if epoch == 1 {
            let count = ctx.pool.cached_count();
            assert!(
                count > 0,
                "BufferPool is empty after epoch 1 — buffers are leaking \
                 instead of being recycled! cached_count = {}",
                count,
            );
        }
    }

    assert!(
        final_loss < 0.15,
        "GPU 3-class training did not converge: final loss = {:.6}",
        final_loss,
    );

    // The pool should have a nonzero number of recycled buffers,
    // proving the Drop → release lifecycle is working.
    let pool_size = ctx.pool.cached_count();
    assert!(
        pool_size > 0,
        "BufferPool is empty after 200 steps — Drop is not returning buffers!",
    );
}

// ---------------------------------------------------------------------------
// Test 3: Trainer API ergonomics
// ---------------------------------------------------------------------------

#[test]
fn test_trainer_api() {
    let (inputs, targets) = make_dataset();
    let model = ClassifierMLP::new();
    let optimizer = AdamW::new(model.parameters(), 0.01);
    let mut trainer = Trainer::new(optimizer);

    // Initial state.
    assert_eq!(trainer.epoch_avg_loss(), 0.0);

    // Run 3 steps.
    for _ in 0..3 {
        trainer
            .train_step(|| {
                let logits = model.forward(&inputs);
                nn::cross_entropy_loss(&logits, &targets)
            })
            .unwrap();
    }

    // Epoch average should be nonzero and finite.
    let avg = trainer.epoch_avg_loss();
    assert!(avg > 0.0 && avg.is_finite(), "bad epoch avg: {}", avg);

    // Reset and verify.
    trainer.reset_epoch();
    assert_eq!(trainer.epoch_avg_loss(), 0.0);
}
