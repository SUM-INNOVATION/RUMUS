//! End-to-end "Tiny GPT" test.
//!
//! Architecture: Token Embedding + Positional Embedding + TransformerBlock + Linear Head.
//! Task: Next-token prediction on a repeating sequence [0,1,2,3,0,1,2,3,...].
//! Proves that attention, LayerNorm, GELU, bmm, softmax, broadcasting,
//! cross-entropy, and AdamW all compose correctly through the autograd engine.

use rumus::nn::{
    cross_entropy_loss, Embedding, LayerNorm, Linear, Module, TransformerBlock,
};
use rumus::optim::AdamW;
use rumus::tensor::Tensor;
use rumus::train::Trainer;


// ---------------------------------------------------------------------------
// TinyGPT model
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct TinyGPT {
    token_emb: Embedding,
    pos_emb: Embedding,
    block: TransformerBlock,
    ln_final: LayerNorm,
    head: Linear,
    seq_len: usize,
}

impl TinyGPT {
    fn new(vocab_size: usize, seq_len: usize, d_model: usize, num_heads: usize) -> Self {
        Self {
            token_emb: Embedding::new(vocab_size, d_model),
            pos_emb: Embedding::new(seq_len, d_model),
            block: TransformerBlock::new(d_model, num_heads),
            ln_final: LayerNorm::new(d_model, 1e-5),
            head: Linear::new(d_model, vocab_size, false),
            seq_len,
        }
    }

    fn forward(&self, token_ids: &Tensor, mask: &Tensor) -> Tensor {
        let b = token_ids.shape()[0];
        let s = token_ids.shape()[1];
        let d = self.token_emb.embed_dim;

        // Token embeddings: [B, S] → [B, S, D]
        let tok = self.token_emb.forward(token_ids);

        // Positional embeddings: [S] → [S, D] → broadcast add to [B, S, D]
        let pos_ids = Tensor::new((0..s).map(|i| i as f32).collect(), vec![s]);
        let pos = self.pos_emb.forward(&pos_ids); // [S, D]
        let x = tok.broadcast_add(&pos);

        // Transformer block
        let x = self.block.forward(&x, Some(mask));

        // Final LayerNorm + vocab projection
        let x = self.ln_final.forward(&x);
        let flat = x.reshape_tracked(vec![b * s, d]);
        let logits = self.head.forward(&flat);
        logits.reshape_tracked(vec![b, s, self.head.weight.tensor.shape()[1]])
    }

    fn parameters(&self) -> Vec<rumus::nn::Parameter> {
        let mut p = self.token_emb.parameters();
        p.extend(self.pos_emb.parameters());
        p.extend(self.block.parameters());
        p.extend(self.ln_final.parameters());
        p.extend(self.head.parameters());
        p
    }
}

// ---------------------------------------------------------------------------
// Data generation
// ---------------------------------------------------------------------------

/// Generate a causal mask: upper triangle = -1e9, lower+diagonal = 0.
fn causal_mask(seq_len: usize) -> Tensor {
    let mut data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            data[i * seq_len + j] = -1e9;
        }
    }
    Tensor::new(data, vec![seq_len, seq_len])
}

/// Generate repeating sequences: `[0, 1, 2, 3, 0, 1, ...]`.
/// Returns (input, target) where target is input shifted left by 1.
fn make_repeating_data(batch: usize, seq_len: usize, vocab: usize) -> (Tensor, Tensor) {
    let mut input_data = Vec::with_capacity(batch * seq_len);
    let mut target_data = Vec::with_capacity(batch * seq_len);
    for b in 0..batch {
        for s in 0..seq_len {
            let token = ((b + s) % vocab) as f32;
            let next = ((b + s + 1) % vocab) as f32;
            input_data.push(token);
            target_data.push(next);
        }
    }
    (
        Tensor::new(input_data, vec![batch, seq_len]),
        Tensor::new(target_data, vec![batch * seq_len]),
    )
}

// ---------------------------------------------------------------------------
// CPU Test
// ---------------------------------------------------------------------------

#[test]
fn test_tiny_gpt_cpu() {
    let vocab = 4;
    let seq_len = 6;
    let d_model = 32;
    let num_heads = 1;
    let batch = 8;

    let (inputs, targets) = make_repeating_data(batch, seq_len, vocab);
    let mask = causal_mask(seq_len);

    let model = TinyGPT::new(vocab, seq_len, d_model, num_heads);
    let optimizer = AdamW::new(model.parameters(), 0.001);
    let mut trainer = Trainer::new(optimizer);

    let mut final_loss = f32::MAX;
    for _epoch in 0..200 {
        trainer.reset_epoch();
        let loss = trainer
            .train_step(|| {
                let logits = model.forward(&inputs, &mask); // [B, S, vocab]
                let logits_flat = logits.reshape_tracked(vec![batch * seq_len, vocab]);
                cross_entropy_loss(&logits_flat, &targets)
            })
            .expect("train_step failed");
        final_loss = loss;
    }

    assert!(
        final_loss < 1.0,
        "Tiny GPT did not converge: final loss = {:.4}",
        final_loss,
    );
}

// ---------------------------------------------------------------------------
// GPU Test
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
#[test]
fn test_tiny_gpt_gpu() {
    use rumus::backend::gpu::context::GpuContext;
    use rumus::nn::ModuleToGpu;

    if !GpuContext::is_available() {
        eprintln!("Skipping GPU Transformer test: no GPU available");
        return;
    }

    let vocab = 4;
    let seq_len = 6;
    let d_model = 32;
    let num_heads = 1;
    let batch = 8;

    let (inputs, targets) = make_repeating_data(batch, seq_len, vocab);
    let mask = causal_mask(seq_len);

    let model = TinyGPT::new(vocab, seq_len, d_model, num_heads);

    // Push everything to GPU.
    for p in model.parameters() { p.tensor.to_gpu(); }
    inputs.to_gpu();
    targets.to_gpu();
    mask.to_gpu();

    let optimizer = AdamW::new(model.parameters(), 0.001);
    let mut trainer = Trainer::new(optimizer);

    let ctx = GpuContext::get().unwrap();
    let mut final_loss = f32::MAX;
    for epoch in 0..200 {
        trainer.reset_epoch();
        let loss = trainer
            .train_step(|| {
                let logits = model.forward(&inputs, &mask);
                let logits_flat = logits.reshape_tracked(vec![batch * seq_len, vocab]);
                cross_entropy_loss(&logits_flat, &targets)
            })
            .expect("train_step failed");
        final_loss = loss;

        // Buffer pool leak check mid-training.
        if epoch == 10 {
            let count = ctx.pool.cached_count();
            assert!(
                count > 0,
                "BufferPool empty at epoch 10 — buffers not recycling",
            );
        }
    }

    assert!(
        final_loss < 1.0,
        "GPU Tiny GPT did not converge: final loss = {:.4}",
        final_loss,
    );
}
