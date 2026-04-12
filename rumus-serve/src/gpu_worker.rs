// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Dedicated GPU worker thread: runs the scheduler loop + generation.
//!
//! Owns the model and KV-cache.  Communicates with the async HTTP
//! handlers via `tokio::sync::mpsc` (receive) and `oneshot` (respond).

use std::collections::HashMap;
use std::time::Instant;

use rumus::tensor::Tensor;
use tokio::sync::mpsc;

use crate::model::{KvCache, MockTransformer};
use crate::scheduler::{self, PaddedBatch};
use crate::server::{InferenceRequest, InferenceResponse};

/// Configuration for the GPU worker.
pub struct WorkerConfig {
    pub max_batch_size: usize,
    pub batch_timeout: std::time::Duration,
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
}

/// Main entry point for the GPU worker thread.
///
/// Runs forever: collects batches, runs prefill + decode, scatters results.
pub fn run(mut rx: mpsc::Receiver<InferenceRequest>, config: WorkerConfig) {
    eprintln!("[gpu_worker] started");

    let model = MockTransformer::new(
        config.vocab_size,
        config.hidden_dim,
        config.num_heads,
        config.num_layers,
    );

    loop {
        // 1. Collect a batch of requests (blocks on first, deadline-drains rest).
        let requests = scheduler::collect_batch(
            &mut rx,
            config.max_batch_size,
            config.batch_timeout,
        );

        if requests.is_empty() {
            // Channel closed — server shutting down.
            eprintln!("[gpu_worker] channel closed, exiting");
            break;
        }

        let batch_start = Instant::now();
        let batch_size = requests.len();
        eprintln!("[gpu_worker] batch of {} requests", batch_size);

        // 2. Pad into uniform tensors + attention mask.
        let padded = scheduler::pad_batch(requests);

        // 3. Run generation (prefill + decode loop).
        let generated = generate(&model, padded);

        let elapsed = batch_start.elapsed();
        eprintln!(
            "[gpu_worker] batch done in {:.1}ms ({} requests)",
            elapsed.as_secs_f64() * 1000.0,
            batch_size,
        );

        // Results are already sent via oneshot in `generate`.
        drop(generated);
    }
}

/// Run the full generation pipeline: prefill → decode loop → scatter results.
fn generate(model: &MockTransformer, batch: PaddedBatch) {
    let batch_size = batch.seq_lens.len();
    let max_new = batch.max_new_tokens;

    // Track the full generated sequence per request.
    let mut generated: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
    {
        let guard = batch.input_ids.data();
        let max_seq = batch.input_ids.shape()[1];
        for b in 0..batch_size {
            let len = batch.seq_lens[b];
            let mut seq = Vec::with_capacity(len + max_new);
            for i in 0..len {
                seq.push(guard[b * max_seq + i] as u32);
            }
            generated.push(seq);
        }
    }

    // ---- Prefill phase ----
    // Run the full sequence through the model with attention masking.
    let mut kv_cache: KvCache = HashMap::new();

    let _guard = rumus::autograd::context::no_grad();

    let prefill_logits = model.prefill(
        &batch.input_ids,
        &batch.attention_mask,
        &mut kv_cache,
    );

    // Extract the first generated token from the last real position per sequence.
    let first_tokens = scheduler::argmax_last_token(
        &prefill_logits,
        &batch.seq_lens,
        false, // prefill: logits are [B, max_seq, V], use seq_lens to find last real pos
    );
    drop(prefill_logits);

    for (b, &tok) in first_tokens.iter().enumerate() {
        generated[b].push(tok);
    }

    // ---- Decode phase (KV-cached) ----
    // Each step: pass only the NEW token [B, 1] + KV-cache.
    for _step in 1..max_new {
        // Build [B, 1] tensor of the most recently generated tokens.
        let last_tokens: Vec<f32> = generated.iter().map(|g| *g.last().unwrap() as f32).collect();
        let new_ids = Tensor::new(last_tokens, vec![batch_size, 1]);

        // Decode step: only processes the new token, extends KV-cache.
        let step_logits = model.decode_step(&new_ids, &mut kv_cache);

        // Extract next tokens (logits are [B, 1, V]).
        let seq_lens_dummy: Vec<usize> = vec![1; batch_size];
        let next_tokens = scheduler::argmax_last_token(
            &step_logits,
            &seq_lens_dummy,
            true, // decode: logits are [B, 1, V]
        );
        drop(step_logits);

        for (b, &tok) in next_tokens.iter().enumerate() {
            generated[b].push(tok);
        }
    }

    // ---- Scatter results back to HTTP handlers ----
    for (b, tx) in batch.response_txs.into_iter().enumerate() {
        let _ = tx.send(InferenceResponse {
            generated_ids: generated[b].clone(),
        });
    }
}
