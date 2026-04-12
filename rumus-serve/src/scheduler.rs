// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Continuous batching scheduler: collection, padding, masking, scatter.

use std::time::{Duration, Instant};

use rumus::tensor::Tensor;
use tokio::sync::mpsc;

use crate::server::{InferenceRequest, InferenceResponse};

/// Padded batch ready for the GPU.
pub struct PaddedBatch {
    /// Input token IDs: [batch, max_seq_len] (f32-encoded).
    pub input_ids: Tensor,
    /// Attention mask: [batch, max_seq_len] — 1.0 for real, 0.0 for padding.
    pub attention_mask: Tensor,
    /// Actual sequence length per request.
    pub seq_lens: Vec<usize>,
    /// Max new tokens per request (we use the global max for the batch).
    pub max_new_tokens: usize,
    /// Response senders (one per request in the batch).
    pub response_txs: Vec<tokio::sync::oneshot::Sender<InferenceResponse>>,
}

/// Collect requests from the channel into a batch.
///
/// Blocks for the first request, then drains additional requests until
/// `max_batch_size` or `timeout` is reached.
pub fn collect_batch(
    rx: &mut mpsc::Receiver<InferenceRequest>,
    max_batch_size: usize,
    timeout: Duration,
) -> Vec<InferenceRequest> {
    // Block for the first request.
    let first = match rx.blocking_recv() {
        Some(req) => req,
        None => return vec![], // channel closed
    };

    let mut batch = vec![first];
    let deadline = Instant::now() + timeout;

    // Drain additional requests until batch is full or deadline expires.
    while batch.len() < max_batch_size {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        // Use try_recv in a tight loop with a small sleep to respect the deadline.
        match rx.try_recv() {
            Ok(req) => batch.push(req),
            Err(mpsc::error::TryRecvError::Empty) => {
                // Brief yield to avoid busy-spinning.
                std::thread::sleep(Duration::from_micros(100));
            }
            Err(mpsc::error::TryRecvError::Disconnected) => break,
        }
    }

    batch
}

/// Pad a batch of requests into uniform tensors + attention mask.
pub fn pad_batch(requests: Vec<InferenceRequest>) -> PaddedBatch {
    let batch_size = requests.len();
    let max_seq = requests.iter().map(|r| r.token_ids.len()).max().unwrap_or(1);
    let max_new_tokens = requests.iter().map(|r| r.max_new_tokens).max().unwrap_or(1);

    let mut input_data = vec![0.0f32; batch_size * max_seq];
    let mut mask_data = vec![0.0f32; batch_size * max_seq];
    let mut seq_lens = Vec::with_capacity(batch_size);
    let mut response_txs = Vec::with_capacity(batch_size);

    for (b, req) in requests.into_iter().enumerate() {
        let len = req.token_ids.len();
        seq_lens.push(len);

        for (i, &tok) in req.token_ids.iter().enumerate() {
            input_data[b * max_seq + i] = tok as f32;
            mask_data[b * max_seq + i] = 1.0; // real token
        }
        // Positions len..max_seq remain 0.0 in both input_data and mask_data (padding).

        response_txs.push(req.response_tx);
    }

    PaddedBatch {
        input_ids: Tensor::new(input_data, vec![batch_size, max_seq]),
        attention_mask: Tensor::new(mask_data, vec![batch_size, max_seq]),
        seq_lens,
        max_new_tokens,
        response_txs,
    }
}

/// Extract the argmax token from the last valid position of each sequence.
///
/// `logits` shape: [batch, seq_or_1, vocab_size]
/// `seq_lens`: actual sequence length per batch element (used during prefill).
/// `use_last_only`: if true, logits is [batch, 1, vocab] (decode step).
pub fn argmax_last_token(
    logits: &Tensor,
    seq_lens: &[usize],
    use_last_only: bool,
) -> Vec<u32> {
    let guard = logits.data();
    let batch = logits.shape()[0];
    let seq_dim = logits.shape()[1];
    let vocab = logits.shape()[2];

    let mut tokens = Vec::with_capacity(batch);

    for b in 0..batch {
        let last_pos = if use_last_only {
            0 // logits is [B, 1, V], position 0 is the only one
        } else {
            seq_lens[b] - 1 // last real (non-padded) position
        };

        let offset = b * seq_dim * vocab + last_pos * vocab;
        let tok_logits = &guard[offset..offset + vocab];

        let next_token = tok_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);

        tokens.push(next_token);
    }

    drop(guard);
    tokens
}
