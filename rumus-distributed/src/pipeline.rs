// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Pipeline parallelism: 1F1B micro-batch schedule with per-micro-batch tapes.

use std::sync::mpsc;

use rumus::autograd::{backward_with_grad, context, GradientStore, Tape};
use rumus::tensor::{GradId, Tensor};

// ---------------------------------------------------------------------------
// PipelineStage
// ---------------------------------------------------------------------------

/// A single pipeline stage on a specific device.
pub struct PipelineStage {
    pub device_index: usize,
    pub forward_fn: Box<dyn Fn(&Tensor) -> Tensor + Send + Sync>,
}

// ---------------------------------------------------------------------------
// PipelineExecutor
// ---------------------------------------------------------------------------

/// 1F1B pipeline executor with per-micro-batch isolated tapes.
pub struct PipelineExecutor {
    pub stages: Vec<PipelineStage>,
    pub num_micro_batches: usize,
}

impl PipelineExecutor {
    pub fn new(stages: Vec<PipelineStage>, num_micro_batches: usize) -> Self {
        Self { stages, num_micro_batches }
    }

    /// Run the pipeline.  Returns per-stage gradient stores.
    pub fn run(
        &self,
        input: &Tensor,
        loss_fn: &(dyn Fn(&Tensor) -> Tensor + Send + Sync),
    ) -> Vec<GradientStore> {
        let p = self.stages.len();
        let m = self.num_micro_batches;
        let batch = input.shape()[0];
        assert!(batch % m == 0);
        let micro_size = batch / m;
        let micros: Vec<Tensor> = (0..m)
            .map(|i| input.slice_range(0, i * micro_size, (i + 1) * micro_size))
            .collect();

        // Channels: stage s sends activations to stage s+1 (fwd) and
        // stage s+1 sends gradients back to stage s (bwd).
        let mut fwd_tx_opts: Vec<Option<mpsc::SyncSender<Tensor>>> = Vec::new();
        let mut fwd_rx_opts: Vec<Option<mpsc::Receiver<Tensor>>> = Vec::new();
        let mut bwd_tx_opts: Vec<Option<mpsc::SyncSender<Tensor>>> = Vec::new();
        let mut bwd_rx_opts: Vec<Option<mpsc::Receiver<Tensor>>> = Vec::new();

        // Stage 0: no fwd_rx, no bwd_tx.
        fwd_rx_opts.push(None);
        bwd_tx_opts.push(None);

        for _ in 0..p.saturating_sub(1) {
            let (ftx, frx) = mpsc::sync_channel(m);
            let (btx, brx) = mpsc::sync_channel(m);
            fwd_tx_opts.push(None); // placeholder, will be set below
            fwd_rx_opts.push(Some(frx));
            bwd_tx_opts.push(Some(btx));
            bwd_rx_opts.push(None); // placeholder
            // Fix: assign properly.
            let last_fwd_tx = fwd_tx_opts.len() - 1;
            fwd_tx_opts[last_fwd_tx - 1] = Some(ftx);
            let last_bwd_rx = bwd_rx_opts.len() - 1;
            bwd_rx_opts[last_bwd_rx - 1] = Some(brx);
        }
        // Last stage: no fwd_tx, no bwd_rx.
        // Already None by construction.

        let grad_stores: Vec<std::sync::Mutex<GradientStore>> =
            (0..p).map(|_| std::sync::Mutex::new(GradientStore::new())).collect();

        std::thread::scope(|scope| {
            let micros_ref = &micros;
            let stages_ref = &self.stages;
            let gs_ref = &grad_stores;

            let mut handles = Vec::with_capacity(p);
            for s in 0..p {
                let my_fwd_rx = fwd_rx_opts[s].take();
                let my_fwd_tx = fwd_tx_opts[s].take();
                let my_bwd_rx = bwd_rx_opts[s].take();
                let my_bwd_tx = bwd_tx_opts[s].take();

                handles.push(scope.spawn(move || {
                    run_stage(
                        s, p, m, stages_ref, micros_ref,
                        my_fwd_rx, my_fwd_tx, my_bwd_rx, my_bwd_tx,
                        gs_ref, loss_fn,
                    );
                }));
            }

            for h in handles { h.join().expect("pipeline thread panicked"); }
        });

        grad_stores.into_iter().map(|m| m.into_inner().unwrap()).collect()
    }
}

fn run_stage(
    stage: usize,
    num_stages: usize,
    num_micro: usize,
    stages: &[PipelineStage],
    micros: &[Tensor],
    fwd_rx: Option<mpsc::Receiver<Tensor>>,
    fwd_tx: Option<mpsc::SyncSender<Tensor>>,
    bwd_rx: Option<mpsc::Receiver<Tensor>>,
    bwd_tx: Option<mpsc::SyncSender<Tensor>>,
    grad_stores: &[std::sync::Mutex<GradientStore>],
    loss_fn: &(dyn Fn(&Tensor) -> Tensor + Send + Sync),
) {
    let is_first = stage == 0;
    let is_last = stage == num_stages - 1;

    let mut saved_tapes: Vec<Option<Tape>> = (0..num_micro).map(|_| None).collect();
    let mut saved_outputs: Vec<Option<Tensor>> = (0..num_micro).map(|_| None).collect();
    let mut saved_input_gids: Vec<Option<GradId>> = (0..num_micro).map(|_| None).collect();

    // === Forward all micro-batches ===
    for mb in 0..num_micro {
        let mut input_t = if is_first {
            micros[mb].clone()
        } else {
            fwd_rx.as_ref().unwrap().recv().expect("fwd recv failed")
        };

        // Track incoming tensor for gradient extraction.
        if !is_first {
            input_t.set_requires_grad(true);
            saved_input_gids[mb] = input_t.grad_id();
        }

        // Fresh isolated tape.
        context::install_tape(Tape::new());
        let output = (stages[stage].forward_fn)(&input_t);
        saved_tapes[mb] = context::take_tape();
        saved_outputs[mb] = Some(output.clone());

        if !is_last {
            fwd_tx.as_ref().unwrap().send(output).expect("fwd send failed");
        }
    }

    // === Backward all micro-batches (reverse) ===
    for mb in (0..num_micro).rev() {
        let output = saved_outputs[mb].take().unwrap();

        if is_last {
            // Last stage: loss + standard backward.
            context::install_tape(saved_tapes[mb].take().unwrap());
            let loss = loss_fn(&output);
            let mut grads = rumus::autograd::backward(&loss).expect("backward failed");

            // Send grad_input to prev stage.
            if !is_first {
                if let Some(gid) = saved_input_gids[mb] {
                    if let Some(gi) = grads.remove(gid) {
                        bwd_tx.as_ref().unwrap().send(gi).expect("bwd send failed");
                    }
                }
            }
            grad_stores[stage].lock().unwrap().merge_from(&mut grads);
        } else {
            // Middle/first: receive grad from next, inject into local tape.
            let grad_output = bwd_rx.as_ref().unwrap().recv().expect("bwd recv failed");
            context::install_tape(saved_tapes[mb].take().unwrap());
            let mut grads = backward_with_grad(&output, grad_output).expect("bwd_with_grad failed");

            if !is_first {
                if let Some(gid) = saved_input_gids[mb] {
                    if let Some(gi) = grads.remove(gid) {
                        bwd_tx.as_ref().unwrap().send(gi).expect("bwd send failed");
                    }
                }
            }
            grad_stores[stage].lock().unwrap().merge_from(&mut grads);
        }
    }
}
