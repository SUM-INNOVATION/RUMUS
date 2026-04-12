// SPDX-License-Identifier: Apache-2.0 OR MIT
//! RUMUS Inference Server — high-throughput LLM serving with continuous batching.

mod gpu_worker;
mod model;
mod scheduler;
mod server;

use std::sync::Arc;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    let port: u16 = args
        .iter()
        .position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);

    let max_batch: usize = args
        .iter()
        .position(|a| a == "--max-batch")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    let batch_timeout_ms: u64 = args
        .iter()
        .position(|a| a == "--batch-timeout-ms")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let vocab_size: usize = args
        .iter()
        .position(|a| a == "--vocab-size")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(32000);

    let hidden_dim: usize = args
        .iter()
        .position(|a| a == "--hidden-dim")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);

    let num_heads: usize = args
        .iter()
        .position(|a| a == "--num-heads")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let num_layers: usize = args
        .iter()
        .position(|a| a == "--num-layers")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    eprintln!("rumus-serve starting...");
    eprintln!("  port:             {}", port);
    eprintln!("  max_batch:        {}", max_batch);
    eprintln!("  batch_timeout_ms: {}", batch_timeout_ms);
    eprintln!("  vocab_size:       {}", vocab_size);
    eprintln!("  hidden_dim:       {}", hidden_dim);
    eprintln!("  num_heads:        {}", num_heads);
    eprintln!("  num_layers:       {}", num_layers);

    // Bounded channel: HTTP handlers → GPU worker.
    let (tx, rx) = tokio::sync::mpsc::channel::<server::InferenceRequest>(1024);

    // Spawn the GPU worker on a dedicated OS thread.
    let worker_config = gpu_worker::WorkerConfig {
        max_batch_size: max_batch,
        batch_timeout: std::time::Duration::from_millis(batch_timeout_ms),
        vocab_size,
        hidden_dim,
        num_heads,
        num_layers,
    };

    std::thread::spawn(move || {
        gpu_worker::run(rx, worker_config);
    });

    // Start the HTTP server.
    let state = Arc::new(server::AppState { tx });
    let app = server::build_router(state);

    let addr = format!("0.0.0.0:{}", port);
    eprintln!("  listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
