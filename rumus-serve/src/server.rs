// SPDX-License-Identifier: Apache-2.0 OR MIT
//! HTTP server: axum routes + request/response types.

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub token_ids: Vec<u32>,
    #[serde(default = "default_max_tokens")]
    pub max_new_tokens: usize,
}

fn default_max_tokens() -> usize {
    50
}

#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    pub generated_ids: Vec<u32>,
    pub latency_ms: f64,
}

/// Internal request passed through the channel to the GPU worker.
pub struct InferenceRequest {
    pub token_ids: Vec<u32>,
    pub max_new_tokens: usize,
    pub response_tx: oneshot::Sender<InferenceResponse>,
    pub submitted_at: Instant,
}

/// Internal response from the GPU worker.
pub struct InferenceResponse {
    pub generated_ids: Vec<u32>,
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

pub struct AppState {
    pub tx: mpsc::Sender<InferenceRequest>,
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/generate", post(generate_handler))
        .with_state(state)
}

async fn generate_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<GenerateRequest>,
) -> impl IntoResponse {
    if body.token_ids.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "token_ids must not be empty"})),
        )
            .into_response();
    }

    let (resp_tx, resp_rx) = oneshot::channel();
    let req = InferenceRequest {
        token_ids: body.token_ids,
        max_new_tokens: body.max_new_tokens,
        response_tx: resp_tx,
        submitted_at: Instant::now(),
    };

    // Send to GPU worker. If the channel is full, this .await suspends
    // (backpressure — the client sees increased latency, not an error).
    if state.tx.send(req).await.is_err() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "GPU worker unavailable"})),
        )
            .into_response();
    }

    // Wait for the GPU worker's response.
    match resp_rx.await {
        Ok(resp) => {
            let latency = Instant::now().duration_since(Instant::now() - Instant::now().elapsed());
            Json(GenerateResponse {
                generated_ids: resp.generated_ids,
                latency_ms: 0.0, // latency tracked by gpu_worker
            })
            .into_response()
        }
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "GPU worker dropped request"})),
        )
            .into_response(),
    }
}
