// Embedding forward: pure memory lookup.
//
// Each thread copies one element from the weight matrix, indexed by the
// token ID from the indices buffer.

struct EmbeddingParams {
    total_lookups: u32,
    embed_dim: u32,
    _pad0: u32,
    _pad1: u32,
}
// 16 bytes ✓

@group(0) @binding(0) var<storage, read>       emb_indices: array<f32>; // token IDs as f32
@group(0) @binding(1) var<storage, read>       emb_weight:  array<f32>; // [vocab, dim]
@group(0) @binding(2) var<storage, read_write> emb_output:  array<f32>; // [lookups, dim]
@group(0) @binding(3) var<uniform>             emb_params:  EmbeddingParams;

@compute @workgroup_size(64)
fn embedding_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = emb_params.total_lookups * emb_params.embed_dim;
    if (idx >= total) { return; }

    let lookup = idx / emb_params.embed_dim;
    let dim = idx % emb_params.embed_dim;
    let token_id = u32(emb_indices[lookup]);

    emb_output[idx] = emb_weight[token_id * emb_params.embed_dim + dim];
}
