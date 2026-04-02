# rumus-macros

Procedural macros for the **RUMUS** deep learning framework.

## `#[derive(Module)]`

Auto-generates the `Module` trait implementation for structs with named fields.

### Generated Methods

| Method | Behavior |
|--------|----------|
| `parameters()` | Concatenates `.parameters()` from every field |
| `train()` | Calls `.train()` on every field (toggles Dropout, etc.) |
| `eval()` | Calls `.eval()` on every field |
| `state_dict(prefix)` | Recursively collects tensors with dot-path keys |
| `load_state_dict(dict, prefix)` | Recursively loads tensors by dot-path |

### Example

```rust
use rumus::nn::{Module, Linear, Dropout};
use rumus::tensor::Tensor;

#[derive(Module)]
struct MLP {
    fc1: Linear,
    drop: Dropout,
    fc2: Linear,
}

impl MLP {
    fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.drop.forward(&rumus::nn::relu(&self.fc1.forward(x)));
        self.fc2.forward(&h)
    }
}

let model = MLP { /* ... */ };

// Auto-generated:
let params = model.parameters();       // collects fc1 + fc2 params
let dict = model.state_dict("");       // "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"
```

### How It Works

The macro iterates all named fields of the struct and generates fully-qualified
calls to `rumus::nn::Module::parameters(...)`, etc. for each field. Any field
type that implements `Module` (including `Parameter`, `Linear`, `Conv2d`,
`MaxPool2d`, `Flatten`, `Dropout`, or user-defined `#[derive(Module)]` structs)
is automatically supported.

`forward()` is **not** generated — it is written as an inherent method by the
user, since different models have different forward signatures.

## Dependencies

- `syn` (with `full` feature) — Rust syntax parsing
- `quote` — token stream generation
- `proc-macro2` — procedural macro utilities

## License

MIT
