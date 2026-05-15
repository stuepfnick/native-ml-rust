![Build Status](https://github.com/stuepfnick/native-ml-rust/actions/workflows/rust.yml/badge.svg)

# native-ml-rust

A neural network built from scratch in **Rust**, optimized for **Apple Silicon (M1 Pro)**.

## Features

- **Neuron**: Single neuron with configurable inputs, sigmoid activation, and gradient descent training.
- **Layer**: Multiple neurons managed as a logical unit, trainable on multiple targets simultaneously.
- **Network**: Multi-layer neural network with forward pass and backpropagation (legacy `Vec<Layer>`).
- **NetworkFlat**: Default CLI mode; same XOR task as legacy `network`, but weights/biases live in one `Vec<f32>` with `LayerSpec` / `NeuronSpec` indexing and full training on that buffer.
- **Activation**: Separate module for activation functions (sigmoid and its derivative).
- **CLI Support**: Control mode, iterations, and learning rate via command-line arguments.

## Usage

All arguments are optional. If the first argument is a number, it's treated as `iterations` (default mode `flat` is used). If it's a word, it's the `mode`.

```
cargo run [-- [mode] [iterations] [learning_rate]]
```

| Argument        | Type    | Default     | Options                    |
|-----------------|---------|-------------|----------------------------|
| `mode`          | string  | `flat`      | `flat`, `network`, `layer`, `neuron` |
| `iterations`    | integer | `10000`     | any positive number        |
| `learning_rate` | float   | `0.1`       | e.g. `0.5`                 |

```bash
cargo run                          # flat (default), 10000 iterations, lr=0.1
cargo run -- 5000                  # flat, 5000 iterations, lr=0.1
cargo run -- 5000 0.5              # flat, 5000 iterations, lr=0.5
cargo run -- layer                 # layer,   10000 iterations, lr=0.1
cargo run -- layer 2000 0.3        # layer,   2000  iterations, lr=0.3
cargo run -- neuron 1000           # neuron,  1000  iterations, lr=0.1
cargo run -- network 5000 0.5      # network, 5000  iterations, lr=0.5
cargo run -- flat                  # same as default `cargo run` (explicit mode)
cargo run -- network               # legacy Vec<Layer> path
```

## Example

The default mode trains a **[2, 3, 1] network** (2 inputs, 3 hidden, 1 output) to learn **XOR**:

| Input        | XOR |
|--------------|-----|
| [0.0, 0.0]   | 0   |
| [0.0, 1.0]   | 1   |
| [1.0, 0.0]   | 1   |
| [1.0, 1.0]   | 0   |

XOR cannot be solved by a single neuron — it requires a hidden layer and backpropagation.

## Project Structure

```
src/
├── main.rs          # Entry point, training and tests
├── network.rs       # Legacy Network (Vec<Layer>) + backprop
├── network_flat.rs  # Flat params + LayerSpec training
├── layer_spec.rs    # Per-layer offsets into params
├── neuron_spec.rs   # Per-neuron offsets into params
├── layer.rs         # Layer struct (multiple neurons)
├── neuron.rs        # Neuron with sigmoid and training
└── activation.rs    # Activation functions (sigmoid)
```

## Dependencies

- [`rand`](https://crates.io/crates/rand) – random weight initialization
