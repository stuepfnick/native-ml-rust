![Build Status](https://github.com/stuepfnick/native-ml-rust/actions/workflows/rust.yml/badge.svg)

# native-ml-rust

A neural network built from scratch in **Rust**, optimized for **Apple Silicon (M1 Pro)**.

## Features

- **Neuron**: Single neuron with configurable inputs, sigmoid activation, and gradient descent training.
- **Layer**: Multiple neurons managed as a logical unit, trainable on multiple targets simultaneously.
- **Network**: Multi-layer neural network with forward pass and backpropagation.
- **Activation**: Separate module for activation functions (sigmoid and its derivative).
- **CLI Support**: Control mode, iterations, and learning rate via command-line arguments.

## Usage

All arguments are optional. If the first argument is a number, it's treated as `iterations` (network mode is used). If it's a word, it's the `mode`.

```
cargo run [-- [mode] [iterations] [learning_rate]]
```

| Argument        | Type    | Default     | Options                    |
|-----------------|---------|-------------|----------------------------|
| `mode`          | string  | `network`   | `network`, `layer`, `neuron` |
| `iterations`    | integer | `10000`     | any positive number        |
| `learning_rate` | float   | `0.1`       | e.g. `0.5`                 |

```bash
cargo run                          # network, 10000 iterations, lr=0.1
cargo run -- 5000                  # network, 5000 iterations, lr=0.1
cargo run -- 5000 0.5              # network, 5000 iterations, lr=0.5
cargo run -- layer                 # layer,   10000 iterations, lr=0.1
cargo run -- layer 2000 0.3        # layer,   2000  iterations, lr=0.3
cargo run -- neuron 1000           # neuron,  1000  iterations, lr=0.1
cargo run -- network 5000 0.5      # network, 5000  iterations, lr=0.5
```

## Example

The default mode trains a **2-2-1 network** to learn **XOR**:

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
├── main.rs        # Entry point, training and tests
├── network.rs     # Network struct with backpropagation
├── layer.rs       # Layer struct (multiple neurons)
├── neuron.rs      # Neuron with sigmoid and training
└── activation.rs  # Activation functions (sigmoid)
```

## Dependencies

- [`rand`](https://crates.io/crates/rand) – random weight initialization
