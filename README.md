![Build Status](https://github.com/stuepfnick/native-ml-rust/actions/workflows/rust.yml/badge.svg)

# native-ml-rust

A neural network built from scratch in **Rust**, optimized for **Apple Silicon (M1 Pro)**.

## Features

- **FlexibleNeuron**: Single neurons with configurable inputs, sigmoid activation, and backpropagation training.
- **Layer**: Multiple neurons managed as a logical unit, trainable on multiple targets simultaneously.
- **CLI Support**: Control iterations and mode (neuron/layer) via command-line arguments.

## Usage

```bash
# Run in layer mode with 5000 iterations
cargo run -- l 5000

# Run in neuron mode with default iterations
cargo run -- n
```

## Example

The project trains a 2-neuron layer to learn the logical functions **AND** and **OR**:

| Input        | AND | OR |
|--------------|-----|----|
| [0.0, 0.0]   | 0   | 0  |
| [0.0, 1.0]   | 0   | 1  |
| [1.0, 0.0]   | 0   | 1  |
| [1.0, 1.0]   | 1   | 1  |

## Project Structure

```
src/
├── main.rs      # Entry point, training and tests
├── layer.rs     # Layer struct (multiple neurons)
└── neuron.rs    # FlexibleNeuron with sigmoid and training
```

## Dependencies

- [`rand`](https://crates.io/crates/rand) – random weight initialization
