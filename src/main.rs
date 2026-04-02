mod neuron; // Deklares the module for the neuron, which contains the implementation of the FlexibleNeuron struct and its methods
mod layer; // Deklares the module for the layer, which contains the implementation of the Layer struct and its methods

use layer::Layer; // Imports the struct for convenient usage in main
use neuron::FlexibleNeuron; // Imports the struct for convenient usage in main

fn main() {
    
    let args: Vec<String> = std::env::args().collect();

    let iterations = if args.len() > 2 {
        args[2].parse::<usize>().unwrap_or(100)
    } else if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(100)
    }
    else
    {
        1000
    };

    let mode = if args.len() > 2 {
        &args[1].to_lowercase()
    } else {
        "layer" // Default mode is "layer"
    };

    if mode == "neuron" {
        println!("Training a single neuron...");
        train_neuron(iterations); // Trains the neuron with the specified number of iterations
    } else {
        println!("Training a layer of neurons...");
        train_layer(iterations); // Trains the layer with the specified number of iterations
    }
}

fn train_layer(iterations: usize) {
    let mut my_layer = Layer::new(2, 2); // Creates a layer with 2 neurons, each expecting 2 inputs

    test_layer(&my_layer); // Tests the layer before training to see initial predictions

    let training_data = [
        ([0.0, 0.0], [0.0, 0.0]), // AND: 0, OR: 0
        ([0.0, 1.0], [0.0, 1.0]), // AND: 0, OR: 1
        ([1.0, 0.0], [0.0, 1.0]), // AND: 0, OR: 1
        ([1.0, 1.0], [1.0, 1.0]), // AND: 1, OR: 1
    ];

    println!("Training the layer ({} iterations)...", iterations);

    for _ in 0..iterations {
        for (inputs, targets) in training_data.iter() {
            my_layer.train(inputs, targets, 0.1);
        }
    }

    test_layer(&my_layer); // Tests the layer after training to see how predictions have improved
}

fn test_layer(my_layer: &Layer) {
    let test_inputs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ];
    println!("Layer: Neuron Weights and Biases:");
    for (i, neuron) in my_layer.neurons.iter().enumerate() {
        println!(" Neuron {}: Weights: {:?}, Bias: {:.4}", i + 1, neuron.weights, neuron.bias);
    }

    for inputs in test_inputs.iter() {
        let outputs = my_layer.predict(inputs);
        println!("Input: {:?} => AND: {:.4}, OR: {:.4}", inputs, outputs[0], outputs[1]);
    }
}

fn train_neuron(iterations: usize) {
    let mut my_neuron = FlexibleNeuron::new(2); // Creates a neuron with 2 inputs

    test_neuron(&my_neuron); // Tests the neuron before training to see initial predictions

    let training_inputs = [
        ([0.0, 0.0], 0.0), // OR: 0
        ([0.0, 1.0], 1.0), // OR: 1
        ([1.0, 0.0], 1.0), // OR: 1
        ([1.0, 1.0], 1.0), // OR: 1
    ];

    println!("Training the neuron ({} iterations)...", iterations);

    for _ in 0..iterations {
        for (inputs, target) in training_inputs.iter() {
            my_neuron.train(inputs, *target, 0.1);
        }
    }

    test_neuron(&my_neuron); // Tests the neuron after training to see how predictions have improved
}

fn test_neuron(my_neuron: &FlexibleNeuron) {
    let test_inputs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ];
    println!("Neuron: Weights: {:?}, Bias: {:.4}", my_neuron.weights, my_neuron.bias);

    for inputs in test_inputs.iter() {
        let output = my_neuron.predict(inputs);
        println!("Input: {:?} => Output: {:.4}", inputs, output);
    }
}