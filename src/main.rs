mod activation; // Deklares the module for activation functions, which contains the implementation of the Activation struct and its method
mod neuron; // Deklares the module for the neuron, which contains the implementation of the FlexibleNeuron struct and its methods
mod layer; // Deklares the module for the layer, which contains the implementation of the Layer struct and its methods
mod network; // Deklares the module for the network, which contains the implementation of the Network struct and its methods

use {layer::Layer, neuron::Neuron, network::Network}; // Imports the struct for convenient usage in main

fn main() {

    let args: Vec<String> = std::env::args().collect();

    // set default values
    let mut mode = "network".to_string(); // Default mode is "network"
    let mut iterations = 10000; // Default number of iterations for training
    let mut learning_rate = 0.1; // Default learning rate for training

        if args.len() > 1
        {
            if let Ok(i) = args[1].parse::<usize>() {
                iterations = i;
                if args.len() > 2 {
                    learning_rate = args[2].parse().unwrap_or(learning_rate);
                }
            } else {
                mode = args[1].to_lowercase();
                if args.len() > 2 {
                    iterations = args[2].parse().unwrap_or(iterations);
                }
                if args.len() > 3 {
                    learning_rate = args[3].parse().unwrap_or(learning_rate);
                }
            }
        }

     println!("Mode: {}, Iterations: {}, Learning Rate: {}", mode, iterations, learning_rate);      

    if mode == "neuron" {
        train_neuron(iterations, learning_rate);
    } else if mode == "layer" {
        train_layer(iterations, learning_rate);
    } else {
        train_network(iterations, learning_rate);
    }
}

fn train_network(iterations: usize, learning_rate: f32) {
    println!("Training a network...");

    let mut my_network = Network::new(&[2, 2, 1]); // Creates a network with 2 input neurons, 1 hidden layer with 3 neurons, and 1 output neuron

    test_network(&my_network); // Tests the network before training to see initial predictions

    let training_data = [
        ([0.0, 0.0], [0.0]), // XOR: 0
        ([0.0, 1.0], [1.0]), // XOR: 1
        ([1.0, 0.0], [1.0]), // XOR: 1
        ([1.0, 1.0], [0.0]), // XOR: 0
    ];

    println!("Training the network ({} iterations)...", iterations);

    for _ in 0..iterations {
        for (inputs, targets) in training_data.iter() {
            my_network.train(inputs, targets, learning_rate);
        }
    }

    test_network(&my_network); // Tests the network after training to see how predictions have improved
}

fn test_network(my_network: &Network) {
    let test_inputs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ];
    my_network.print();

    for inputs in test_inputs.iter() {
        let outputs = my_network.predict(inputs);
        println!("Input: {:?} => Output: {:.4}", inputs, outputs[0]);
    }
}

fn train_layer(iterations: usize, learning_rate: f32) {
    println!("Training a layer of neurons...");

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
            my_layer.train(inputs, targets, learning_rate);
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
    my_layer.print(); // Prints the weights and biases of each neuron in the layer

    for inputs in test_inputs.iter() {
        let outputs = my_layer.predict(inputs);
        println!("Input: {:?} => AND: {:.4}, OR: {:.4}", inputs, outputs[0], outputs[1]);
    }
}

fn train_neuron(iterations: usize, learning_rate: f32) {
    println!("Training a single neuron...");

    let mut my_neuron = Neuron::new(2); // Creates a neuron with 2 inputs

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
            my_neuron.train(inputs, *target, learning_rate);
        }
    }

    test_neuron(&my_neuron); // Tests the neuron after training to see how predictions have improved
}

fn test_neuron(my_neuron: &Neuron) {
    let test_inputs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ];
    my_neuron.print();

    for inputs in test_inputs.iter() {
        let output = my_neuron.predict(inputs);
        println!("Input: {:?} => Output: {:.4}", inputs, output);
    }
}