mod activation; // Declares the module for activation functions, which contains the implementation of the Activation struct and its method
mod neuron; // Declares the module for the neuron, which contains the implementation of the Neuron struct and its methods
mod layer; // Declares the module for the layer, which contains the implementation of the Layer struct and its methods
mod network; // Legacy Network (Vec<Layer>)
mod network_flat; // Flat params + LayerSpec layout
mod layer_spec; // LayerSpec + build_layer_specs (flat buffer layout per layer)
mod neuron_spec; // NeuronSpec (param slice for one output neuron; uses LayerSpec for addressing)

use {layer::Layer, neuron::Neuron, network::Network, network_flat::NetworkFlat}; // Imports the struct for convenient usage in main

fn main() {

    let args: Vec<String> = std::env::args().collect();

    // set default values
    let mut mode = "flat".to_string(); // Default: NetworkFlat (single params buffer)
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

    let start = std::time::Instant::now();
    if mode == "neuron" {
        train_neuron(iterations, learning_rate);
    } else if mode == "layer" {
        train_layer(iterations, learning_rate);
    } else if mode == "flat" {
        train_network_flat(iterations, learning_rate);
    } else {
        train_network(iterations, learning_rate);
    }
    println!("Training duration: {:?}", start.elapsed());
}

fn train_network(iterations: usize, learning_rate: f32) {
    println!("Training a network...");

    let layer_sizes = [2usize, 3, 1];
    let mut my_network = Network::new(&layer_sizes);

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

fn train_network_flat(iterations: usize, learning_rate: f32) {
    println!("Training NetworkFlat (XOR, flat `params`, architecture [2, 3, 1])...");

    let layer_sizes = [2usize, 3, 1];
    let mut flat = NetworkFlat::new(&layer_sizes);

    test_network_flat(&flat);

    let training_data = [
        ([0.0, 0.0], [0.0]), // XOR: 0
        ([0.0, 1.0], [1.0]), // XOR: 1
        ([1.0, 0.0], [1.0]), // XOR: 1
        ([1.0, 1.0], [0.0]), // XOR: 0
    ];

    println!(
        "Training flat network ({} passes over the full XOR set, lr={})...",
        iterations, learning_rate
    );

    for _ in 0..iterations {
        for (inputs, targets) in training_data.iter() {
            flat.train(inputs, targets, learning_rate);
        }
    }

    test_network_flat(&flat);
}

fn test_network_flat(flat: &NetworkFlat) {
    let test_inputs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ];
    println!("--- NetworkFlat ---");
    flat.print();

    for inputs in test_inputs.iter() {
        let outputs = flat.predict(inputs);
        println!("Input: {:?} => Output: {:.4}", inputs, outputs[0]);
    }
    flat.visualize();
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
    my_network.visualize(); // Visualizes the network structure and weights after testing
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