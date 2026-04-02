use crate::Neuron;

/// A layer of neurons in a neural network.
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {

    /// Create a new layer with the specified number of neurons and inputs.
    ///
    /// # Arguments
    /// * `num_neurons` - The number of neurons in the layer.
    /// * `num_inputs` - The number of inputs to each neuron, which determines the number of weights.
    /// # Returns
    /// * A new instance of Layer with the specified number of neurons, each initialized
    pub fn new(num_neurons: usize, num_inputs: usize) -> Self {
        let neurons = (0..num_neurons)
            .map(|_| Neuron::new(num_inputs))
            .collect();
        Self { neurons }
    }

    /// Predict the output of the layer given a set of inputs.
    ///
    /// # Arguments
    /// * `inputs` - A slice of f32 representing the input values.
    /// # Returns
    /// * A vector of f32 representing the output values of each neuron in the layer.
    pub fn predict(&self, inputs: &[f32]) -> Vec<f32> {
        self.neurons.iter().map(|n| n.predict(inputs)).collect::<Vec<f32>>()
    }

    /// Train the layer using a single training example.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A slice of f32 representing the input values.
    /// * `targets` - A slice of f32 representing the target values for each neuron in the layer.
    /// * `learning_rate` - A f32 representing the learning rate.
    pub fn train(&mut self, inputs: &[f32], targets: &[f32], learning_rate: f32) {
        for (neuron, &target) in self.neurons.iter_mut().zip(targets.iter()) {
            neuron.train(inputs, target, learning_rate);
        }
    }
}