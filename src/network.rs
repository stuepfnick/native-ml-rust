use crate::layer::Layer;

pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {

    /// Create a new network with the specified layer sizes.
    ///
    /// # Arguments
    /// * `layer_sizes` - A slice of usize representing the number of neurons in each layer.
    ///                   The first element is the input size, the last element is the output size,
    ///                   and the elements in between are the sizes of the hidden layers.
    pub fn new(layer_sizes: &[usize]) -> Self {
        let layers = layer_sizes
            .windows(2)
            .map(|w| Layer::new(w[0], w[1]))
            .collect();
        Self { layers }
    }

    /// Forward pass through the network
    ///
    /// # Arguments
    /// * `inputs` - A slice of f32 representing the input values
    /// 
    /// # Returns
    /// A vector of f32 representing the output values.
    pub fn predict(&self, inputs: &[f32]) -> Vec<f32> {
        let mut output = inputs.to_vec();
        for layer in &self.layers {
            output = layer.predict(&output);
        }
        output
    }
    /// Forward pass through the network, returning activations of all layers
    ///
    /// # Arguments
    /// * `inputs` - A slice of f32 representing the input values.
    /// 
    /// # Returns
    /// A vector of vectors of f32 representing the activations of each layer, including the input layer as the first element.
    fn forward(&self, inputs: &[f32]) -> Vec<Vec<f32>> {
        let mut activations = vec![inputs.to_vec()]; // Start with the input layer as the first activation
        let mut current = inputs.to_vec();

        for layer in &self.layers {
            current = layer.predict(&current);
            activations.push(current.clone());
        }

        activations
    }

    pub fn train(&mut self, inputs: &[f32], targets: &[f32], learning_rate: f32) {
        // Perform a forward pass to get the activations of all layers
        let activations = self.forward(inputs);

        // Start backpropagation from the output layer and move backwards through the layers
        self.backpropagate(activations, targets, learning_rate);
    }

    fn backpropagate(&mut self, activations: Vec<Vec<f32>>, targets: &[f32], learning_rate: f32) {
        let last_output = activations.last().expect("No activations found");
        let mut current_errors: Vec<f32> = targets.iter()
            .zip(last_output.iter())
            .map(|(t, o)| t - o)// target - output
            .collect();

        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            let inputs = &activations[i]; // Inputs to the current layer are the activations of the previous layer
            current_errors = layer.update(inputs, &current_errors, learning_rate);
        }
    }

    pub fn print(&self) {
        println!("Network: {} Layers", self.layers.len());
        self.layers.iter().for_each(|l| l.print());
    }
}