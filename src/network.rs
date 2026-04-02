use crate::layer::Layer;

pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {

    /// Create a new network with the specified layer sizes.
    ///
    /// # Arguments
    ///
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
    ///
    /// * `inputs` - A slice of f32 representing the input values.
    ///
    /// # Returns
    ///
    /// A vector of f32 representing the output values.
    pub fn predict(&self, inputs: &[f32]) -> Vec<f32> {
        let mut output = inputs.to_vec();
        for layer in &self.layers {
            output = layer.predict(&output);
        }
        output
    }

}