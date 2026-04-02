use rand::Rng;

/// A flexible neuron that can be used in a neural network.
pub struct Neuron {
    pub weights: Vec<f32>,
    pub bias: f32,
}

impl Neuron {

    /// Create a new neuron with random weights and bias.
    /// # Arguments
    /// * `n` - The number of inputs to the neuron, which determines the number of weights.
    /// # Returns
    /// * A new instance of Neuron with random weights and bias.
    pub fn new(n: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

        Self {
            weights,
            bias: rng.gen_range(-0.5..0.5),
        }
    }

    /// Predict the output of the neuron given a set of inputs.
    ///
    /// # Arguments
    /// * `inputs` - A slice of f32 representing the input values.
    ///
    /// # Returns
    /// * A f32 representing the output value after applying the activation function.
    pub fn predict(&self, inputs: &[f32]) -> f32 {
        let mut sum = self.bias;
        for i in 0..self.weights.len() {
            sum += inputs[i] * self.weights[i];
        }
        Self::sigmoid(sum)
    }

    /// Train the neuron using a single training example.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A slice of f32 representing the input values.
    /// * `target` - A f32 representing the target value.
    /// * `learning_rate` - A f32 representing the learning rate.
    pub fn train(&mut self, inputs: &[f32], target: f32, learning_rate: f32) {
        // Aktuelle Vorhersage berechnen
        let prediction = self.predict(inputs);
        // Fehler berechnen
        let error = target - prediction;

        // Ableitung der Sigmoid-Funktion;
        let gradient = error * (prediction * (1.0 - prediction)); 

        // Gewichte anpassen
        for i in 0..self.weights.len() {
            self.weights[i] += learning_rate * gradient * inputs[i];
        }
        // Bias (Hemmschwelle) anpassen
        self.bias += learning_rate * gradient;
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

}
