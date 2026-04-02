use crate::FlexibleNeuron;

pub struct Layer {
    pub neurons: Vec<FlexibleNeuron>,
}

impl Layer {
    pub fn new(num_neurons: usize, num_inputs: usize) -> Self {
        let neurons = (0..num_neurons)
            .map(|_| FlexibleNeuron::new(num_inputs))
            .collect();
        Self { neurons }
    }

    pub fn predict(&self, inputs: &[f32]) -> Vec<f32> {
        self.neurons.iter().map(|n| n.predict(inputs)).collect()
    }

    pub fn train(&mut self, inputs: &[f32], targets: &[f32], learning_rate: f32) {
        for (neuron, &target) in self.neurons.iter_mut().zip(targets.iter()) {
            neuron.train(inputs, target, learning_rate);
        }
    }
}