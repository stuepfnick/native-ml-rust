use crate::activation::Activation;
use crate::layer_spec::{build_layer_specs, LayerSpec};
use crate::neuron_spec::NeuronSpec;
use rand::Rng;

/// Single `Vec<f32>` parameter buffer + per-layer layout (`LayerSpec`).
pub struct NetworkFlat {
    pub params: Vec<f32>,
    pub layer_specs: Vec<LayerSpec>,
}

impl NetworkFlat {
    pub fn new(layer_sizes: &[usize]) -> Self {
        if layer_sizes.len() < 2 {
            panic!("NetworkFlat must have at least an input size and an output layer");
        }
        let (layer_specs, total_params) = build_layer_specs(layer_sizes);
        println!("NetworkFlat total parameters: {}", total_params);
        let mut rng = rand::thread_rng();
        let params: Vec<f32> = (0..total_params)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        Self {
            params,
            layer_specs,
        }
    }

    pub fn predict(&self, inputs: &[f32]) -> Vec<f32> {
        let mut current = inputs.to_vec();
        for spec in &self.layer_specs {
            current = Self::layer_forward(&self.params, spec, &current);
        }
        current
    }

    fn layer_forward(params: &[f32], spec: &LayerSpec, inputs: &[f32]) -> Vec<f32> {
        debug_assert_eq!(inputs.len(), spec.in_features);
        let mut out = Vec::with_capacity(spec.out_features);
        for j in 0..spec.out_features {
            let n = NeuronSpec::from_layer(spec, j);
            let mut sum = params[n.bias_index()];
            for i in 0..spec.in_features {
                sum += params[n.param_start + i] * inputs[i];
            }
            out.push(Activation::sigmoid(sum));
        }
        out
    }

    fn forward(&self, inputs: &[f32]) -> Vec<Vec<f32>> {
        let mut activations = Vec::with_capacity(self.layer_specs.len() + 1);
        let mut current = inputs.to_vec();
        activations.push(current.clone());
        for spec in &self.layer_specs {
            current = Self::layer_forward(&self.params, spec, &current);
            activations.push(current.clone());
        }
        activations
    }

    pub fn train(&mut self, inputs: &[f32], targets: &[f32], learning_rate: f32) {
        let activations = self.forward(inputs);
        self.backpropagate(activations, targets, learning_rate);
    }

    fn backpropagate(
        &mut self,
        activations: Vec<Vec<f32>>,
        targets: &[f32],
        learning_rate: f32,
    ) {
        let last_output = activations.last().expect("no activations");
        let mut current_errors: Vec<f32> = targets
            .iter()
            .zip(last_output.iter())
            .map(|(t, o)| t - o)
            .collect();

        for i in (0..self.layer_specs.len()).rev() {
            let spec = &self.layer_specs[i];
            let layer_inputs = &activations[i];
            let layer_outputs = &activations[i + 1];
            current_errors = Self::flat_layer_update(
                &mut self.params,
                spec,
                layer_inputs,
                layer_outputs,
                &current_errors,
                learning_rate,
            );
        }
    }

    /// Same update rule as `Layer::update`, writing into `params` via `NeuronSpec` indices.
    fn flat_layer_update(
        params: &mut [f32],
        spec: &LayerSpec,
        inputs: &[f32],
        outputs: &[f32],
        current_errors: &[f32],
        learning_rate: f32,
    ) -> Vec<f32> {
        let inn = spec.in_features;
        let mut next_errors = vec![0.0_f32; inn];

        for j in 0..spec.out_features {
            let n = NeuronSpec::from_layer(spec, j);
            let gradient = current_errors[j] * Activation::sigmoid_derivative(outputs[j]);

            for i in 0..inn {
                next_errors[i] += gradient * params[n.param_start + i];
            }
            for i in 0..inn {
                params[n.param_start + i] += learning_rate * gradient * inputs[i];
            }
            params[n.bias_index()] += learning_rate * gradient;
        }
        next_errors
    }

    pub fn print(&self) {
        println!(
            "NetworkFlat: {} params, {} layer specs",
            self.params.len(),
            self.layer_specs.len(),
        );
        if let Some(layer) = self.layer_specs.first() {
            let n0 = NeuronSpec::from_layer(layer, 0);
            println!(
                "  neuron 0 (layer 0): weights params[{}..{})  ({} values), bias at index {}, block len {}",
                n0.param_start,
                n0.param_start + n0.in_features,
                n0.in_features,
                n0.bias_index(),
                n0.param_len(),
            );
            debug_assert_eq!(NeuronSpec::iter_for_layer(layer).count(), layer.out_features);
        }
    }

    pub fn visualize(&self) {
        println!("\n--- Decision Landscape (NetworkFlat) ---");

        for y_step in (0..=10).rev() {
            let y = y_step as f32 / 10.0;
            print!("{:.1} | ", y);

            for x_step in 0..=20 {
                let x = x_step as f32 / 20.0;
                let output = self.predict(&[x, y])[0];
                let symbol = match output {
                    v if v > 0.9 => "█",
                    v if v > 0.7 => "▓",
                    v if v > 0.4 => "▒",
                    v if v > 0.1 => "░",
                    _ => " ",
                };
                print!("{}", symbol);
            }
            println!();
        }
        println!("    +----------------------");
        println!("     0.0       0.5       1.0  (x)");
    }
}
