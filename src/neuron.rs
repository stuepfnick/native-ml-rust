use rand::Rng;

pub struct FlexibleNeuron {
    pub weights: Vec<f32>,
    pub bias: f32,
}

impl FlexibleNeuron {
    pub fn new(n: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

        Self {
            weights,
            bias: rng.gen_range(-0.5..0.5),
        }
    }

    pub fn predict(&self, inputs: &[f32]) -> f32 {
        let mut sum = self.bias;
        for i in 0..self.weights.len() {
            sum += inputs[i] * self.weights[i];
        }
        Self::sigmoid(sum)
    }

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
