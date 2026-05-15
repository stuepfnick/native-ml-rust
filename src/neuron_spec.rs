use crate::layer_spec::LayerSpec;

/// One output neuron’s block in the flat `params` buffer: `in_features` weights, then one bias.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NeuronSpec {
    pub param_start: usize,
    pub in_features: usize,
}

impl NeuronSpec {
    /// `neuron_index` in `0 .. layer.out_features`.
    pub fn from_layer(layer: &LayerSpec, neuron_index: usize) -> Self {
        assert!(
            neuron_index < layer.out_features,
            "neuron_index {} out of range for out_features {}",
            neuron_index,
            layer.out_features
        );
        let stride = layer.in_features + 1;
        Self {
            param_start: layer.weight_start + neuron_index * stride,
            in_features: layer.in_features,
        }
    }

    /// All output neurons in `layer`, left-to-right in the flat layout.
    pub fn iter_for_layer(layer: &LayerSpec) -> impl Iterator<Item = NeuronSpec> + '_ {
        let in_features = layer.in_features;
        let stride = in_features + 1;
        let start = layer.weight_start;
        let out_features = layer.out_features;
        (0..out_features).map(move |j| NeuronSpec {
            param_start: start + j * stride,
            in_features,
        })
    }

    pub const fn param_len(self) -> usize {
        self.in_features + 1
    }

    pub const fn bias_index(self) -> usize {
        self.param_start + self.in_features
    }
}
