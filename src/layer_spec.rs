/// Metadata for one weight matrix layer: inputs → outputs, stored flat as
/// `out_features` blocks of `in_features + 1` floats (weights then bias per neuron).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerSpec {
    pub in_features: usize,
    pub out_features: usize,
    pub weight_start: usize,
}

pub fn build_layer_specs(layer_sizes: &[usize]) -> (Vec<LayerSpec>, usize) {
    if layer_sizes.len() < 2 {
        panic!("layer_sizes must contain at least input and output width");
    }

    let mut cursor = 0usize;
    let mut specs = Vec::with_capacity(layer_sizes.len() - 1);

    for w in layer_sizes.windows(2) {
        let in_features = w[0];
        let out_features = w[1];
        let block = out_features * (in_features + 1);

        specs.push(LayerSpec {
            in_features,
            out_features,
            weight_start: cursor,
        });
        cursor += block;
    }

    (specs, cursor)
}
