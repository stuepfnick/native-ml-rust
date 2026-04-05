pub struct Activation;

impl Activation {
    /// Sigmoid activation function
    ///
    /// # Arguments
    /// * `x` - A f32 representing the input value.
    ///
    /// # Returns
    /// A f32 representing the output value after applying the sigmoid function.
    pub fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    
    /// Derivative of the sigmoid function
    /// 
    /// # Arguments
    /// * `output` - A f32 representing the output value of the sigmoid function (i.e., the predicted value).
    /// 
    /// # Returns
    /// A f32 representing the derivative of the sigmoid function, which is used in backprop
    pub fn sigmoid_derivative(output: f32) -> f32 {
        output * (1.0 - output)
    }
}