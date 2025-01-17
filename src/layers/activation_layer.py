from .base_layer import Layer

# inherit from base class Layer
class ActivationLayer(Layer):
    def __init__(self, activation_function):
        self.activation = activation_function
        self.activation_prime = activation_function.prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error