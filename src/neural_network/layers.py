import numpy as np

class DenseLayer:
    def __init__(self, n_inputs, n_neurons, weight_scale=0.01, seed=42):
        
        np.random.RandomState(seed)
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        
        self.weights = np.random.rand(n_neurons, n_inputs) * weight_scale
        self.biases = np.zeros((n_neurons, 1))
        
        # Para el backpropagation
        self.X_cache = None
        self.dW = np.zeros((n_neurons, n_inputs))
        self.db = np.zeros((n_neurons, 1))
        
    def forward(self, n_inputs):
        # falta validacion
        self.output = np.dot(self.weights, n_inputs) + self.biases
        
