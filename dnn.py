import numpy as np

# Dense Layer
class DenseLayer:
    def __init__(self, neurons, input_size=None,  activation=None, dropout_rate=0):
        self.weights = None
        self.biases = None
        self.input_size = input_size
        self.neurons = neurons
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dropout_mask = None

        if activation == 'relu':
            self.activation_func = self.relu
        elif self.activation == 'sigmoid':
            self.activation_func = self.sigmoid

    def forward(self, inputs):
        self.inputs = inputs
        self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=inputs.shape)
        self.dropout_inputs = inputs * self.dropout_mask
        self.z = np.dot(self.dropout_inputs, self.weights) + self.biases
        self.outputs = self.activation_func(self.z) if self.activation is not None else self.z
        return self.outputs

    def backward(self, grad, learning_rate):
        if self.activation is not None:
            if self.activation == 'relu':
                activation_derivative = self.relu_derivative
            elif self.activation == 'sigmoid':
                activation_derivative = self.sigmoid_derivative

                grad *= activation_derivative(self.z)
        grad_w = np.dot(self.dropout_inputs.T, grad)
        grad_b = np.sum(grad, axis=0)
        grad_inputs = np.dot(grad, self.weights.T)
        self.weights -= learning_rate * grad_w
        self.biases -= learning_rate * grad_b
        grad_inputs *= self.dropout_mask
        return grad_inputs

      # Activation functions
    def relu(self,x):
        return np.maximum(0, x)

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def relu_derivative(self,x):
        return np.where(x > 0, 1, 0)

    def sigmoid_derivative(self,x):
        sigmoid_x = self.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)

# Dense Network
class DenseNetwork:
    def __init__(self, layers):
        self.layers = layers

        input_size = layers[0].input_size
        for layer in layers:
            layer.weights = np.random.randn(input_size, layer.neurons) * 0.01
            layer.biases = np.zeros(layer.neurons)
            input_size = layer.neurons

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad, learning_rate):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    def train(self, x_train, y_train, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            outputs = self.forward(x_train)
            loss = np.mean((outputs - y_train) ** 2)
            grad = 2 * (outputs - y_train) / x_train.shape[0]
            self.backward(grad, learning_rate)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: Loss = {loss}")

    def predict(self, x_test):
        return self.forward(x_test)