# Example
import numpy as np
from dnn import DenseLayer, DenseNetwork

x_train = np.array([[0, 0, 1, 5], [0, 1, 2, 4], [1, 0, 3, 6], [1, 1, 4, 2]])
y_train = np.array([[0], [1], [1], [0]])
x_test = np.array([[0, 0, 1, 5], [0, 1, 2, 4], [1, 0, 3, 6], [1, 1, 4, 2]])


model = DenseNetwork(
   [DenseLayer(neurons=128, input_size=4, activation='relu', dropout_rate=0.01),
    DenseLayer(neurons=16, activation='sigmoid'),
    DenseLayer(neurons=1)]
)

model.train(x_train, y_train, num_epochs=20000, learning_rate=0.1)

predictions = model.predict(x_test)
print("Predictions:")
print(predictions)