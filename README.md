# Dense-Network
Implementation of Dense  Neural Network

Example of DenseNetwork using my dnn.py package you can see in example.py


model = DenseNetwork
(
   [DenseLayer(neurons=128, input_size=4, activation='relu', dropout_rate=0.01),
    DenseLayer(neurons=16, activation='sigmoid'),
    DenseLayer(neurons=1)]
)


model.train(x_train, y_train, num_epochs=20000, learning_rate=0.1)

predictions = model.predict(x_test)