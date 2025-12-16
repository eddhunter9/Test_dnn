import numpy as np
import math

class Perceptron:
    def __init__(self):
        self.activation_func = math.tanh
        self.weights = []
        self.bias = 0

    def forward(self, x = list[float]):
        weighted_sum = np.dot(x, self.weights) + self.bias #iloczyn skalarny
        output = self.activation_func(weighted_sum)

        return output

    def train(self, X_train: list[list[float]], y_expected: list[float], n_iter: int, learn_rate: float):
        number_inp = len(X_train[0])
        self.weights = np.random.randn(number_inp)
        self.bias = np.random.randn()

        for _ in range(n_iter):
            for i, x in enumerate(X_train):
                y_predicted = self.forward(x)
                error = y_expected[i] - y_predicted
                correction = error * learn_rate
                self.weights = self.weights + correction * x
                self.bias = self.bias + correction

    def predict(self, X: list[list[float]]):
        predictions = []
        for _, x in enumerate(X):
            output = self.forward(x)
            predictions.append(output)

        return predictions




