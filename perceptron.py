import numpy as np
import math

class Perceptron:
    def __init__(self):
        self.activation_func = math.tanh
        self.weights = []
        self.bias = 0

    def forward(self, x = list[float]):
        weighted_sum = np.dot(x, self.weights) + self.bias
        output = self.activation_func(weighted_sum)

        return output

