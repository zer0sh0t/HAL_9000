import numpy as np
import math
from HAL_9000.activation_functions import Sigmoid, ReLU, LeakyReLU, TanH, ELU
from HAL_9000.loss_functions import CrossEntropy, SquareLoss
from HAL_9000.utils import train_test_split, to_categorical, normalize, accuracy_score, bar_widgets
import progressbar


class Perceptron:
    def __init__(self, n_iter, activ_fn, loss_fn, lr):
        self.n_iter = n_iter
        self.activ_fn = activ_fn()
        self.loss_fn = loss_fn()
        self.lr = lr
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)
        _, n_outputs = np.shape(y)

        i = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-i, i, (n_features, n_outputs))
        self.b = np.zeros((1, n_outputs))

        for i in self.progressbar(range(self.n_iter)):
            # forward propagation
            z = X.dot(self.w) + self.b
            a = self.activ_fn(z)

            # backward propagation
            grad = self.loss_fn.gradient(y, a) * self.activ_fn.gradient(z)
            grad_wrt_w = X.T.dot(grad)
            grad_wrt_b = np.sum(grad, axis=0, keepdims=True)

            self.w -= self.lr * grad_wrt_w
            self.b -= self.lr * grad_wrt_b

    def predict(self, X):
        z = X.dot(self.w) + self.b
        a = self.activ_fn(z)

        return a
