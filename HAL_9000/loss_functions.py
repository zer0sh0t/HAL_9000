import numpy as np
from HAL_9000.utils import accuracy_score


class Loss(object):
    def loss(self, y, a):
        return NotImplementedError()

    def gradient(self, y, a):
        raise NotImplementedError()

    def acc(self, y, a):
        return 0


class SquareLoss(Loss):
    def __init__(self):
        pass

    def loss(self, y, a):
        return 0.5 * np.power((y - a), 2)

    def gradient(self, y, a):
        return -(y - a)


class CrossEntropy(Loss):
    def __init__(self):
        pass

    def loss(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def gradient(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)

    def acc(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))
