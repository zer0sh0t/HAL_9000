import numpy as np
import math
from HAL_9000.utils import train_test_split, to_categorical, normalize, accuracy_score, bar_widgets
from HAL_9000.loss_functions import CrossEntropy
import progressbar


class MLP:
    def __init__(self, n_hidden, hid_afn, out_afn, n_iter, lr):
        self.n_hidden = n_hidden
        self.n_iter = n_iter
        self.lr = lr
        self.hid_afn = hid_afn()
        self.out_afn = out_afn()
        self.loss_fn = CrossEntropy()
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

    def init_parameters(self, X, y):
        n_samples, n_features = np.shape(X)
        _, n_outputs = np.shape(y)

        i = 1 / math.sqrt(n_features)
        self.wh = np.random.uniform(-i, i, (n_features, self.n_hidden))
        self.bh = np.zeros((1, self.n_hidden))

        i_ = 1 / math.sqrt(self.n_hidden)
        self.wo = np.random.uniform(-i_, i_, (self.n_hidden, n_outputs))
        self.bo = np.zeros((1, n_outputs))

    def fit(self, X, y):
        self.init_parameters(X, y)
        for i in self.progressbar(range(self.n_iter)):
            # forward propagation
            z_hid = X.dot(self.wh) + self.bh
            a_hid = self.hid_afn(z_hid)

            z_out = a_hid.dot(self.wo) + self.bo
            a_out = self.out_afn(z_out)

            # backward propagation
            grad_out = self.loss_fn.gradient(
                y, a_out) * self.out_afn.gradient(z_out)
            grad_wo = a_hid.T.dot(grad_out)
            grad_bo = np.sum(grad_out, axis=0, keepdims=True)

            grad_hid = grad_out.dot(self.wo.T) * self.hid_afn.gradient(z_hid)
            grad_wh = X.T.dot(grad_hid)
            grad_bh = np.sum(grad_wh, axis=0, keepdims=True)

            self.wo -= self.lr * grad_wo
            self.bo -= self.lr * grad_bo

            self.wh -= self.lr * grad_wh
            self.bh -= self.lr * grad_bh

    def predict(self, X):
        z_hid = X.dot(self.wh) + self.bh
        a_hid = self.hid_afn(z_hid)

        z_out = a_hid.dot(self.wo) + self.bo
        a_out = self.out_afn(z_out)

        return a_out
