import numpy as np
import progressbar
from terminaltables import AsciiTable
from HAL_9000.utils import batch_iterator, bar_widgets


class Brain:
    def __init__(self, loss, opt, val_data=None):
        self.layers = []
        self.loss_fn = loss()
        self.opt = opt
        self.losses = {"train_loss": [], "val_loss": []}
        self.accuracies = {"train_acc": [], "val_acc": []}
        self.prog_bar = progressbar.ProgressBar(widgets=bar_widgets)
        self.val_set = None
        if val_data:
            X, y = val_data
            self.val_set = {"X": X, "y": y}

    def set_trainable(self, trainable):
        for layer in self.layers:
            layers.trainable = trainable

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(shape=self.layers[-1].output_shape())
        if hasattr(layer, "init_parameters"):
            layer.init_parameters(opt=self.opt)

        self.layers.append(layer)

    def for_prop(self, X, training=True):
        a = X
        for layer in self.layers:
            a = layer.forward_propagation(a, training)

        return a

    def back_prop(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward_propagation(grad)

    def train(self, X, y):
        a = self.for_prop(X)
        loss = np.mean(self.loss_fn.loss(y, a))
        acc = self.loss_fn.acc(y, a)
        grad = self.loss_fn.gradient(y, a)
        self.back_prop(grad)

        return loss, acc

    def test(self, X, y):
        a = self.for_prop(X, training=False)
        loss = np.mean(self.loss_fn.loss(y, a))
        acc = self.loss_fn.acc(y, a)

        return loss, acc

    def fit(self, X, y, epochs, batch_size):
        for _ in self.prog_bar(range(epochs)):
            batch_loss = []
            batch_acc = []
            for Xb, yb in batch_iterator(X, y, batch_size=batch_size):
                loss, acc = self.train(Xb, yb)
                batch_loss.append(loss)
                batch_acc.append(acc)

            self.losses["train_loss"].append(np.mean(batch_loss))
            self.accuracies["train_acc"].append(np.mean(batch_acc))

            if self.val_set is not None:
                loss, acc = self.test(self.val_set["X"], self.val_set["y"])
                self.losses["val_loss"].append(loss)
                self.accuracies["val_acc"].append(acc)

        return self.losses["train_loss"], self.accuracies["train_acc"], self.losses["val_loss"], self.accuracies["val_acc"]

    def predict(self, X):
        a = self.for_prop(X, training=False)

        return a

    def summary(self, name="Model Summary"):
        print(AsciiTable([[name]]).table)
        print("Input Shape: %s" % str(self.layers[0].input_shape))
        table_data = [["Layer Type", "Parameters", "Output Shape"]]
        tot_params = 0
        for layer in self.layers:
            layer_name = layer.layer_name()
            params = layer.n_parameters()
            out_shape = layer.output_shape()
            table_data.append([layer_name, str(params), str(out_shape)])
            tot_params += params

        print(AsciiTable(table_data).table)
        print("Total Parameters: %d\n" % tot_params)
