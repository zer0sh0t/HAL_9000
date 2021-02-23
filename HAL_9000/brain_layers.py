import gym
import copy
import math
import random
import numpy as np
from HAL_9000.activation_functions import Sigmoid, ReLU, LeakyReLU, TanH, ELU, Softmax


class Layer(object):
    def set_input_shape(self, shape):
        self.input_shape = shape

    def layer_name(self):
        return self.__class__.__name__

    def n_parameters(self):
        return 0

    def forward_propagation(self, X, training):
        raise NotImplementedError()

    def backward_propagation(self, grad):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()


class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
        self.n_units = n_units
        self.input_shape = input_shape
        self.input = None
        self.trainable = True
        self.w = None
        self.b = None

    def init_parameters(self, opt):
        i = 1 / math.sqrt(self.input_shape[0])
        self.w = np.random.uniform(-i, i, (self.input_shape[0], self.n_units))
        self.b = np.zeros((1, self.n_units))
        self.w_opt = copy.copy(opt)
        self.b_opt = copy.copy(opt)

    def n_parameters(self):
        return np.prod(np.shape(self.w) + np.shape(self.b))

    def forward_propagation(self, X, training=True):
        self.layer_input = X
        a = X.dot(self.w) + self.b
        return a

    def backward_propagation(self, grad):
        w = self.w
        if self.trainable:
            grad_w = self.layer_input.T.dot(grad)
            grad_b = np.sum(grad, axis=0, keepdims=True)
            self.w = self.w_opt.update(self.w, grad_w)
            self.b = self.b_opt.update(self.b, grad_b)

        grad = grad.dot(w.T)
        return grad

    def output_shape(self):
        return (self.n_units, )


activ_fns = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'elu': ELU,
    'softmax': Softmax,
    'leaky_relu': LeakyReLU,
    'tanh': TanH,
}


class Activation(Layer):
    def __init__(self, name):
        self.activ_name = name
        self.activ_fn = activ_fns[name]()
        self.trainable = True

    def layer_name(self):
        return "Activation (%s)" % (self.activ_fn.__class__.__name__)

    def forward_propagation(self, X, training=True):
        self.layer_input = X
        return self.activ_fn(X)

    def backward_propagation(self, grad):
        return grad * self.activ_fn.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape


class Conv2D(Layer):
    def __init__(self, n_filters, filter_shape, input_shape=None, padding="same shape", stride=1):
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.input_shape = input_shape
        self.padding = padding
        self.stride = stride
        self.trainable = True

    def init_parameters(self, opt):
        fh, fw = self.filter_shape
        c = self.input_shape[0]
        i = 1 / math.sqrt(np.prod(self.filter_shape))
        self.w = np.random.uniform(-i, i, size=(self.n_filters, c, fh, fw))
        self.b = np.zeros((self.n_filters, 1))
        self.w_opt = copy.copy(opt)
        self.b_opt = copy.copy(opt)

    def n_parameters(self):
        return np.prod(self.w.shape) + np.prod(self.b.shape)

    def forward_propagation(self, X, training=True):
        bs, c, h, w = X.shape
        self.layer_input = X
        self.X_lat = img_2_lat(X, self.filter_shape,
                               stride=self.stride, output_shape=self.padding)
        self.w_lat = self.w.reshape((self.n_filters, -1))
        a = self.w_lat.dot(self.X_lat) + self.b
        a = a.reshape(self.output_shape() + (bs, ))

        return a.transpose(3, 0, 1, 2)

    def backward_propagation(self, grad):
        grad = grad.transpose(1, 2, 3, 0).reshape(self.n_filters, -1)
        if self.trainable:
            grad_w = grad.dot(self.X_lat.T).reshape(self.w.shape)
            grad_b = np.sum(grad, axis=1, keepdims=True)
            self.w = self.w_opt.update(self.w, grad_w)
            self.b = self.b_opt.update(self.b, grad_b)

        grad = self.w_lat.T.dot(grad)
        grad = lat_2_img(grad, self.layer_input.shape, self.filter_shape,
                         stride=self.stride, output_shape=self.padding)

        return grad

    def output_shape(self):
        c, h, w = self.input_shape
        fh, fw = self.filter_shape
        ph, pw = get_pads(self.filter_shape, output_shape=self.padding)
        oh = (h + np.sum(ph) - fh) / self.stride + 1
        ow = (w + np.sum(pw) - fw) / self.stride + 1

        return self.n_filters, int(oh), int(ow)


class SlowConv2D(Layer):
    def __init__(self, n_filters, filter_shape, input_shape=None, pad=0, stride=1):
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.input_shape = input_shape
        self.pad = pad
        self.stride = stride
        self.trainable = True

    def init_parameters(self, opt):
        fh, fw = self.filter_shape
        c = self.input_shape[0]
        i = 1 / math.sqrt(np.prod(self.filter_shape))
        self.w = np.random.uniform(-i, i, size=(self.n_filters, c, fh, fw))
        self.b = np.zeros((self.n_filters))
        self.w_opt = copy.copy(opt)
        self.b_opt = copy.copy(opt)

    def n_parameters(self):
        return np.prod(self.w.shape) + np.prod(self.b.shape)

    def forward_propagation(self, X, training=True):
        self.layer_input = X
        N, C, H, W = X.shape
        _, _, FH, FW = self.w.shape
        X_pad = np.pad(X, [(0,), (0,), (self.pad,), (self.pad,)])
        F, H_out, W_out = self.output_shape()
        output = np.zeros((N, F, H_out, W_out))

        for n in range(N):
            for f in range(F):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        height, width = h_out * self.stride, w_out * self.stride
                        output[n, f, h_out, w_out] = np.sum(
                            X_pad[n, :, height:height+FH, width:width+FW] * self.w[f, :]) + self.b[f]

        return output

    def backward_propagation(self, grad):
        if self.trainable:
            grad_w = np.zeros_like(self.w)
            grad_b = np.sum(grad, axis=(0, 2, 3))
            self.b = self.b_opt.update(self.b, grad_b)

        X = self.layer_input
        N, C, H, W = X.shape
        _, _, FH, FW = self.w.shape
        F, H_out, W_out = self.output_shape()
        X_pad = np.pad(X, [(0,), (0,), (self.pad,), (self.pad,)])
        output = np.zeros_like(X)
        grad_xpad = np.zeros_like(X_pad)

        for n in range(N):
            for f in range(F):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        height, width = h_out * self.stride, w_out * self.stride
                        if self.trainable:
                            grad_w[f, :] += X_pad[n, :, height:height+FH,
                                                  width:width+FW] * grad[n, f, h_out, w_out]

                        grad_xpad[n, :, height:height+FH, width:width +
                                  FW] += grad[n, f, h_out, w_out] * self.w[f, :]

        if self.trainable:
            self.w = self.w_opt.update(self.w, grad_w)
        output = grad_xpad[:, :, self.pad:self.pad+H, self.pad:self.pad+W]
        return output

    def output_shape(self):
        c, h, w = self.input_shape
        fh, fw = self.filter_shape
        oh = (h + (2*self.pad) - fh) / self.stride + 1
        ow = (w + (2*self.pad) - fw) / self.stride + 1

        return self.n_filters, int(oh), int(ow)


class LSTM(Layer):
    def __init__(self, n_units, input_shape=None):
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W_out = None
        self.Wx = None
        self.Wh = None

    def init_parameters(self, opt):
        T, D = self.input_shape
        i = 1 / math.sqrt(D)
        self.Wout = np.random.uniform(-i, i, (self.n_units, D))

        i = 1 / math.sqrt(4 * self.n_units)
        self.Wx = np.random.uniform(-i, i, (D, 4 * self.n_units))
        self.Wh = np.random.uniform(-i, i, (self.n_units, 4 * self.n_units))
        self.b = np.zeros((4 * self.n_units,))

        self.Wx_opt = copy.copy(opt)
        self.Wh_opt = copy.copy(opt)
        self.Wout_opt = copy.copy(opt)
        self.b_opt = copy.copy(opt)

    def n_parameters(self):
        return np.prod(self.W_out.shape) + np.prod(self.Wx.shape) + np.prod(self.Wh.shape) + np.prod(self.b.shape)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self, X, training=True):
        self.cache = []
        self.layer_input = X
        N, T, D = X.shape
        self.h = np.zeros((N, T, self.n_units))
        prev_h = np.zeros((N, self.n_units))
        prev_c = np.zeros((N, self.n_units))
        i = 0
        f = 0
        o = 0
        g = 0
        gate_i = 0
        gate_f = 0
        gate_o = 0
        gate_g = 0
        next_c = np.zeros((N, self.n_units))
        next_h = np.zeros((N, self.n_units))

        for t in range(T):
            self.x = X[:, t, :]
            if t == 0:
                self.cache.append((self.x, prev_h, prev_c, i, f, o, g, gate_i, gate_f,
                                   gate_o, gate_g, next_c, next_h))
            if t == 1:
                self.cache.pop(0)
            self._step_forward()
            next_h = self.cache[-1][-1]
            self.h[:, t, :] = next_h

        output = np.dot(self.h, self.Wout)
        return output

    def backward_propagation(self, grad):
        X = self.layer_input
        N, T, D = X.shape
        grad_ = np.zeros_like(X)
        grad_Wx = np.zeros_like(self.Wx)
        grad_Wh = np.zeros_like(self.Wh)
        grad_Wout = np.zeros_like(self.Wout)
        grad_b = np.zeros_like(self.b)
        self.dprev_c = np.zeros((N, self.n_units))
        self.dprev_h = np.zeros((N, self.n_units))

        for t in reversed(range(T)):
            self.grad_next = grad[:, t, :]
            self._step_backward(t)
            grad_[:, t, :] = self.dx
            grad_Wx += self.dWx
            grad_Wh += self.dWh
            grad_Wout += self.dWout
            grad_b += self.b

        for g in [grad_Wh, grad_Wx, grad_Wout, grad_b, grad_]:
            np.clip(g, -5, 5, out=g)

        self.Wh = self.Wh_opt.update(self.Wh, grad_Wh)
        self.Wx = self.Wx_opt.update(self.Wx, grad_Wx)
        self.Wout = self.Wout_opt.update(self.Wout, grad_Wout)
        self.b = self.b_opt.update(self.b, grad_b)
        return grad_

    def _step_forward(self):
        prev_c, prev_h = self.cache[-1][-2], self.cache[-1][-1]

        x = self.x
        a = np.dot(prev_h, self.Wh) + np.dot(x, self.Wx) + self.b
        i, f, o, g = np.split(a, 4, axis=1)
        gate_i, gate_f, gate_o, gate_g = self.sigmoid(
            i), self.sigmoid(f), self.sigmoid(o), np.tanh(g)
        next_c = gate_f * prev_c + gate_i * gate_g
        next_h = gate_o * np.tanh(next_c)

        self.cache.append((x, prev_h, prev_c, i, f, o, g, gate_i,
                           gate_f, gate_o, gate_g, next_c, next_h))

    def _step_backward(self, t):
        (x, prev_h, prev_c, i, f, o, g, gate_i, gate_f,
         gate_o, gate_g, next_c, next_h) = self.cache[t]

        self.dWout = np.dot(next_h.T, self.grad_next)
        dnext_h = np.dot(self.grad_next, self.Wout.T)
        dnext_h += self.dprev_h
        dgate_o = dnext_h * np.tanh(next_c)
        dnext_c = dnext_h * gate_o * (1 - np.tanh(next_c)**2)
        dnext_c += self.dprev_c

        dgate_f = dnext_c * prev_c
        dgate_i = dnext_c * gate_g
        dgate_g = dnext_c * gate_i
        self.dprev_c = dnext_c * gate_f

        dg = dgate_g * (1 - np.tanh(g) ** 2)
        do = dgate_o * self.sigmoid(o) * (1 - self.sigmoid(o))
        df = dgate_f * self.sigmoid(f) * (1 - self.sigmoid(f))
        di = dgate_i * self.sigmoid(i) * (1 - self.sigmoid(i))

        dinputs = np.concatenate((di, df, do, dg), axis=1)
        self.dx = np.dot(dinputs, self.Wx.T)
        self.dprev_h = np.dot(dinputs, self.Wh.T)
        self.dWx = np.dot(x.T, dinputs)
        self.dWh = np.dot(prev_h.T, dinputs)
        self.db = np.sum(dinputs, axis=0)

    def output_shape(self):
        return self.input_shape


class VanillaRNN(Layer):
    def __init__(self, n_units, input_shape=None):
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.Whh = None
        self.Wxh = None
        self.Why = None

    def init_parameters(self, opt):
        timesteps, input_dim = self.input_shape
        i = 1 / math.sqrt(input_dim)
        self.Wxh = np.random.uniform(-i, i, (self.n_units, input_dim))

        i = 1 / math.sqrt(self.n_units)
        self.Whh = np.random.uniform(-i, i, (self.n_units, self.n_units))
        self.Why = np.random.uniform(-i, i, (input_dim, self.n_units))
        self.b = np.zeros((self.n_units,))

        self.Whh_opt = copy.copy(opt)
        self.Wxh_opt = copy.copy(opt)
        self.Why_opt = copy.copy(opt)
        self.b_opt = copy.copy(opt)

    def n_parameters(self):
        return np.prod(self.Whh.shape) + np.prod(self.Wxh.shape) + np.prod(self.Why.shape) + np.prod(self.b.shape)

    def forward_propagation(self, X, training=True):
        self.layer_input = X
        N, T, D = X.shape
        self.total_h_prev = np.zeros((N, T, self.n_units))
        self.h = np.zeros((N, T, self.n_units))
        self.h_prev = np.zeros((N, self.n_units))

        for t in range(T):
            self.x = X[:, t, :]
            self._step_forward()
            self.h[:, t, :] = self.h_next
            self.total_h_prev[:, t, :] = self.h_prev
            self.h_prev = self.h_next

        output = np.dot(self.h, self.Why.T)
        return output

    def backward_propagation(self, grad):
        X = self.layer_input
        N, T, D = X.shape
        grad_ = np.zeros((N, T, D))
        grad_Wxh = np.zeros((self.n_units, D))
        grad_Whh = np.zeros((self.n_units, self.n_units))
        grad_Why = np.zeros((D, self.n_units))
        grad_b = np.zeros((self.n_units,))
        self.dh_prev = 0

        for t in reversed(range(T)):
            self.grad_next = grad[:, t, :]
            self._step_backward(t)
            grad_[:, t, :] = self.dx
            grad_Wxh += self.dWxh
            grad_Whh += self.dWhh
            grad_Why += self.dWhy
            grad_b += self.db

        for g in [grad_Whh, grad_Wxh, grad_Why, grad_b, grad_]:
            np.clip(g, -5, 5, out=g)

        self.Whh = self.Whh_opt.update(self.Whh, grad_Whh)
        self.Wxh = self.Wxh_opt.update(self.Wxh, grad_Wxh)
        self.Why = self.Why_opt.update(self.Why, grad_Why)
        self.b = self.b_opt.update(self.b, grad_b)
        return grad_

    def _step_forward(self):
        h_linear = np.dot(self.h_prev, self.Whh) + \
            np.dot(self.x, self.Wxh.T) + self.b
        self.h_next = np.tanh(h_linear)

    def _step_backward(self, t):
        self.dWhy = np.dot(self.grad_next.T, self.h[:, t, :])
        dh = np.dot(self.grad_next, self.Why)
        dh += self.dh_prev
        dh = (1 - (self.h[:, t, :] ** 2)) * dh
        self.dh_prev = np.dot(dh, self.Whh.T)
        self.dWhh = np.dot(self.total_h_prev[:, t, :].T, dh)
        self.dWxh = np.dot(dh.T, self.layer_input[:, t, :])
        self.dx = np.dot(dh, self.Wxh)
        self.db = np.sum(dh, axis=0)

    def output_shape(self):
        return self.input_shape


class OldVanillaRNN(Layer):
    def __init__(self, n_units, input_shape=None, activation='tanh', trunc=5):
        self.input_shape = input_shape
        self.n_units = n_units
        self.activ = activ_fns[activation]()
        self.trainable = True
        self.trunc = trunc
        self.Whh = None
        self.Wxh = None
        self.Why = None

    def init_parameters(self, opt):
        timesteps, input_dim = self.input_shape
        i = 1 / math.sqrt(input_dim)
        self.Wxh = np.random.uniform(-i, i, (self.n_units, input_dim))

        i = 1 / math.sqrt(self.n_units)
        self.Whh = np.random.uniform(-i, i, (self.n_units, self.n_units))
        self.Why = np.random.uniform(-i, i, (input_dim, self.n_units))

        self.Whh_opt = copy.copy(opt)
        self.Wxh_opt = copy.copy(opt)
        self.Why_opt = copy.copy(opt)

    def n_parameters(self):
        return np.prod(self.Whh.shape) + np.prod(self.Wxh.shape) + np.prod(self.Why.shape)

    def forward_propagation(self, X, training=True):
        self.layer_input = X
        batch_size, timesteps, input_dim = X.shape

        self.h_ba = np.zeros((batch_size, timesteps, self.n_units))
        self.h = np.zeros((batch_size, timesteps+1, self.n_units))
        self.y = np.zeros((batch_size, timesteps, input_dim))

        self.h[:, -1] = np.zeros((batch_size, self.n_units))

        for t in range(timesteps):
            self.h_ba[:, t] = self.h[:, t -
                                     1].dot(self.Whh.T) + X[:, t].dot(self.Wxh.T)
            self.h[:, t] = self.activ(self.h_ba[:, t])
            self.y[:, t] = self.h[:, t].dot(self.Why.T)

        return self.y

    def backward_propagation(self, grad):
        _, timesteps, _ = grad.shape

        grad_Wxh = np.zeros_like(self.Wxh)
        grad_Whh = np.zeros_like(self.Whh)
        grad_Why = np.zeros_like(self.Why)

        grad_ = np.zeros_like(grad)
        for t in reversed(range(timesteps)):
            grad_Why += grad[:, t].T.dot(self.h[:, t])
            grad_h = grad[:, t].dot(self.Why) * \
                self.activ.gradient(self.h_ba[:, t])

            grad_[:, t] = grad_h.dot(self.Wxh)

            for t_ in reversed(np.arange(max(0, t - self.trunc), t+1)):
                grad_Wxh += grad_h.T.dot(self.layer_input[:, t_])
                grad_Whh += grad_h.T.dot(self.h[:, t_-1])

                grad_h = grad_h.dot(self.Whh) * \
                    self.activ.gradient(self.h_ba[:, t_-1])

        for g in [grad_Whh, grad_Wxh, grad_Why, grad_]:
            np.clip(g, -5, 5, out=g)

        self.Whh = self.Whh_opt.update(self.Whh, grad_Whh)
        self.Wxh = self.Wxh_opt.update(self.Wxh, grad_Wxh)
        self.Why = self.Why_opt.update(self.Why, grad_Why)
        return grad_

    def output_shape(self):
        return self.input_shape


class BatchNorm2D(Layer):
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.trainable = True
        self.eps = 0.01
        self.running_mean = None
        self.running_var = None

    def init_parameters(self, opt):
        self.gamma = np.ones(self.input_shape)
        self.beta = np.zeros(self.input_shape)

        self.gamma_opt = copy.copy(opt)
        self.beta_opt = copy.copy(opt)

    def n_parameters(self):
        return np.prod(self.gamma.shape) + np.prod(self.beta.shape)

    def forward_propagation(self, X, training=True):
        if self.running_mean is None:
            self.running_mean = np.mean(X, axis=0)
            self.running_var = np.var(X, axis=0)

        if training and self.trainable:
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0)
            self.running_mean = self.momentum * \
                self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * \
                self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var

        X_centered = X - mean
        self.stddev_inv = 1 / np.sqrt(var + self.eps)

        self.X_norm = X_centered * self.stddev_inv
        output = self.gamma * self.X_norm + self.beta

        return output

    def backward_propagation(self, grad):
        if self.trainable:
            grad_gamma = np.sum(grad * self.X_norm, axis=0)
            grad_beta = np.sum(grad, axis=0)

            self.gamma = self.gamma_opt.update(self.gamma, grad_gamma)
            self.beta = self.beta_opt.update(self.beta, grad_beta)

        batch_size = grad.shape[0]
        grad = (1 / batch_size) * self.stddev_inv * (batch_size * grad - np.sum(grad, axis=0) -
                                                     self.X_norm * np.sum(grad * self.X_norm, axis=0))

        return grad

    def output_shape(self):
        return self.input_shape


class LayerNorm(Layer):
    def __init__(self):
        self.trainable = True
        self.eps = 0.01

    def init_parameters(self, opt):
        self.gamma = np.ones(self.input_shape).reshape(-1)
        self.beta = np.zeros(self.input_shape).reshape(-1)

        self.gamma_opt = copy.copy(opt)
        self.beta_opt = copy.copy(opt)

    def n_parameters(self):
        return np.prod(self.gamma.shape) + np.prod(self.beta.shape)

    def forward_propagation(self, X, training=True):
        batch_size = X.shape[0]
        X = X.reshape(-1, batch_size)
        mean = np.mean(X, axis=0)
        var = np.var(X, axis=0)
        X_centered = X - mean

        self.stddev_inv = 1 / np.sqrt(var + self.eps)
        self.X_norm = X_centered * self.stddev_inv
        self.X_norm = self.X_norm.reshape(batch_size, -1)
        output = self.gamma * self.X_norm + self.beta
        output = output.reshape((output.shape[0], ) + self.input_shape)

        return output

    def backward_propagation(self, grad):
        grad = grad.reshape(grad.shape[0], -1)
        if self.trainable:
            grad_gamma = np.sum(grad * self.X_norm, axis=0)
            grad_beta = np.sum(grad, axis=0)

            self.gamma = self.gamma_opt.update(self.gamma, grad_gamma)
            self.beta = self.beta_opt.update(self.beta, grad_beta)

        batch_size = grad.shape[0]
        grad = (1 / batch_size) * self.stddev_inv * (batch_size * grad.T - np.sum(grad.T, axis=0) -
                                                     self.X_norm.T * np.sum(grad.T * self.X_norm.T, axis=0))
        grad = grad.reshape((grad.shape[1], ) + self.input_shape)

        return grad

    def output_shape(self):
        return self.input_shape


class Dropout(Layer):
    def __init__(self, p=0.2):
        self.p = p
        self.mask = None
        self.input_shape = None
        self.n_units = None
        self.trainable = True

    def forward_propagation(self, X, training=True):
        m = (1 - self.p)
        if training:
            self.mask = np.random.uniform(size=X.shape) > self.p
            m = self.mask
        return X * m

    def backward_propagation(self, grad):
        return grad * self.mask

    def output_shape(self):
        return self.input_shape


class Flatten(Layer):
    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        self.prev_shape = None
        self.trainable = True

    def forward_propagation(self, X, training=True):
        self.prev_shape = X.shape
        X = X.reshape((X.shape[0], -1))
        return X

    def backward_propagation(self, grad):
        return grad.reshape(self.prev_shape)

    def output_shape(self):
        return (np.prod(self.input_shape), )


class PoolLayer(Layer):
    def __init__(self, pool_shape=(2, 2), stride=1, padding="none"):
        self.pool_shape = pool_shape
        self.stride = stride
        self.padding = padding
        self.trainable = True

    def forward_propagation(self, X, training=True):
        self.layer_input = X
        batch_size, channels, height, width = X.shape
        _, out_height, out_width = self.output_shape()

        X = X.reshape(batch_size*channels, 1, height, width)
        X_col = img_2_lat(X, self.pool_shape, self.stride, self.padding)

        output = self._pool_forward(X_col)
        output = output.reshape(out_height, out_width, batch_size, channels)
        output = output.transpose(2, 3, 0, 1)

        return output

    def backward_propagation(self, grad):
        batch_size, _, _, _ = grad.shape
        channels, height, width = self.input_shape
        grad = grad.transpose(2, 3, 0, 1).ravel()

        grad_col = self._pool_backward(grad)
        grad = lat_2_img(grad_col, (batch_size * channels, 1, height, width),
                         self.pool_shape, self.stride, self.padding)
        grad = grad.reshape((batch_size,) + self.input_shape)

        return grad

    def output_shape(self):
        channels, height, width = self.input_shape
        if self.padding == "same shape":
            out_height = height
            out_width = width
        else:
            out_height = (height - self.pool_shape[0]) / self.stride + 1
            out_width = (width - self.pool_shape[1]) / self.stride + 1

        return channels, int(out_height), int(out_width)


class MaxPool2D(PoolLayer):
    def _pool_forward(self, X_col):
        arg_max = np.argmax(X_col, axis=0).flatten()
        output = X_col[arg_max, range(arg_max.size)]
        self.cache = arg_max
        return output

    def _pool_backward(self, accum_grad):
        accum_grad_col = np.zeros((np.prod(self.pool_shape), accum_grad.size))
        arg_max = self.cache
        accum_grad_col[arg_max, range(accum_grad.size)] = accum_grad
        return accum_grad_col


class AvgPool2D(PoolLayer):
    def _pool_forward(self, X_col):
        output = np.mean(X_col, axis=0)
        return output

    def _pool_backward(self, accum_grad):
        accum_grad_col = np.zeros((np.prod(self.pool_shape), accum_grad.size))
        accum_grad_col[:, range(accum_grad.size)] = 1. / \
            accum_grad_col.shape[0] * accum_grad
        return accum_grad_col


class SlowMaxPool2D(Layer):
    def __init__(self, pool_shape=(2, 2), stride=1):
        self.pool_shape = pool_shape
        self.stride = stride
        self.trainable = True

    def forward_propagation(self, X, training=True):
        self.layer_input = X
        N, C, H, W = X.shape
        FH, FW = self.pool_shape
        _, H_out, W_out = self.output_shape()
        output = np.zeros((N, C, H_out, W_out))

        for n in range(N):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    height, width = h_out * self.stride, w_out * self.stride
                    output[n, :, h_out, w_out] = np.max(
                        X[n, :, height:height+FH, width:width+FW], axis=(-2, -1))

        return output

    def backward_propagation(self, grad):
        X = self.layer_input
        N, C, H, W = X.shape
        FH, FW = self.pool_shape
        _, H_out, W_out = self.output_shape()
        output = np.zeros_like(X)

        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        height, width = h_out * self.stride, w_out * self.stride
                        idx = np.unravel_index(
                            np.argmax(X[n, c, height:height+FH, width:width+FW]), (FH, FW))
                        output[n, c, height:height+FH, width:width +
                               FW][idx] = grad[n, c, h_out, w_out]

        return output

    def output_shape(self):
        channels, height, width = self.input_shape
        out_height = (height - self.pool_shape[0]) / self.stride + 1
        out_width = (width - self.pool_shape[1]) / self.stride + 1
        return channels, int(out_height), int(out_width)


class Reshape(Layer):
    def __init__(self, shape, input_shape=None):
        self.prev_shape = None
        self.trainable = True
        self.shape = shape
        self.input_shape = input_shape

    def forward_pass(self, X, training=True):
        self.prev_shape = X.shape
        return X.reshape((X.shape[0], ) + self.shape)

    def backward_pass(self, grad):
        return grad.reshape(self.prev_shape)

    def output_shape(self):
        return self.shape


class DQN():
    def __init__(self, env_name='CartPole-v1', epsilon=1, min_epsilon=0.1, gamma=0.9, decay_rate=0.005):
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate

        self.memory = []
        self.memory_size = 200

        self.env = gym.make(env_name)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

    def give_brain(self, brain):
        self.brain = brain(n_inputs=self.n_states, n_outputs=self.n_actions)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.brain.predict(state), axis=1)[0]

        return action

    def stack_memory(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def random_play(self, replay):
        X = np.zeros((len(replay), self.n_states))
        y = np.zeros((len(replay), self.n_actions))

        states = np.array([s[0] for s in replay])
        new_states = np.array([ns[3] for ns in replay])

        Q = self.brain.predict(states)
        nQ = self.brain.predict(new_states)

        for i in range(len(replay)):
            s, a, r, ns, d = replay[i]
            Q[i][a] = r
            if not d:
                Q[i][a] += self.gamma * np.amax(nQ[i])

            X[i] = s
            y[i] = Q[i]

        return X, y

    def train(self, epochs, batch_size):
        max_reward = 0
        for epoch in range(epochs):
            state = self.env.reset()
            total_reward = 0
            epoch_loss = []
            while True:
                action = self.select_action(state)
                new_state, reward, done, _ = self.env.step(action)
                self.stack_memory(state, action, reward, new_state, done)

                bs = min(len(self.memory), batch_size)
                replay = random.sample(self.memory, bs)

                X, y = self.random_play(replay)
                loss, _ = self.brain.train(X, y)
                epoch_loss.append(loss)

                total_reward += reward
                state = new_state
                if done:
                    break

            self.epsilon = self.min_epsilon + \
                (1.0 - self.min_epsilon) * np.exp(-self.decay_rate * epoch)

            max_reward = max(max_reward, total_reward)

        print("Training Done!!")

    def play(self, epochs):
        for epoch in range(epochs):
            state = self.env.reset()
            total_reward = 0
            while True:
                self.env.render()
                action = np.argmax(self.brain.predict(state), axis=1)[0]
                new_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = new_state

                if done:
                    break

            print(f"Epoch: {epoch} | Total Reward: {total_reward}")

        self.env.close()


class NeuroEvolution():
    def __init__(self, brain_fn, population_size, mutation_rate):
        self.brain_fn = brain_fn
        self.pop_size = population_size
        self.mut_rate = mutation_rate

    def build_brain(self, id):
        brain = self.brain_fn(n_inputs=self.X.shape[1],
                              n_outputs=self.y.shape[1])
        brain.id = id
        brain.accuracy = 0
        brain.fitness = 0

        return brain

    def init_pop(self):
        self.population = []
        for _ in range(self.pop_size):
            brain = self.build_brain(np.random.randint(9999999))
            self.population.append(brain)

    def mutate(self, baby, var=1):
        for layer in baby.layers:
            if hasattr(layer, "w"):
                mutation_mask = np.random.binomial(1, p=self.mut_rate,
                                                   size=layer.w.shape)
                layer.w += np.random.normal(loc=0, scale=var,
                                            size=layer.w.shape) * mutation_mask
                mutation_mask = np.random.binomial(1, p=self.mut_rate,
                                                   size=layer.b.shape)
                layer.b += np.random.normal(loc=0, scale=var,
                                            size=layer.b.shape) * mutation_mask

        return baby

    def inherit_genes(self, baby, parent):
        for i in range(len(baby.layers)):
            if hasattr(baby.layers[i], "w"):
                baby.layers[i].w = parent.layers[i].w.copy()
                baby.layers[i].b = parent.layers[i].b.copy()

    def fuck(self, parent0, parent1):
        baby0 = self.build_brain(id=parent0.id+1)
        baby1 = self.build_brain(id=parent1.id+1)

        self.inherit_genes(baby0, parent0)
        self.inherit_genes(baby1, parent1)

        for i in range(len(baby0.layers)):
            if hasattr(baby0.layers[i], "w"):
                n_neurons = baby0.layers[i].w.shape[1]
                z = np.random.randint(0, n_neurons)

                baby0.layers[i].w[:, z:] = parent1.layers[i].w[:, z:].copy()
                baby0.layers[i].b[:, z:] = parent1.layers[i].b[:, z:].copy()

                baby1.layers[i].w[:, z:] = parent0.layers[i].w[:, z:].copy()
                baby1.layers[i].b[:, z:] = parent0.layers[i].b[:, z:].copy()

        return baby0, baby1

    def calc_fitness(self):
        for baby in self.population:
            loss, acc = baby.test(self.X, self.y)
            baby.accuracy = acc
            baby.fitness = 1 / (loss + 1e-8)

    def begin_evolution(self, X, y, generations):
        self.X, self.y = X, y
        self.init_pop()
        n_winners = int(self.pop_size * 0.4)
        n_parents = self.pop_size - n_winners

        for i in range(generations):
            self.calc_fitness()
            best_fit_idx = np.argsort(
                [brain.fitness for brain in self.population])[::-1]
            self.population = [self.population[i] for i in best_fit_idx]

            fittest_brain = self.population[0]

            next_population = [self.population[i] for i in range(n_winners)]
            total_fitness = np.sum(
                [brain.fitness for brain in self.population])
            parent_probs = [brain.fitness /
                            total_fitness for brain in self.population]

            parents = np.random.choice(self.population, size=n_parents,
                                       p=parent_probs, replace=False)

            for i in np.arange(0, len(parents), 2):
                baby0, baby1 = self.fuck(parents[i], parents[i+1])
                next_population += [self.mutate(baby0), self.mutate(baby1)]

            self.population = next_population

        return fittest_brain


def get_pads(filter_shape, output_shape="same shape"):
    if output_shape == "none":
        return (0, 0), (0, 0)
    elif output_shape == "same shape":
        fh, fw = filter_shape

        ph1 = int(math.floor((fh - 1)/2))
        ph2 = int(math.ceil((fh - 1)/2))
        pw1 = int(math.floor((fw - 1)/2))
        pw2 = int(math.ceil((fw - 1)/2))

        return (ph1, ph2), (pw1, pw2)


def img_2_lat_idx(images_shape, filter_shape, padding, stride=1):
    bs, c, h, w = images_shape
    fh, fw = filter_shape
    ph, pw = padding
    oh = int((h + np.sum(ph) - fh) / stride + 1)
    ow = int((w + np.sum(pw) - fw) / stride + 1)

    x = np.repeat(np.arange(c), fh * fw).reshape(-1, 1)

    y0 = np.repeat(np.arange(fh), fw)
    y0 = np.tile(y0, c)
    y1 = stride * np.repeat(np.arange(oh), ow)

    z0 = np.tile(np.arange(fw), fh * c)
    z1 = stride * np.tile(np.arange(ow), oh)

    y = y0.reshape(-1, 1) + y1.reshape(1, -1)
    z = z0.reshape(-1, 1) + z1.reshape(1, -1)

    return (x, y, z)


def img_2_lat(images, filter_shape, stride, output_shape='same shape'):
    fh, fw = filter_shape
    ph, pw = get_pads(filter_shape, output_shape)
    padded_img = np.pad(images, ((0, 0), (0, 0), ph, pw), mode='constant')

    x, y, z = img_2_lat_idx(images.shape, filter_shape, (ph, pw), stride)
    lat = padded_img[:, x, y, z]
    c = images.shape[1]

    lat = lat.transpose(1, 2, 0).reshape(fh * fw * c, -1)
    return lat


def lat_2_img(lat, images_shape, filter_shape, stride, output_shape='same shape'):
    bs, c, h, w = images_shape
    ph, pw = get_pads(filter_shape, output_shape)

    hap = h + np.sum(ph)
    wap = w + np.sum(pw)
    blank_img = np.zeros((bs, c, hap, wap))

    x, y, z = img_2_lat_idx(images_shape, filter_shape, (ph, pw), stride)

    lat = lat.reshape(c * np.prod(filter_shape), -1, bs)
    lat = lat.transpose(2, 0, 1)
    np.add.at(blank_img, (slice(None), x, y, z), lat)

    return blank_img[:, :, ph[0]:h+ph[0], pw[0]:w+pw[0]]
