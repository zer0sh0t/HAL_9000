import math
import random
import numpy as np
import copy
import gym
from HAL_9000.activation_functions import Sigmoid, ReLU, LeakyReLU, TanH, ELU, Softmax
import time


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


class VanillaRNN(Layer):
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

        for g in [grad_Whh, grad_Wxh, grad_Why]:
            np.clip(g, -5, 5, out=g)

        self.Whh = self.Whh_opt.update(self.Whh, grad_Whh)
        self.Wxh = self.Wxh_opt.update(self.Wxh, grad_Wxh)
        self.Why = self.Why_opt.update(self.Why, grad_Why)

        return grad_

    def output_shape(self):
        return self.input_shape


class Dropout(Layer):
    def __init__(self, p=0.2):
        self.p = p
        self.mask = None
        self.input_shape = None
        self.n_units = None
        self.trainable = True

    def forward_pass(self, X, training=True):
        m = (1 - self.p)
        if training:
            self.mask = np.random.uniform(size=X.shape) > self.p
            m = self.mask
        return X * m

    def backward_pass(self, grad):
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


class DQN:
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


class NeuroEvolution:
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
