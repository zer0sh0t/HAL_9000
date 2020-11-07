from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

import HAL_9000
from HAL_9000.brain_layers import NeuroEvolution, Activation, Dense
from HAL_9000.utils import train_test_split, to_categorical, normalize
from HAL_9000.loss_functions import CrossEntropy
from HAL_9000.optimizers import Adam

data = datasets.load_digits()
X = normalize(data.data)
y = data.target
y = to_categorical(y.astype("int"))


def brain_fn(n_inputs, n_outputs):
    brain = HAL_9000.Brain(loss=CrossEntropy, opt=Adam())
    brain.add(Dense(10, input_shape=(n_inputs,)))
    brain.add(Activation("relu"))
    brain.add(Dense(n_outputs))
    brain.add(Activation("softmax"))

    return brain


brain_fn(n_inputs=X.shape[1], n_outputs=y.shape[1]).summary()
population_size = 100
mutation_rate = 0.1
generations = 10

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, seed=1)

model = NeuroEvolution(brain_fn, population_size, mutation_rate)

best_brain = model.begin_evolution(X_train, y_train, generations)
loss, acc = best_brain.test(X_test, y_test)
print(acc * 100)
