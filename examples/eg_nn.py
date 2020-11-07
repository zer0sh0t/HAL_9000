import numpy as np
import math
from sklearn import datasets

import HAL_9000
from HAL_9000.optimizers import Adam
from HAL_9000.loss_functions import CrossEntropy
from HAL_9000.activation_functions import Softmax
from HAL_9000.brain_layers import Dense, Activation
from HAL_9000.utils import train_test_split, normalize, to_categorical, accuracy_score

data = datasets.load_digits()
X = data.data
X = normalize(X)
n_samples, n_features = np.shape(X)

y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

brain = HAL_9000.Brain(loss=CrossEntropy, opt=Adam())
brain.add(Dense(input_shape=(n_features,), n_units=64))
brain.add(Activation('relu'))
brain.add(Dense(n_units=64))
brain.add(Activation('relu'))
brain.add(Dense(n_units=10))
brain.add(Activation('softmax'))

brain.summary()
t_l, t_a, _, _ = brain.fit(X_train, to_categorical(
    y_train), epochs=50, batch_size=50)

y_pred = np.argmax(brain.predict(X_test), axis=1)
acc = accuracy_score(y_test, y_pred)
print(acc * 100)
