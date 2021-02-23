from sklearn import datasets
import matplotlib.pyplot as plt
import math
import numpy as np

import HAL_9000
from HAL_9000.utils import train_test_split, to_categorical, normalize, accuracy_score, bar_widgets
from HAL_9000.optimizers import SGD, Adam, RMSprop
from HAL_9000.loss_functions import CrossEntropy
from HAL_9000.brain_layers import Dense, Conv2D, Flatten, Activation, BatchNorm2D, LayerNorm, MaxPool2D, AvgPool2D


data = datasets.load_digits()
X = data.data
y = data.target
y = to_categorical(y.astype("int"))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, seed=1)

X_train = X_train.reshape((-1, 1, 8, 8))
X_test = X_test.reshape((-1, 1, 8, 8))

# print(np.argmax(y_test[69]))
# plt.imshow(X_test[69, 0, :, :], cmap='gray')
# plt.show()
# z = X_test[69, 0, :, :]
# z = z.reshape(1, 1, 8, 8)
# print(z.shape)

brain = HAL_9000.Brain(loss=CrossEntropy, opt=Adam())

brain.add(Conv2D(n_filters=3, filter_shape=(2, 2),
                 input_shape=(1, 8, 8), padding='same shape', stride=1))
brain.add(Activation('relu'))
# to use avgpool, replace MaxPool2D with AvgPool2D
brain.add(MaxPool2D())
# to use layernorm, replace BatchNorm2D with LayerNorm
brain.add(BatchNorm2D())

brain.add(Conv2D(n_filters=3, filter_shape=(
    2, 2), stride=1, padding='same shape'))
brain.add(Activation('relu'))
brain.add(MaxPool2D())
brain.add(BatchNorm2D())

brain.add(Flatten())

brain.add(Dense(20))
brain.add(Activation('relu'))

brain.add(Dense(10))
brain.add(Activation('softmax'))

brain.summary()

train_loss, train_acc, _, _ = brain.fit(
    X_train, y_train, epochs=50, batch_size=256)

loss, accuracy = brain.test(X_test, y_test)
print("Accuracy:", accuracy)

t = 10
prediction = brain.predict(X_test[t, :, :, :].reshape(1, 1, 8, 8))
y_pred = np.argmax(prediction)
print("Prediction:", y_pred, "Label:", np.argmax(y_test[t]))
