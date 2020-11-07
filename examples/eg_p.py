import numpy as np
from sklearn import datasets

import HAL_9000
from HAL_9000.activation_functions import Softmax
from HAL_9000.loss_functions import CrossEntropy
from HAL_9000.utils import train_test_split, accuracy_score, normalize, to_categorical

data = datasets.load_digits()
X = data.data
y = data.target

X = normalize(X)
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, seed=1)

model = HAL_9000.Perceptron(n_iter=2000, activ_fn=Softmax,
                            loss_fn=CrossEntropy, lr=1e-3)
model.fit(X_train, y_train)

prediction = model.predict(X_test)
y_pred = np.argmax(prediction, axis=1)
y_test = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
