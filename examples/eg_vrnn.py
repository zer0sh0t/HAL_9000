import numpy as np
import HAL_9000
from HAL_9000.brain_layers import VanillaRNN, Activation
from HAL_9000.optimizers import Adam
from HAL_9000.loss_functions import CrossEntropy
from HAL_9000.utils import train_test_split, accuracy_score

data = open('input_vrnn.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print(data_size, vocab_size)

char_to_idx = {}
idx_to_char = {}
for i, char in enumerate(chars):
    char_to_idx[char] = i
    idx_to_char[i] = char


batch_size = 150
time_steps = 10

X = np.zeros((batch_size, time_steps, vocab_size))
y = np.zeros((batch_size, time_steps, vocab_size))
z = np.zeros(vocab_size)
for i in range(batch_size):
    q = []
    for char in data[i:i+time_steps+1]:
        idx = char_to_idx[char]
        q.append(idx)

    for j in range(time_steps):
        z[q[j]] = 1
        X[i][j] = z
        z[q[j]] = 0

        z[q[j+1]] = 1
        y[i][j] = z
        z[q[j+1]] = 0

    data = data[i+time_steps:]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

brain = HAL_9000.Brain(loss=CrossEntropy, opt=Adam())
brain.add(VanillaRNN(30, input_shape=(time_steps, vocab_size),
                     activation='tanh', trunc=5))
brain.add(Activation('softmax'))

loss, acc, _, _ = brain.fit(X_train, y_train, epochs=100, batch_size=10)
print(loss[-1], acc[-1])

y_pred = np.argmax(brain.predict(X_test), axis=2)
y_test = np.argmax(y_test, axis=2)
# print(y_pred.shape)
# print(y_test.shape)
acc = np.mean(accuracy_score(y_test, y_pred))

print(acc * 100)

text = ""
for yp in y_pred:
    for idx in yp:
        char = idx_to_char[idx]
        text += char

print(text)
