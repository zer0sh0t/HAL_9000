import HAL_9000
import numpy as np
from HAL_9000.brain_layers import VanillaRNN, Activation, LSTM
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

N = 150
T = 20
X = np.zeros((N, T), dtype=np.uint8)
y = np.zeros((N, T), dtype=np.uint8)
for i in range(N):
    q = []
    for char in data[i:i+T+1]:
        idx = char_to_idx[char]
        q.append(idx)

    X[i] = q[:len(q)-1]
    y[i] = q[1:len(q)]
    data = data[i+T:]

W = np.eye(vocab_size)
X = W[X]
y = W[y]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

brain = HAL_9000.Brain(loss=CrossEntropy, opt=Adam())
# to use lstm, replace VanillaRNN with LSTM
brain.add(VanillaRNN(50, input_shape=(T, vocab_size)))
brain.add(Activation('softmax'))

loss, acc, _, _ = brain.fit(X_train, y_train, epochs=500, batch_size=128)
print(loss[-1], acc[-1])

y_pred = np.argmax(brain.predict(X_test), axis=2)
y_test = np.argmax(y_test, axis=2)
# print(y_pred.shape)
# print(y_test.shape)
acc = np.mean(accuracy_score(y_test, y_pred))
print(acc)

text = "you are"
idx = []
for t in text:
    idx.append(char_to_idx[t])

vec = W[idx]
vec = np.array([vec])
for _ in range(100):
    out = brain.predict(vec)
    idx.append(np.argmax(out[:, -1], axis=-1)[0])
    vec = np.array([W[idx]])

out_text = ""
for i in idx:
    char = idx_to_char[i]
    out_text += char

print(out_text)
