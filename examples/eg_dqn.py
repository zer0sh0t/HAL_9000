import numpy as np
import HAL_9000
from HAL_9000.optimizers import Adam
from HAL_9000.loss_functions import SquareLoss
from HAL_9000.brain_layers import DQN, Dense, Activation

env_name = 'Acrobot-v1'

dqn = DQN(env_name=env_name, epsilon=0.9,
          min_epsilon=0.1, gamma=0.8, decay_rate=0.005)


def brain(n_inputs, n_outputs):
    brain = HAL_9000.Brain(loss=SquareLoss, opt=Adam())
    brain.add(Dense(30, input_shape=(n_inputs,)))
    brain.add(Activation("relu"))
    brain.add(Dense(n_outputs))

    return brain


dqn.give_brain(brain)
dqn.brain.summary()

dqn.train(epochs=100, batch_size=40)
dqn.play(epochs=50)
