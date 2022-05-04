import random
from collections import deque
from operator import itemgetter

import numpy as np
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size, epsilon=0.9):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=20000)

        self.epsilon = epsilon
        self.variable_epsilon = True
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.gamma = 0.95

        self.learning_rate = 0.001
        self.batch = 256

        self.model = self._build_model()
        self.model.summary()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

        self.iterations = 0
        self.copy = 16

    def _build_model(self):

        model = Sequential()
        model.add(Dense(32, input_shape=(self.state_size, 1), activation='relu'))
        # model.add(Dense(16, activation='relu'))
        model.add(Flatten())
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, action_list):
        # epsilon-greedy exploration
        if np.random.uniform(0, 1) <= self.epsilon:
            return np.random.choice(action_list)
        state = np.reshape(state, [1, 25, 1]) / 62

        # Find action-values for state and sort highest to lowest
        action_values = self.model.predict(state)
        sorted_pairs = sorted(enumerate(action_values), key=itemgetter(1), reverse=True)

        # return highest-valued valid action
        for action, value in sorted_pairs:
            if action in action_list:
                return action

    def replay(self):
        # sample batch of transitions and organise into separate lists
        minibatch = random.sample(self.memory, self.batch)
        states, actions, rewards, next_states, done = zip(*minibatch)
        states = np.reshape(states, [self.batch, self.state_size, 1]) / 62
        next_states = np.reshape(next_states, [self.batch, self.state_size, 1]) / 62

        # Approximate state-action pairs for all states in the batch
        q_states = self.model.predict(states)
        q_next_states = self.target_model.predict(next_states)
        # Ensure terminal states have 0 expected reward
        q_next_states[np.array(done)] = 0
        # "actual" q value of states using reward and next state estimates
        target_q = rewards + self.gamma * np.max(q_next_states, axis=1)

        # Reshape data and fit model
        list0 = []
        for a, b, c in zip(q_states, actions, target_q):
            a[b] = c
            list0.append(a)
        list0 = np.reshape(list0, [self.batch, self.action_size, 1])
        self.model.fit(np.array(states), list0, batch_size=self.batch, verbose=0, shuffle=False)

        self.iterations += 1
        # Update target_network weights every 16 iterations
        if self.iterations % self.copy == 0:
            self.target_model.set_weights(self.model.get_weights())

    def reduce_epsilon(self):
        if self.epsilon > self.epsilon_min and self.variable_epsilon:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        try:
            self.model.save_weights(name)
        except RuntimeError:
            print("Error: Exception when saving model weights")

    def get_batch_size(self):
        return self.batch
