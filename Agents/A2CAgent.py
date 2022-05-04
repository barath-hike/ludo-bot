import time

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Flatten


class Actor(keras.Model):
    def __init__(self, n_actions, state_size):
        super(Actor, self).__init__()
        self.n_actions = n_actions
        self.state_size = state_size

        self.model = keras.Sequential(
            layers=[
                Dense(32, input_shape=(1, self.state_size, 1), activation='relu'),
                Dense(16, activation='relu'),
                Flatten(),
                Dense(4, activation='softmax')
            ])

    def call(self, state):
        return self.model(state)


class Critic(keras.Model):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.state_size = state_size

        self.model = keras.Sequential(
            layers=[
                Dense(32, input_shape=(1, self.state_size, 1), activation='relu'),
                Dense(16, activation='relu'),
                Flatten(),
                Dense(1, activation='linear')
            ])

    def call(self, state):
        return self.model(state)


class Agent:
    def __init__(self, alpha=1e-4, gamma=0.95, epsilon=0.9, n_actions=4, input_dim=25):
        tf.random.set_seed(time.time())
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_size = input_dim
        self.loss = keras.losses.Huber()

        self.actor = Actor(n_actions=self.n_actions, state_size=self.state_size)
        self.critic = Critic(state_size=self.state_size)

        self.a_opt = tf.keras.optimizers.Adam(learning_rate=self.alpha)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=self.alpha)

    def act(self, s, action_list):
        # Convert state and query network for probabilities
        state = np.reshape(s, [1, self.state_size, 1]) / 62
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        probs = self.actor.call(state)
        # filter out invalid actions and create distribution
        valid = []
        for a in action_list:
            valid.append(probs[0][a])
        action_probabilities = tfp.distributions.Categorical(probs=valid)
        # Sample action from distribution
        action = 0
        if len(action_list) > 1:
            action = action_probabilities.sample().numpy()

        # store and return action and log_prob of action
        return action_list[action]

    def learn(self, state, action, reward, state_, done):
        state = np.reshape(state, [1, self.state_size, 1]) / 62
        state_ = np.reshape(state_, [1, self.state_size, 1]) / 62
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state_ = tf.convert_to_tensor(state_, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            state_value = self.critic.call(state)
            probs = self.actor.call(state)
            state_value_ = self.critic(state_)

            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(action)

            bonus = tf.convert_to_tensor(log_prob * probs[0][action], dtype=tf.float32)
            delta = reward + self.gamma * state_value_ * (1 - int(done)) - state_value + self.alpha * bonus

            actor_loss = -log_prob * tf.stop_gradient(delta)
            # Huber Loss
            if delta <= 1:
                critic_loss = 0.5 * delta ** 2
            else:
                critic_loss = delta - 0.5

        grads1 = tape1.gradient(actor_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(critic_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return actor_loss, critic_loss

    def reduce_alpha(self):
        if self.alpha > 1e-15:
            self.alpha *= 0.999975

    def reduce_epsilon(self):
        if self.epsilon > 0.01:
            self.epsilon *= 0.99975

    def save_model(self, name_a, name_b):
        try:
            self.actor.save_weights(name_a)
            self.critic.save_weights(name_b)
        except RuntimeError:
            print("Error: Exception when saving model weights")

    def load_model(self, name_a, name_b):
        self.actor.load_weights(name_a)
        self.critic.load_weights(name_b)

    def load_actor(self, name):
        self.actor.load_weights(name)
