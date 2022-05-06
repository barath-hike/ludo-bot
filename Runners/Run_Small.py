import os
import random
from collections import deque

import numpy as np
from tqdm import tqdm

from Boards.Small_Board import SmallBoard


def choose_rand(a):
    return np.random.choice(a)


class TabularAgent:
    def __init__(self):
        self.Q = np.zeros((17, 17, 17, 17, 2))
        self.memory = deque(maxlen=100_000)
        self.epsilon = 1
        self.eps_min = 0.01
        self.alpha = 0.001
        self.gamma = 0.99
        self.batch = 1024

    def act(self, s, act_list):
        if len(act_list) == 1:
            return act_list[0]
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(act_list)
        best_a = -1
        max_q = -1000000
        for a in act_list:
            if self.Q[s[1], s[2], s[3], s[4]][a] > max_q:
                max_q = self.Q[s[1], s[2], s[3], s[4], a]
                best_a = a
            if self.Q[s[1], s[2], s[3], s[4]][a] == max_q:
                if np.random.uniform(0, 1) < 0.5:
                    max_q = self.Q[s[1], s[2], s[3], s[4], a]
                    best_a = a

        return best_a

    def q_learn(self, s, a, r, s_, done):
        pred = self.Q[s[1], s[2], s[3], s[4], a]
        next_q = 0
        if not done:
            next_q = np.amax(self.Q[s_[1], s_[2], s_[3], s_[4]])

        self.Q[s[1], s[2], s[3], s[4], a] = pred + self.alpha * (r + self.gamma * next_q - pred)

    def remember(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def replay(self):
        batch = random.sample(self.memory, self.batch)
        for transition in batch:
            self.q_learn(transition[0], transition[1], transition[2], transition[3], transition[4])

    def save_q(self, file_path, file):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        f = open(file, "w")
        q = np.reshape(self.Q, (4913, 34))
        np.savetxt(f, q, fmt='%-7.8f')
        f.flush()
        f.close()

    def load_q(self, f):
        dat = np.loadtxt(f)
        print(dat.shape)
        dat = dat.reshape((17, 17, 17, 17, 2))
        self.Q = dat

    def reduce_eps(self):
        if self.epsilon > self.eps_min:
            self.epsilon *= 0.999975


class TabularTDLambda(object):
    def __init__(self, _lambda=0.5, eps=1):
        self.Q = np.zeros((17, 17, 17, 17))
        self.E = []

        self.epsilon = eps
        self.gamma = 0.99
        self.alpha = 0.001
        self.eps_min = 0.01
        self._lambda = _lambda

    def get_q(self):
        return self.Q

    def set_q(self, q):
        self.Q = q

    def clear_e(self):
        self.E = []

    def act(self, s, act_list, state_list):
        self.E.append([s[1], s[2], s[3], s[4]])
        if len(act_list) == 1:
            return act_list[0]
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(act_list)
        best_v = -1
        best_a = -1
        for state, a in zip(state_list, act_list):
            if self.Q[state[1], state[2], state[3], state[4]] > best_v:
                best_v = self.Q[state[1], state[2], state[3], state[4]]
                best_a = a
            if self.Q[state[1], state[2], state[3], state[4]] == best_v:
                if np.random.uniform(0, 1) < 0.5:
                    best_v = self.Q[state[1], state[2], state[3], state[4]]
                    best_a = a

        return best_a

    def update(self, s, reward, s_):
        if len(self.E) == 1:
            return
        state_q = self.Q[s[1], s[2], s[3], s[4]]
        next_s_q = self.Q[s_[1], s_[2], s_[3], s_[4]]
        delta = reward + self.gamma * next_s_q - state_q
        for st, e in self.calc_elig(self.E):
            self.Q[st[0], st[1], st[2], st[3]] += self.alpha * delta * e

    def calc_elig(self, traj):
        unique = len(np.unique(traj, axis=0))
        traces = np.zeros(unique)
        states = []
        count = 0
        for s in traj:
            if s not in states:
                states.append(s)
                traces[count] = 1
                count += 1
            else:
                traces[states.index(s)] += 1
            traces *= self._lambda * self.gamma

        return zip(states, traces)

    def save_q(self, file_path, file):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        f = open(file, "w")
        q = np.reshape(self.Q, (289, 289))
        np.savetxt(f, q, fmt='%-7.8f')
        f.flush()
        f.close()

    def load_q(self, f):
        dat = np.loadtxt(f)
        print(dat.shape)
        dat = dat.reshape((17, 17, 17, 17))
        self.Q = dat

    def reduce_eps(self):
        if self.epsilon > self.eps_min:
            self.epsilon *= 0.999975


def run_game(num_ep):
    env = SmallBoard()
    # agent0 = TabularAgent()
    agent0 = TabularTDLambda()

    agent0_reward = []
    agent1_reward = []
    episode_length = []

    for ep in tqdm(range(0, num_ep), ascii=True, unit="e"):
        agent0.clear_e()
        # agent1.clear_e()
        s, _, game_over, player_turn = env.reset()
        step = 0
        episode_reward = [0.0, 0.0]
        while not game_over:
            s = env.roll_dice()
            action_list, state_list = env.get_next_states(player_turn)

            if action_list:
                if player_turn == 0:
                    # action = agent0.act(s, action_list)
                    action = agent0.act(s, action_list, state_list)
                else:
                    action = choose_rand(action_list)

                new_s, reward, game_over, player_turn_temp = env.make_step(action)

                if player_turn == 0:
                    # agent0.remember(s, action, reward[0], new_s, game_over)
                    agent0.update(s, reward[0], new_s)

                episode_reward[player_turn] += reward[player_turn]

                player_turn = player_turn_temp
                step += 1
            else:
                player_turn = (player_turn + 1) % 2

            if game_over:
                agent0_reward.append(episode_reward[0])
                agent1_reward.append(episode_reward[1])
                episode_length.append(step)

        agent0.reduce_eps()

        if ep % 1000 == 0:
            print(np.average(agent0_reward[-50_000:]), np.average(agent1_reward[-50_000:]),
                  np.average(episode_length[-5_000:]))  # , count, count - V_prev)

    agent0.save_q("results/small/q_learn/", "results/small/q_learn/000.txt")

    file_path = "results/small/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    f = open("results/small/000.txt", "a")

    f.write(','.join(str(e) for e in agent0_reward))
    f.write("\n")
    f.write(','.join(str(e) for e in agent1_reward))
    f.write("\n")
    f.write(','.join(str(e) for e in episode_length))
    f.write("\n")
    f.flush()
    f.close()


if __name__ == '__main__':
    run_game(400_000)
