import os

import numpy as np
from tqdm import tqdm

from Agents.DQNAgent import DQNAgent
from Boards.Full_Board import FullBoard


def choose_rand(a):
    return np.random.choice(a)


def run_game(num_ep, model_output):
    env = FullBoard()
    agent0 = DQNAgent(env.state_size(), env.action_size())
    print(env.state_size(), env.action_size())

    batch = agent0.get_batch_size()

    agent0_reward = []
    agent1_reward = []
    agent2_reward = []
    agent3_reward = []
    episode_length = []

    output_dir = 'model_output/DQN/%s/' % model_output

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for ep in tqdm(range(0, num_ep), ascii=True, unit="e"):
        s, _, game_over, player_turn = env.reset()
        step = 0
        episode_reward = [0.0, 0.0, 0.0, 0.0]
        while not game_over:
            # Roll dice and retrieve valid actions
            env.roll_dice()
            action_list = env.get_next_states(player_turn)

            if action_list:
                if player_turn == 0:
                    # Process state and query agent for action
                    s_ = env.convert_state(0)
                    action = agent0.act(s_, action_list)
                else:
                    action = choose_rand(action_list)

                new_s, reward, game_over, player_turn_temp = env.make_step(action)

                if player_turn == 0:
                    # Store transition in replay buffer
                    new_s_ = env.convert_state(0)
                    agent0.remember(s_, action, reward[0], new_s_, game_over)

                episode_reward[player_turn] += reward[player_turn]

                player_turn = player_turn_temp
                step += 1
            else:
                player_turn = (player_turn + 1) % 4

            if game_over:
                # Store episode information
                agent0_reward.append(episode_reward[0])
                agent1_reward.append(episode_reward[1])
                agent2_reward.append(episode_reward[2])
                agent3_reward.append(episode_reward[3])
                episode_length.append(step / 4)

        agent0.reduce_epsilon()
        if len(agent0.memory) > batch:
            agent0.replay()

        if ep % 100 == 0:
            agent0.save(output_dir + "weights_" + '{:04d}'.format(ep) + ".hdf5")

    return [agent0_reward, agent1_reward, agent2_reward, agent3_reward, episode_length]
