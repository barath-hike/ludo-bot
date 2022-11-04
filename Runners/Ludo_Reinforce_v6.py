import os

import numpy as np
from tqdm import tqdm
import gc

from Boards.Speed_leedo_2p_v6 import FullBoard
from Agents.ReinforceAgent import Agent
from Intelligent_bot.simulate_reinforce import simulate

from slack import WebClient
bot_name='Reinforce Bot-2'
slack_client=WebClient(token='xoxb-2151902985-2324529348736-eNKHbuseT407uHVBjejHGJvy')
icon_url=''
channel='#rl-bot'

def choose_rand(a):
    return np.random.choice(a)


def run_game(num_ep, model_output):

    env = FullBoard()
    agent0 = Agent(17, env.action_size())
    agent0.load_model('../DataBot/models/mlp_bot_models_v2/model_84.hdf5')

    batch_size = 10

    agent0_reward = []
    agent1_reward = []
    output_dir = 'model_output/REINFORCE_v6/%s/' % model_output

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for ep in tqdm(range(0, num_ep), ascii=True, unit="e"):

        batch = [[], []]
        rew_path = []

        # Generate batches of experience
        for it in range(0, batch_size):
            traj, traj1 = generate_episode(agent0, env)
            # Append trajectories and calculate average reward per episode
            batch[0].extend(traj[0])
            batch[1].extend(traj[1])
            rew_path.append(traj[2])

            batch[0].extend(traj1[0])
            batch[1].extend(traj1[1])
            rew_path.append(traj1[2])

        # Learn from experience
        agent0.learn(batch, agent0.get_returns(rew_path))

        if ep % 100 == 0:
            agent0.save_model(output_dir + "weights_" + '{:04d}'.format(ep) + ".hdf5")
            gc.collect()

        # if ep % 1000 == 0 and ep != 0:
        #     win_rate = simulate(output_dir + "weights_" + '{:04d}'.format(ep) + ".hdf5")
        #     message = 'Win rate against intelligent bot after {0} episodes: {rate:0.2f}%'.format(ep*10, rate=win_rate*100)
        #     slack_client.chat_postMessage(channel=channel,
        #                   text=message,
        #                   username=bot_name,
        #                   icon_url=icon_url)

    return [agent0_reward, agent1_reward]


def generate_episode(agent, env):

    s, _, game_over, player_turn = env.reset()
    state_size = 17

    states = []
    actions = []
    rewards = []

    states1 = []
    actions1 = []
    rewards1 = []

    episode_reward = [0.0, 0.0]

    while not game_over:

        player_turn_temp = env.get_player_turn()
        env.roll_dice()[0]
        player_turn = env.get_player_turn()

        if player_turn == player_turn_temp:

            action_list = env.get_next_states(player_turn)

            if action_list:
                if player_turn > -1:
                    # Process state and query agent for action
                    s_ = env.convert_state(0)
                    action = agent.act(s_, action_list)
                else:
                    action = choose_rand(action_list)

                new_s, reward, game_over, player_turn_temp, _ = env.make_step(action)

                if player_turn == 0:
                    # Store S, A, R in episode trajectory
                    states.append(np.reshape(agent.preprocess(s_), [1, state_size]))
                    actions.append(action)
                    rewards.append(reward[player_turn])

                elif player_turn == 1:
                    states1.append(np.reshape(agent.preprocess(s_), [1, state_size]))
                    actions1.append(action)
                    rewards1.append(reward[player_turn])

                episode_reward[player_turn] += reward[player_turn]

    episode = (states, actions, rewards)
    episode1 = (states1, actions1, rewards1)

    return episode, episode1