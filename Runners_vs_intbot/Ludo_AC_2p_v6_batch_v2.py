import os

import numpy as np
from tqdm import tqdm

from Agents.A2CAgent_batch_v2 import Agent
from Boards.Speed_leedo_2p_v6 import FullBoard
from Intelligent_bot.intelligent_bot import Bot
from Intelligent_bot.simulate import simulate
import gc

from slack import WebClient
bot_name='A2C Bot-1'
slack_client=WebClient(token='xoxb-2151902985-2324529348736-eNKHbuseT407uHVBjejHGJvy')
icon_url=''
channel='#rl-bot'

def choose_rand(a):
    return np.random.choice(a)


def run_game(num_ep, model_output):
    
    env = FullBoard()
    agent0 = Agent(n_actions=env.action_size(), input_dim=17, alpha=1e-8, max_val=env.max_val())
    agent0.load_actor('../../DataBot/models/mlp_bot_models_v2/model_84.hdf5')
    print('model_84.hdf5')

    # output_dir_a = '../trainers_vs_intbot/model_output_vs_intbot/A2C_v6_batch_v2/0001/actor/weights_490000.hdf5'
    # output_dir_c = '../trainers_vs_intbot/model_output_vs_intbot/A2C_v6_batch_v2/0001/critic/weights_490000.hdf5'

    # agent0.load_model(output_dir_a, output_dir_c)

    bot0 = Bot()

    wins = []

    agent0_reward = []
    agent1_reward = []
    episode_length = []

    output_dir_a = 'model_output_vs_intbot/A2C_v6_batch_v2/%s/actor/' % model_output
    output_dir_c = 'model_output_vs_intbot/A2C_v6_batch_v2/%s/critic/' % model_output

    if not os.path.exists(output_dir_a):
        os.makedirs(output_dir_a)
    if not os.path.exists(output_dir_c):
        os.makedirs(output_dir_c)

    for ep in tqdm(range(0, num_ep), ascii=True, unit="e"):
        step = 0
        s, _, game_over, player_turn = env.reset()
        episode_reward = [0.0, 0.0, 0.0, 0.0]

        param1 = []
        param2 = []
        param3 = []
        param4 = []
        param5 = []

        while not game_over:

            player_turn_temp = env.get_player_turn()
            env.roll_dice()[0]
            player_turn = env.get_player_turn()

            if player_turn == player_turn_temp:

                action_list = env.get_next_states(player_turn)

                if action_list:
                    if player_turn == 0:
                        s_t = env.convert_state(player_turn)
                        action = agent0.act(s_t[:-2], action_list)
                    else:
                        s_t = env.return_state()
                        action = bot0.act(state=s_t, p=player_turn)

                    s_, reward, game_over, player_turn_temp, _ = env.make_step(action)

                    if player_turn == 0:
                        s_t_ = env.convert_state(player_turn)
                        param1.append(s_t[:-2])
                        param2.append(action)
                        param3.append(reward[player_turn])
                        param4.append(s_t_[:-2])
                        param5.append(game_over)

                    episode_reward[player_turn] += reward[player_turn]

                    step += 1

            if game_over:
                agent0.learn(np.array(param1), np.array(param2), np.array(param3), np.array(param4), np.array(param5))

                if ep % 100 == 0:
                    gc.collect()
                    
                if episode_reward[0] > 0:
                    wins.append(1)
                else:
                    wins.append(0)
                # print('Wins: ', np.sum(wins[-100:]))

                agent0_reward.append(episode_reward[0])
                agent1_reward.append(episode_reward[1])
                episode_length.append(step / 2)

        if ep > 1000:
            agent0.reduce_alpha()

        if ep % 100 == 0:
            # print(np.average(agent0_reward[-1000:]), np.average(agent1_reward[-1000:]))
            agent0.save_model(output_dir_a + "weights_" + '{:04d}'.format(ep) + ".hdf5", 
                              output_dir_c + "weights_" + '{:04d}'.format(ep) + ".hdf5")

        if ep % 10000 == 0 and ep != 0:
            win_rate = simulate(output_dir_a + "weights_" + '{:04d}'.format(ep) + ".hdf5", 
                     output_dir_c + "weights_" + '{:04d}'.format(ep) + ".hdf5")
            message = 'Win rate against intelligent bot after {0} episodes: {rate:0.2f}%'.format(ep, rate=win_rate*100)
            slack_client.chat_postMessage(channel=channel,
                          text=message,
                          username=bot_name,
                          icon_url=icon_url)

    return [agent0_reward, agent1_reward, episode_length]
