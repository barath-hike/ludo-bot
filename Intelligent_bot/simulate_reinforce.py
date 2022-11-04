import numpy as np
from tqdm import tqdm

from scipy.stats import truncnorm

import sys
sys.path.append('../')
sys.path.append('../../')
from DataBot.data_bot import BotMLP
from Boards.Speed_leedo_2p_v6 import FullBoard
from Intelligent_bot.intelligent_bot import Bot

def simulate(output_dir):

    env = FullBoard()
    agent0 = BotMLP()
    bot0 = Bot()
    
    wins = []

    agent0.load_model(output_dir)

    num_ep = 10000

    for ep in tqdm(range(0, num_ep)):
        s, _, game_over, player_turn = env.reset()
        episode_reward = [0.0, 0.0]

        time = 0
        while time < 360 or player_turn == 0:

            time += truncnorm.rvs(-2, 2, loc=3, scale=1)
            player_turn_temp = env.get_player_turn()
            env.roll_dice()[0]
            player_turn = env.get_player_turn()

            if player_turn == player_turn_temp:

                action_list = env.get_next_states(player_turn)

                if action_list:
                    if player_turn == 0:
                        s_t = env.return_state()
                        action = agent0.act(np.array(s_t[:-2]), action_list)
                    else:
                        s_t = env.return_state()
                        action = bot0.act(state=s_t, p=player_turn)

                    time += truncnorm.rvs(-2, 2, loc=3, scale=1)
                    s_, reward, game_over, player_turn_temp, game_reward = env.make_step(action)

                    episode_reward[player_turn] += reward[player_turn]
        
        if game_reward[0] > game_reward[1]:
            wins.append(1)
        else:
            wins.append(0)

    return np.sum(wins)/num_ep