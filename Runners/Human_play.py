import numpy as np

from Agents.DQNAgent import DQNAgent
from Boards.Full_Board import FullBoard


def choose_rand(a):
    return np.random.choice(a)


def choose_human(s, act, skip):
    if skip and len(act) == 1:
        return act[0]
    print("===================")
    print("    PLAYER TURN    ")
    print("===================")
    print(s)
    print("===================")
    print(act)

    while True:
        action = input("Please choose an action: ")
        try:
            val = int(action)
            if val in act:
                return val
        except ValueError:
            "Error: input must be an integer and a valid move"


def run_game(skip_single, agent_weights):
    env = FullBoard()
    agent0 = DQNAgent(env.state_size(), env.action_size())
    agent0.load(agent_weights)
    print(env.state_size(), env.action_size())

    s, _, game_over, player_turn = env.reset()
    step = 0
    while not game_over:
        # Roll dice and retrieve valid actions
        env.roll_dice()
        action_list = env.get_next_states(player_turn)

        if action_list:
            if player_turn == 0:
                # Process state and query agent for action
                s_ = env.convert_state(0)
                action = choose_human(s_, action_list, skip_single)
            else:
                s_ = env.convert_state(player_turn)
                action = agent0.act(s_, action_list)

            new_s, reward, game_over, player_turn_temp = env.make_step(action)

            if game_over:
                print(new_s)
                print("!!!!!!!!!!!!!!!!!!!!!!!")
                print("   WINNER: PLAYER {}   ".format(player_turn))
                print("!!!!!!!!!!!!!!!!!!!!!!!")

            player_turn = player_turn_temp
            step += 1

        else:
            player_turn = (player_turn + 1) % 4


if __name__ == '__main__':
    skip_single_actions = True
    run_game(skip_single_actions)
