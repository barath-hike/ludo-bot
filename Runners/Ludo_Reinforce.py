import os

import numpy as np
from tqdm import tqdm

from Full_Board import FullBoard
from ReinforceAgent import REINFORCEAgent


def choose_rand(a):
    return np.random.choice(a)


def run_game(num_ep, model_output):
    env = FullBoard()
    agent0 = REINFORCEAgent(env.state_size(), env.action_size())
    batch_size = 200

    agent0_reward = []
    agent1_reward = []
    agent2_reward = []
    agent3_reward = []
    episode_length = []
    output_dir = 'model_output/REINFORCE/%s/' % model_output

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for ep in tqdm(range(0, num_ep), ascii=True, unit="e"):

        batch = [[], []]
        rew_path = []
        r0_ = 0
        r1_ = 0
        r2_ = 0
        r3_ = 0
        p_count = 0.0
        # Generate batches of experience
        for it in range(0, batch_size):
            traj, episode_reward, step_count = generate_episode(agent0, env)
            # Append trajectories and calculate average reward per episode
            batch[0].extend(traj[0])
            batch[1].extend(traj[1])
            rew_path.append(traj[2])
            r0_ += episode_reward[0]
            r1_ += episode_reward[1]
            r2_ += episode_reward[2]
            r3_ += episode_reward[3]
            p_count += (step_count / 4)

        # Store episode information
        agent0_reward.append(r0_ / batch_size)
        agent1_reward.append(r1_ / batch_size)
        agent2_reward.append(r2_ / batch_size)
        agent3_reward.append(r3_ / batch_size)
        episode_length.append(p_count / batch_size)
        # Learn from experience
        agent0.learn(batch, agent0.get_returns(rew_path))

        if ep % 100 == 0:
            agent0.save(output_dir + "weights_" + '{:04d}'.format(ep) + ".hdf5")
        if ep % 5 == 0:
            ave_reward = np.average(agent0_reward[-10:])
            print("e={:d}".format(ep), "r={:.3f}".format(r0_ / batch_size),
                  "ave={:.3f}".format(ave_reward))
    print(agent0_reward)
    print(agent1_reward)
    print(agent2_reward)
    print(agent3_reward)
    return [agent0_reward, agent1_reward, agent2_reward, agent3_reward, episode_length]


def generate_episode(agent, env):
    s, _, game_over, player_turn = env.reset()
    state_size = env.state_size()
    step = 0
    states = []
    actions = []
    rewards = []
    episode_reward = [0.0, 0.0, 0.0, 0.0]
    while not game_over:
        # Roll dice and retrieve valid actions
        env.roll_dice()
        action_list = env.get_next_states(player_turn)

        if action_list:
            if player_turn == 0:
                # Process state and query agent for action
                s_ = env.convert_state(0)
                action = agent.act(s_, action_list)
            else:
                action = choose_rand(action_list)

            new_s, reward, game_over, player_turn_temp = env.make_step(action)

            if player_turn == 0:
                # Store S, A, R in episode trajectory
                states.append(np.reshape(s_, [1, state_size, 1]) / 62)
                actions.append(action)
                rewards.append(reward[0])

            episode_reward[player_turn] += reward[player_turn]
            if not game_over:
                player_turn = player_turn_temp
            step += 1
        else:
            player_turn = (player_turn + 1) % 4

    episode = (states, actions, rewards)

    return episode, episode_reward, step
