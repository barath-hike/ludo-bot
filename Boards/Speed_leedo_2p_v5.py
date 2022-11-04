import numpy as np


def _seed_random():
    np.random.seed(None)


class FullBoard:
    def __init__(self):
        self.reward = [0.0, 0.0]
        self.players = 2
        self.player_turn = 0
        self.pieces = 4

        self.state = [0, 1, 1, 1, 1, 27, 27, 27, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.state_reward = [0, 0, 0, 0, 0, 0, 0, 0]

        self.state_reward_p = {}
        self.state_reward_p['0'] = [0, 0, 0, 0, 0, 0, 0, 0]
        self.state_reward_p['1'] = [0, 0, 0, 0, 0, 0, 0, 0]

        self.start = [1, 27]
        self.inflect = [51, 25]
        self.new_start = [51, 63]
        self.end = [57, 69]

        self.six_flag = 0

        self.turns = 0
        self.max_value = 69
        self.reset()

        self.safe = [1, 9, 14, 22, 27, 35, 40, 48]
        self.action_list = []
        self.state_list = []
        self.cut_list = []
        self.home_list = []

    def roll_dice(self):

        self.state[0] = np.random.randint(1, 7)

        if self.state[0] == 6:
            self.six_flag += 1
        else:
            self.six_flag = 0

        if self.six_flag == 3:
            self.player_turn = (self.player_turn + 1) % self.players
            self.turns += 1
            self.six_flag = 0

        return self.state

    def get_next_states(self, p):
        self.find_possible_moves()
        if not self.action_list:
            self.player_turn = (self.player_turn + 1) % self.players
            self.turns += 1
        return self.action_list

    def get_player_turn(self):
        return self.player_turn


    # Step state,
    def make_step(self, a):
        self.state = np.copy(self.state_list[self.action_list.index(a)]).tolist()
        old_state_reward = np.copy(self.state_reward_p[str(int(self.player_turn))]).tolist()
        self.state_reward_p[str(int(self.player_turn))] = np.copy(self.state_reward_list[self.action_list.index(a)]).tolist()
        self.state_reward = np.copy(self.state_reward_list[self.action_list.index(a)]).tolist()

        if self.cut_list[self.action_list.index(a)] == 1:
            self.six_flag = 0

        if self.home_list[self.action_list.index(a)] == 1:
            self.six_flag = 0

        if self.state[0] != 6 and self.cut_list[self.action_list.index(a)]==0 and self.home_list[self.action_list.index(a)]==0:
            self.player_turn = (self.player_turn + 1) % self.players
            self.turns += 1

        game_over = self.is_game_over()

        self.reward = [np.sum(self.state_reward[:4]) - np.sum(old_state_reward[:4]),
                       np.sum(self.state_reward[4:8]) - np.sum(old_state_reward[4:8])]

        return self.state, self.reward, (game_over != -1), self.player_turn


    def state_size(self):
        return len(self.state)

    def action_size(self):
        return self.pieces

    def max_val(self):
        return self.max_value

    def return_state(self):
        return self.state

    # Checks if any player has all pieces == 0
    def is_game_over(self):
        for i in range(0, self.players):
            won = True
            for j in range(0, self.pieces):
                if self.state[1 + j + i * self.pieces] != self.end[i]:
                    won = False
            if won:
                return i
        if self.turns >= 100:
            return int(np.sum(self.state_reward[:4]) < np.sum(self.state_reward[4:]))
        # No winner
        return -1


    # Finds all moves available to the current player and the corresponding after-state
    def find_possible_moves(self):
        p = self.player_turn
        self.action_list = []
        self.state_list = []
        self.state_reward_list = []
        self.cut_list = []
        self.home_list = []

        for j in range(0, self.pieces):

            temp_state = np.copy(self.state).tolist()
            temp_state_reward = np.copy(self.state_reward).tolist()

            old_pos = temp_state[1 + p * self.pieces + j]
            if old_pos != self.end[p]:
                new_pos = (old_pos + self.state[0])

                if new_pos > self.inflect[p] and old_pos <= self.inflect[p]:
                    new_pos = (new_pos - self.inflect[p]) + self.new_start[p]
                elif old_pos < 52 and new_pos >= 52 and p != 0:
                    new_pos = new_pos % 52

                if new_pos <= self.end[p]:
                    temp_state[1 + p * self.pieces + j] = new_pos
                    temp_state_reward[p * self.pieces + j] += self.state[0]

                    is_pos_occupied = self.check_opp_pieces(new_pos, p)

                    if is_pos_occupied != 0 and is_pos_occupied not in self.safe:  # move knocks off opponent
                        self.cut_list.append(1)
                        self.cut_flag = True
                        opp = int((is_pos_occupied-1) / self.pieces)
                        temp_state[is_pos_occupied] = self.start[opp]

                        temp_state_reward[p * self.pieces + j] += temp_state_reward[is_pos_occupied-1]
                        temp_state_reward[is_pos_occupied-1] = 0
                    else:
                        self.cut_list.append(0)

                    temp_state = self._process(temp_state)
                    if new_pos == self.end[p] and old_pos < self.end[p]:
                        temp_state_reward[p * self.pieces + j] *= (1 + temp_state[1 + 2 * self.players * self.pieces + p])
                        self.home_list.append(1)
                    else:
                        self.home_list.append(0)

                    temp_state[(1 + self.players * self.pieces):(1 + 2 * self.players * self.pieces)] = np.copy(temp_state_reward).tolist()

                    self.state_list.append(temp_state)
                    self.state_reward_list.append(temp_state_reward)
                    self.action_list.append(j)


    def check_opp_pieces(self, pos, p):
        clash = [i for i, e in enumerate(self.state) if e == pos and i > 0 and i < (self.players * self.pieces + 1) 
                            and int((i-1) / self.pieces) != p]
        clash1 = [i for i, e in enumerate(self.state) if e == pos and i > 0 and i < (self.players * self.pieces + 1)]

        if len(clash1) == 1 and len(clash) == 1:
            return clash[0]
        else:
            return 0


    def reset(self):
        _seed_random()
        self.state = [0, 1, 1, 1, 1, 27, 27, 27, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.state_reward = [0, 0, 0, 0, 0, 0, 0, 0]
        self.state_reward_p['0'] = [0, 0, 0, 0, 0, 0, 0, 0]
        self.state_reward_p['1'] = [0, 0, 0, 0, 0, 0, 0, 0]
        
        self.roll_dice()
        self.reward = [0.0, 0.0]
        self.turns = 0
        self.player_turn = np.random.randint(0, self.players)
        return self.state, self.reward, False, self.player_turn


    # Convert the current state to point of view for player p
    # Optionally process specific state s
    def convert_state(self, p, s=None):
        state_ = np.copy(self.state).tolist()
        if s:
            state_ = np.copy(s).tolist()

        # Cycle the state representation such that the current player is in position 0
        if p != 0:
            state_ = self._cycle_viewpoint(state_)

        return state_


    # Calculate s & h for each player and fix state
    def _process(self, s):
        for i in range(0, self.players):
            home = 0
            for j in range(0, self.pieces):
                if s[1 + self.pieces * i + j] == self.end[i]:
                    home += 1
            s[1 + 2 * self.pieces * self.players + i] = home
        return s

    def _cycle_viewpoint(self, s):
        return [*[s[0]], *s[5:9], *s[1:5], *s[13:17], *s[9:13], s[18], s[17]]

    def get_state(self):
        return self.state