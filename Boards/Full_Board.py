import numpy as np


def _seed_random():
    np.random.seed(None)


class FullBoard:
    def __init__(self):
        self.reward = [0.0, 0.0, 0.0, 0.0]
        self.players = 4
        self.player_turn = 0
        self.pieces = 4

        self.state = [0, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 4, 0, 4, 0, 4, 0, 4, 0]
        self.reset()

        self.action_list = []
        self.state_list = []
        # self.action_types = []

    def roll_dice(self):
        self.state[0] = np.random.randint(1, 7)
        return self.state

    def has_options(self, p):
        opt = False
        for i in range(0, self.pieces):
            val = self.state[1 + self.pieces * p + i]
            if val != 62 and val != 0:
                opt = True
        return opt

    def get_next_states(self, p):
        self.find_possible_moves()
        # Roll up to 2 more times if no pieces are out
        if not self.has_options(p) and not self.action_list:
            self.roll_dice()
            self.find_possible_moves()
            if not self.has_options(p) and not self.action_list:
                self.roll_dice()
                self.find_possible_moves()
        if not self.action_list:
            self.player_turn = (self.player_turn + 1) % self.players
        return self.action_list

    # Step state,
    def make_step(self, a):
        self.state = self.state_list[self.action_list.index(a)]
        game_over = self.is_game_over()
        self.reward = [0.0, 0.0, 0.0, 0.0]
        if self.is_game_over() != -1:
            self.reward[game_over] = 1.0
        else:
            self.player_turn = (self.player_turn + 1) % self.players
        return self.state, self.reward, (game_over != -1), self.player_turn

    def state_size(self):
        return len(self.state)

    def action_size(self):
        return self.pieces

    # Checks if any player has all pieces == 0
    def is_game_over(self):
        for i in range(0, self.players):
            won = True
            for j in range(0, self.pieces):
                if self.state[1 + j + i * 4] != 0:
                    won = False
            if won:
                return i
        # No winner
        return -1

    # Finds all moves available to the current player and the corresponding after-state
    def find_possible_moves(self):
        p = self.player_turn
        self.action_list = []
        self.state_list = []
        # If a 6 is rolled, the player can move any pieces within the start region (with value 62)
        if self.state[0] == 6:
            for i in range(0, self.pieces):
                if self.state[1 + p * self.pieces + i] == 62:
                    if self.check_player_pieces(61, p):  # Valid move
                        temp_state = np.copy(self.state)
                        temp_state[1 + p * self.pieces + i] = 61
                        is_pos_occupied = self.check_opp_pieces(61, p)

                        if is_pos_occupied != 0:  # move knocks off opponent
                            temp_state[is_pos_occupied] = 62

                        self.state_list.append(self._process(temp_state))
                        self.action_list.append(i)
        # Any piece not within the start region can also move STATE[0] squares
        for j in range(0, self.pieces):
            if self.state[1 + p * self.pieces + j] != 62:
                temp_state = np.copy(self.state)
                new_pos = temp_state[1 + p * self.pieces + j] - self.state[0]

                if new_pos >= 0 and self.check_player_pieces(new_pos, p):  # Valid move
                    temp_state[1 + p * self.pieces + j] = new_pos
                    is_pos_occupied = self.check_opp_pieces(new_pos, p)

                    if is_pos_occupied != 0:  # move knocks off opponent
                        temp_state[is_pos_occupied] = 62

                    self.state_list.append(self._process(temp_state))
                    self.action_list.append(j)

    # Checks to see if the new position is occupied by the player's own piece
    def check_player_pieces(self, pos, p):
        if pos != 0:
            for i in range(0, self.pieces):
                if self.state[1 + p * self.pieces + i] == pos:
                    return False
        return True

    # Checks to see if the new position is occupied by an opponent's piece
    # Returns the index value (from STATE) for the piece
    def check_opp_pieces(self, pos, p):
        if pos > 5:
            for i in range(1, self.players):
                op_player = (p + i) % self.players
                new_pos = pos + 14 * i

                if new_pos >= 62:
                    new_pos = (new_pos % 62) + 6

                for j in range(0, self.pieces):
                    if self.state[1 + op_player * self.pieces + j] == new_pos:
                        return 1 + op_player * self.pieces + j
        return 0

    def reset(self):
        _seed_random()
        self.state = [0, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 4, 0, 4, 0, 4, 0, 4, 0]
        self.roll_dice()
        self.reward = [0.0, 0.0, 0.0, 0.0]
        self.player_turn = np.random.randint(0, 4)
        return self.state, self.reward, False, self.player_turn

    # Convert the current state to point of view for player p
    # Optionally process specific state s
    def convert_state(self, p, s=None):
        state_ = np.copy(self.state)
        if s:
            state_ = np.copy(s)

        # Cycle the state representation such that the current player is in position 0
        while p != 0:
            state_ = self._cycle_viewpoint(state_)
            p += 1
            p = p % 4
        # For each other player & their pieces
        for i in range(1, 4):
            for j in range(0, 4):
                # Get the position of the piece
                pos = state_[1 + self.pieces * i + j]
                # 0 inaccessible pieces and convert the rest
                if pos == 62:
                    new_pos = 0
                elif pos < 6:
                    new_pos = 0
                else:
                    new_pos = pos - (14 * i)
                    if new_pos < 6:
                        new_pos = 62 + new_pos - 6
                # update position in state_
                state_[1 + self.pieces * i + j] = new_pos

        return state_

    # Calculate s & h for each player and fix state
    def _process(self, s):
        for i in range(0, 4):
            home = 0
            start = 0
            for j in range(0, 4):
                if s[1 + self.pieces * i + j] == 62:
                    start += 1
                if s[1 + self.pieces * i + j] == 0:
                    home += 1
            s[1 + self.pieces * self.players + 2 * i] = start
            s[1 + self.pieces * self.players + 2 * i + 1] = home

        return s

    def _cycle_viewpoint(self, s):
        return [*[s[0]], *s[13:17], *s[1:13], *s[23:25], *s[17:23]]

    def get_state(self):
        return self.state
