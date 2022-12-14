import numpy as np


def _seed_random():
    np.random.seed(None)


class FullBoard:
    def __init__(self):
        self.reward = [0.0, 0.0, 0.0, 0.0]
        self.players = 4
        self.player_turn = 0
        self.pieces = 4

        # self.state = [0, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 4, 0, 4, 0, 4, 0, 4, 0]
        # self.state = [0, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 0, 0, 0, 0, 0, 0, 0, 0]
        self.state = [0, 1, 1, 1, 1, 14, 14, 14, 14, 27, 27, 27, 27, 40, 40, 40, 40, 0, 0, 0, 0]

        self.start = [1, 14, 27, 40]
        self.inflect = [51, 12, 25, 38]
        self.new_start = [51, 57, 63, 69]
        self.end = [57, 63, 69, 76]

        self.max_value = 76
        self.reset()

        self.safe = [9, 14, 22, 27, 35, 40, 48]
        self.action_list = []
        self.state_list = []
        # self.action_types = []

    def roll_dice(self):
        self.state[0] = np.random.randint(1, 7)
        return self.state

    # def has_options(self, p):
    #     opt = False
    #     for i in range(0, self.pieces):
    #         val = self.state[1 + self.pieces * p + i]
    #         if val != self.end[p]:
    #             opt = True
    #     return opt

    def get_next_states(self, p):
        self.find_possible_moves()
        # Roll up to 2 more times if no pieces are out
        # if not self.has_options(p) and not self.action_list:
        #     self.roll_dice()
        #     self.find_possible_moves()
        #     if not self.has_options(p) and not self.action_list:
        #         self.roll_dice()
        #         self.find_possible_moves()
        # if not self.action_list:
        #     self.player_turn = (self.player_turn + 1) % self.players
        return self.action_list

    # Step state,
    def make_step(self, a):
        self.state = self.state_list[self.action_list.index(a)]
        game_over = self.is_game_over()
        self.reward = [0.0, 0.0, 0.0, 0.0]
        if self.is_game_over() != -1:
            self.reward[game_over] = 1.0
        elif self.state[0] != 6:
            self.player_turn = (self.player_turn + 1) % self.players
        return self.state, self.reward, (game_over != -1), self.player_turn

    def state_size(self):
        return len(self.state)

    def action_size(self):
        return self.pieces

    def max_val(self):
        return self.max_value

    # Checks if any player has all pieces == 0
    def is_game_over(self):
        for i in range(0, self.players):
            won = True
            for j in range(0, self.pieces):
                if self.state[1 + j + i * self.pieces] != self.end[i]:
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
        # if self.state[0] == 6:
        #     for i in range(0, self.pieces):
        #         if self.state[1 + p * self.pieces + i] == 62:
        #             if self.check_player_pieces(61, p):  # Valid move
        #                 temp_state = np.copy(self.state)
        #                 temp_state[1 + p * self.pieces + i] = 61
        #                 is_pos_occupied = self.check_opp_pieces(61, p)

        #                 if is_pos_occupied != 0:  # move knocks off opponent
        #                     temp_state[is_pos_occupied] = 56

        #                 self.state_list.append(self._process(temp_state))
        #                 self.action_list.append(i)
        # Any piece not within the start region can also move STATE[0] squares
        for j in range(0, self.pieces):
            # if self.state[1 + p * self.pieces + j] != 62:
            temp_state = np.copy(self.state)
            old_pos = temp_state[1 + p * self.pieces + j]
            if old_pos != self.end[p]:
                new_pos = (old_pos + self.state[0])

                if new_pos > self.inflect[p] and old_pos <= self.inflect[p]:
                    new_pos = (new_pos - self.inflect[p]) + self.new_start[p]
                elif old_pos < 52 and new_pos >= 52 and p != 0:
                    new_pos = new_pos % 52

                if new_pos <= self.end[p]:
                    temp_state[1 + p * self.pieces + j] = new_pos
                    is_pos_occupied = self.check_opp_pieces(new_pos, p)

                    if is_pos_occupied != 0 and is_pos_occupied not in self.safe:  # move knocks off opponent
                        opp = int((is_pos_occupied-1) / self.pieces)
                        temp_state[is_pos_occupied] = self.start[opp]

                    self.state_list.append(self._process(temp_state))
                    self.action_list.append(j)

    # Checks to see if the new position is occupied by the player's own piece
    # def check_player_pieces(self, pos, p):
    #     if pos != 0:
    #         for i in range(0, self.pieces):
    #             if self.state[1 + p * self.pieces + i] == pos:
    #                 return False
    #     return True

    # Checks to see if the new position is occupied by an opponent's piece
    # Returns the index value (from STATE) for the piece
    def check_opp_pieces(self, pos, p):
        clash = [i for i, e in enumerate(self.state) if e == pos and i > 0 and i < (self.players * self.pieces + 1) 
                            and int((i-1) / self.pieces) != p]

        if len(clash) == 1:
            return clash[0]

        else:
            return 0

        # if pos > 5:
        #     for i in range(1, self.players):
        #         op_player = (p + i) % self.players
        #         new_pos = (pos + 13 * i) % 56

        #         # if new_pos >= 62:
        #         #     new_pos = (new_pos % 62) + 6

        #         for j in range(0, self.pieces):
        #             if self.state[1 + op_player * self.pieces + j] == new_pos:
        #                 return 1 + op_player * self.pieces + j
        # return 0

    def reset(self):
        _seed_random()
        # self.state = [0, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 4, 0, 4, 0, 4, 0, 4, 0]
        self.state = [0, 1, 1, 1, 1, 14, 14, 14, 14, 27, 27, 27, 27, 40, 40, 40, 40, 0, 0, 0, 0]
        self.roll_dice()
        self.reward = [0.0, 0.0, 0.0, 0.0]
        self.player_turn = np.random.randint(0, self.players)
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
            p = p % self.players
        # For each other player & their pieces
        # for i in range(1, 4):
        #     for j in range(0, 4):
        #         # Get the position of the piece
        #         pos = state_[1 + self.pieces * i + j]
        #         # 0 inaccessible pieces and convert the rest
        #         # if pos == 56:
        #             # new_pos = 0
        #         # elif pos < 6:
        #         #     new_pos = 0
        #         # else:
        #         new_pos = pos - (13 * i)
        #             # if new_pos < 6:
        #             #     new_pos = 62 + new_pos - 6
        #         # update position in state_
        #         state_[1 + self.pieces * i + j] = new_pos

        return state_

    # Calculate s & h for each player and fix state
    def _process(self, s):
        for i in range(0, self.players):
            home = 0
            for j in range(0, self.pieces):
                if s[1 + self.pieces * i + j] == self.end[i]:
                    home += 1
            s[1 + self.pieces * self.players + i] = home
        return s

    def _cycle_viewpoint(self, s):
        return [*[s[0]], *s[13:17], *s[1:13], s[20], *s[17:20]]

    def get_state(self):
        return self.state
