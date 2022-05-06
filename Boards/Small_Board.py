import numpy as np


def _seed_random():
    np.random.seed(None)


class SmallBoard:
    def __init__(self):
        self.reward = [0.0, 0.0]
        self.players = 2
        self.player_turn = 0
        self.pieces = 2

        self.state = [0, 16, 16, 16, 16]
        self.roll_dice()

        self.reset()

        self.action_list = []
        self.state_list = []

    def roll_dice(self):
        self.state[0] = np.random.randint(1, 5)
        return self.state

    # MUST BE CALLED BEFORE MAKING A STEP
    def get_next_states(self, p):
        # Finds and returns valid moves with relating after-state
        self.find_possible_moves(p)
        # If no pieces out the player gets 3x attempts
        if not self.has_options(p) and not self.action_list:
            self.roll_dice()
            self.find_possible_moves(p)
            if not self.has_options(p) and not self.action_list:
                self.roll_dice()
                self.find_possible_moves(p)
        if not self.action_list:
            self.player_turn = (self.player_turn + 1) % self.players
        return self.action_list, self.state_list

    # Determine if the player has any pieces on the board
    def has_options(self, p):
        opt = False
        for i in range(0, self.pieces):
            val = self.state[1 + self.pieces * p + i]
            if val != 16 and val != 0:
                opt = True
        return opt

    # Step environment by one
    def make_step(self, a):
        self.state = self.state_list[self.action_list.index(a)]
        self.player_turn = (self.player_turn + 1) % self.players
        game_over = self.is_game_over()
        if self.is_game_over() != -1:
            for i in range(0, self.players):
                self.reward[i] = 1.0 if i == game_over else 0.0
        return self.state, self.reward, (game_over != -1), self.player_turn

    def state_size(self):
        return len(self.state)

    def action_size(self):
        return self.pieces

    def is_game_over(self):
        for i in range(0, self.players):
            won = True
            for j in range(0, self.pieces):
                if self.state[1 + j + i * self.pieces] != 0:
                    won = False
            if won:
                return i
        return -1

    def find_possible_moves(self, p):
        self.action_list = []
        self.state_list = []
        # If a 6 is rolled, the player can move any pieces within the start region
        if self.state[0] == 4:
            for i in range(0, self.pieces):
                if self.state[1 + p * self.pieces + i] == 16:
                    if self.check_player_pieces(15, p):
                        temp_state = np.copy(self.state)
                        temp_state[1 + p * self.pieces + i] = 15

                        is_pos_occupied = self.check_opp_pieces(15, p)
                        if is_pos_occupied != 0:
                            temp_state[is_pos_occupied] = 16
                        self.state_list.append(temp_state)
                        self.action_list.append(i)
        # Any piece not within the start region can also move STATE[0] squares
        for j in range(0, self.pieces):
            if self.state[1 + p * self.pieces + j] != 16:
                temp_state = np.copy(self.state)
                new_pos = temp_state[1 + p * self.pieces + j] - self.state[0]
                if new_pos >= 0 and self.check_player_pieces(new_pos, p):
                    temp_state[1 + p * self.pieces + j] = new_pos
                    is_pos_occupied = self.check_opp_pieces(new_pos, p)
                    if is_pos_occupied != 0:
                        temp_state[is_pos_occupied] = 16
                    self.state_list.append(temp_state)
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
        if pos > 3:
            op_player = (p + 1) % self.players
            new_pos = (pos + 6) % 12
            if new_pos < 4:
                new_pos += 12
            for j in range(0, self.pieces):
                if self.state[1 + op_player * self.pieces + j] == new_pos:
                    return 1 + op_player * self.pieces + j
        return 0

    # Reset environment to start
    def reset(self):
        _seed_random()
        self.state = [self.roll_dice(), 16, 16, 16, 16]
        self.reward = [0.0, 0.0]
        self.player_turn = np.random.randint(0, 2)
        return self.state, self.reward, False, self.player_turn
