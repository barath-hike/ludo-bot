import numpy as np

class Bot:

    def __init__(self):

        self.state = [0, 1, 1, 1, 1, 27, 27, 27, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.players = 2
        self.player_turn = 0
        self.pieces = 4

        self.mov = [
            list(range(1,58)),
            list(range(27,52)) + list(range(26)) + list(range(64, 70))
        ]

        self.start = [1, 27]
        self.inflect = [51, 25]
        self.new_start = [51, 63]
        self.end = [57, 69]

        self.safe = [1, 9, 14, 22, 27, 35, 40, 48]

        self.pawn_priority = [0, 0, 0, 0]

        self.pawn_pos = [1, 1, 1, 1]
        self.new_pos = [1, 1, 1, 1]
        self.opp_pawn_pos = [1, 1, 1, 1]

        self.pawn_score = [0, 0, 0, 0]
        self.opp_pawn_score = [0, 0, 0, 0]


    def reset(self):
        self.state = [0, 1, 1, 1, 1, 27, 27, 27, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.pawn_priority = [0, 0, 0, 0]

        self.pawn_pos = [1, 1, 1, 1]
        self.new_pos = [1, 1, 1, 1]
        self.opp_pawn_pos = [1, 1, 1, 1]

        self.pawn_score = [0, 0, 0, 0]
        self.opp_pawn_score = [0, 0, 0, 0]


    def act(self, state = None, p = None):

        if state:
            self.state = np.copy(state)
        
        if p is not None:
            self.player_turn = p

        self.pawn_pos = self.state[self.player_turn * self.pieces + 1 : self.player_turn * self.pieces + 1 + self.pieces]

        for i in range(self.pieces):
            idx = self.mov[self.player_turn].index(self.pawn_pos[i]) + self.state[0]
            if idx > len(self.mov[self.player_turn]) - 1:
                self.new_pos[i] = -1
            else:
                self.new_pos[i] = self.mov[self.player_turn][idx]

        self.opp_pawn_pos = self.state[(1 - self.player_turn) * self.pieces + 1 : (1 - self.player_turn) * self.pieces + 1 + self.pieces]

        self.pawn_score = self.state[self.player_turn * self.pieces + self.players * self.pieces + 1 : 
                                        self.player_turn * self.pieces + self.players * self.pieces + 1 + self.pieces]
        self.opp_pawn_score = self.state[(1 - self.player_turn) * self.pieces + self.players * self.pieces + 1 : 
                                            (1 - self.player_turn) * self.pieces + self.players * self.pieces + 1 + self.pieces]

        if np.sum(np.array(self.new_pos) == -1) == 4:
            return -1

        movablePawnIdx = self.setPriorityBasedOnHomeDistance()
        if movablePawnIdx != -1:
            return movablePawnIdx

        movablePawnIdx = self.setPriorityBasedOnOpponentPawnCut()
        if movablePawnIdx != -1:
            return movablePawnIdx

        movablePawnIdx = self.updatePriorityIfGettingSafe()
        if movablePawnIdx != -1:
            return movablePawnIdx
        
        self.decreasePriorityIfInDangerZone()

        sorted_priority = np.argsort(self.pawn_priority)

        if self.pawn_priority[sorted_priority[0]] == np.inf:
            return -1
        else:
            return sorted_priority[0]


    def setPriorityBasedOnHomeDistance(self):
        
        for i in range(self.pieces):
            if self.new_pos[i] == -1:
                self.pawn_priority[i] = np.inf
            else:
                self.pawn_priority[i] = 56 - self.mov[self.player_turn].index(self.new_pos[i])
                if self.pawn_priority[i] == 0:
                    return i

        return -1


    def setPriorityBasedOnOpponentPawnCut(self):

        safe = np.copy(self.safe).tolist()

        for i in range(self.pieces):
            for j in range(self.pieces):
                if i!=j:
                    if self.opp_pawn_pos[i] == self.opp_pawn_pos[j]:
                        if self.opp_pawn_pos[i] not in safe:
                            safe = safe + [self.opp_pawn_pos[i]]

        priority = np.inf
        ridx = -1

        for i in range(self.pieces):
            for j in range(self.pieces):
                if self.new_pos[i] == self.opp_pawn_pos[j] and self.opp_pawn_pos[j] not in safe:
                    self.pawn_priority[i] = self.pawn_priority[i] * 0.1 / self.opp_pawn_score[j]
                    if self.pawn_priority[i] < priority:
                        ridx = i
        
        return ridx
        

    def updatePriorityIfGettingSafe(self):

        priority = np.inf
        ridx = -1

        for i in range(self.pieces):
            if self.new_pos[i] in self.safe or self.new_pos[i] in self.pawn_pos:
                self.pawn_priority[i] = self.pawn_priority[i] * 0.1 / self.pawn_score[i]
                if self.pawn_priority[i] < priority:
                    ridx = i

        return ridx


    def decreasePriorityIfInDangerZone(self):

        for i in range(self.pieces):
            for j in range(self.pieces):
                if self.new_pos[i] not in self.mov[self.player_turn][-6:]:
                    if self.opp_pawn_pos[j] not in self.mov[(1 - self.player_turn)][-6:]:
                        opp_idx = self.mov[(1 - self.player_turn)].index(self.opp_pawn_pos[j])
                        if self.new_pos[i] in self.mov[(1 - self.player_turn)][opp_idx + 1 : opp_idx + 7]:
                            self.pawn_priority[i] = self.pawn_priority[i] * 10