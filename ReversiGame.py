import numpy as np

class Piece:
    BLACK = -1
    WHITE = 1
    EMPTY = 0


class ReversiGame:
    def __init__(self, opposing_agent):
        self.opponent = opposing_agent

        # setup the board
        self.board = np.zeros((8,8))

        self.invalid_reward = -100
        self.win_reward = 1
        self.lose_reward = -1
        self.default_reward = 0


    def reset(self):
        self.board = np.zeros((8,8))
        self.board[3, 3] = Piece.WHITE
        self.board[3, 4] = Piece.BLACK
        self.board[4, 3] = Piece.BLACK
        self.board[4, 4] = Piece.WHITE

        self.turn = Piece.BLACK

        if np.random.random() < 0.5:
            self.__opponentmakessmove__()
            self.playerturn = Piece.WHITE
        else:
            self.playerturn = Piece.BLACK

        return self.__makestate__()


    def step(self, action):
        assert self.turn == self.playerturn

        current_turn = self.turn
        row, col = self.__idx2move__(action)

        valid = self.__makemove__(row, col, current_turn, True)

        # terminate immediately with invalid move
        if not valid:
            return self.__makestate__(), self.invalid_reward, True, None


        # opponent makes a move, goes until not their turn anymore
        while self.turn == -self.playerturn:
            self.__opponentmakessmove__()

        # compute reward
        reward = 0
        if self.turn == Piece.EMPTY:
            reward = current_turn * self.__findresult__()


        return self.__makestate__(), reward, self.turn == Piece.EMPTY, None


    def set_opponent(self, opposing_agent):
        self.opponent = opposing_agent


    def __idx2move__(self, idx):
        row = idx // 8
        col = idx % 8

        return row, col


    def __makemove__(self, row, col, color, mod):
        if row >= 8 or col >= 8 or row < 0 or col < 0:
            return False
        if self.board[row, col] != Piece.EMPTY:
            return False

        legal = False

        # flip downward pieces
        # i = row + 1; i < 8; i++
        for i in range(row + 1, 8):
            # matching piece on other end
            if self.board[i, col] == color:
                # retrace back
                if mod:
                    for j in range(i - 1, row, -1):
                        self.board[j, col] = color

                # legal if made it past the first one
                if i > row + 1:
                    legal = True
                break

            elif self.board[i, col] == Piece.EMPTY:
                break

        # flip upward pieces
        # i = row -1; i >= 0; i--
        for i in range(row - 1, -1, -1):
            if self.board[i, col] == color:
                if mod:
                    for j in range(i + 1, row):
                        self.board[j, col] = color

                if i < row - 1:
                    legal = True
                break

            elif self.board[i, col] == Piece.EMPTY:
                break

        # flip leftward pieces
        # i = col - 1; i >= 0; i--
        for i in range(col - 1, -1, -1):
            if self.board[row, i] == color:
                if mod:
                    for j in range(i + 1, col):
                        self.board[row, j] = color

                if i < col - 1:
                    legal = True
                break

            elif self.board[row, i] == Piece.EMPTY:
                break

        # flip rightward pieces
        # i = col + 1; i < 8; i++
        for i in range(col + 1, 8):
            if self.board[row, i] == color:
                if mod:
                    for j in range(i - 1, col, -1):
                        self.board[row, j] = color

                if i > col + 1:
                    legal = True
                break

            elif self.board[row, i] == Piece.EMPTY:
                break

        # flip northwest diagonal pieces
        # i = 1; i <= min(row, col); i++
        for i in range(1, min(row, col) + 1):
            if self.board[row - i, col - i] == color:
                if mod:
                    for j in range(i - 1, 0, -1):
                        self.board[row - j, col - j] = color

                if i > 1:
                    legal = True
                break
            elif self.board[row - i, col - i] == Piece.EMPTY:
                break

        # flip northeast diagonal pieces
        # i = 1; i <= min(row, 7-col); i++
        for i in range(1, min(row, 7 - col) + 1):
            if self.board[row - i, col + i] == color:
                if mod:
                    for j in range(i - 1, 0, -1):
                        self.board[row - j, col + j] = color

                if i > 1:
                    legal = True
                break
            elif self.board[row - i, col + i] == Piece.EMPTY:
                break

        # flip southwest diagonal pieces
        # i = 1; i <= min(7-row, col); i++
        for i in range(1, min(7 - row, col) + 1):
            if self.board[row + i, col - i] == color:
                if mod:
                    for j in range(i - 1, 0, -1):
                        self.board[row + j, col - j] = color

                if i > 1:
                    legal = True
                break
            elif self.board[row + i, col - i] == Piece.EMPTY:
                break

        # flip southeast diagonal pieces
        # i = 1; i <= min(7-row, 7-col); i++
        for i in range(1, min(7 - row, 7 - col) + 1):
            if self.board[row + i, col + i] == color:
                if mod:
                    for j in range(i - 1, 0, -1):
                        self.board[row + j, col + j] = color

                if i > 1:
                    legal = True
                break
            elif self.board[row + i, col + i] == Piece.EMPTY:
                break

        if mod and legal:
            self.board[row, col] = color
            self.__setnextturn__(-color)

        return legal


    def __opponentmakessmove__(self):
        # arg sort of the move values, reversed
        moves = reversed(np.argsort(self.opponent.get_move_values(self.__makestate__())))

        # keep trying moves until one works
        for move in moves:
            row, col = self.__idx2move__(move)
            if self.__makemove__(row, col, self.turn, True):
                break

    def __checkmoves__(self, color):
        for i in range(8):
            for j in range(8):
                if self.__makemove__(i, j, color, False):
                    return True
        return False


    def __setnextturn__(self, default):
        if not self.__checkmoves__(default):
            if not self.__checkmoves__(-default):
                self.turn = Piece.EMPTY
            else:
                self.turn = -default
        else:
            self.turn = default


    def __makestate__(self):
        return np.multiply(self.turn, self.board)


    def __findresult__(self):
        diff = np.sum(self.board)
        if diff > 0:
            return Piece.WHITE
        elif diff < 0:
            return Piece.BLACK
        else:
            return Piece.EMPTY
