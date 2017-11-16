import chess

def from_board(board):
    instance = board.copy()
    instance.__class__ = InsightBoard
    return instance

class InsightBoard(chess.Board):

    def in_opening(self):
        if self.fullmove_number > 10:#TODO better definition of opening (eg. all pieces developed)
            return False
        return True

    def my_rank(self):
        if self.turn == chess.WHITE:
            return chess.BB_RANK_1
        return chess.BB_RANK_8

    def is_in_bigcenter(self, square):
        return chess.square_rank(square) in [2,3,4,5] and chess.square_file(square) in [2,3,4,5]

    def copy(self, stack=True):
         board = super(InsightBoard, self).copy(stack)
         return board