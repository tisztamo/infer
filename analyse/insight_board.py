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

    def num_developed_pieces(self):
        FORE_RANKS_MASK = 0x00ffffffffffff00
        squares = chess.SquareSet(FORE_RANKS_MASK)
        retval = [0, 0]
        for square in squares:
            piece = self.piece_at(square)
            if piece is not None and piece.piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                retval[int(piece.color)] += 1
        return retval

    def copy(self, stack=True):
         board = super(InsightBoard, self).copy(stack)
         return board