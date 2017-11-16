import chess
from hint import Hint

class DontOpenWhenLateHint():
    def __init__(self):
        pass

    def getHint(self, board, move, best_move):
        if not board.in_opening():
            return None
        if board.piece_type_at(move.to_square) != chess.PAWN \
            or board.piece_type_at(best_move.to_square) == chess.PAWN:
            return None
        if not board.is_in_bigcenter(move.to_square):
            return None
        developed_pieces = board.num_developed_pieces()
        if developed_pieces[int(board.turn)] < developed_pieces[1 - int(board.turn)]:
            return Hint(6, "Don't open a position if you are late in development")
        return None