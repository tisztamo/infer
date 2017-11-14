import chess
from hint import Hint

class DevelopKnightsFirstHint():
    def __init__(self):
        pass

    def getHint(self, board, move, best_move):
        if len(board.move_stack) < 2 or not board.in_opening():
            return None
        if board.piece_type_at(move.from_square) != chess.BISHOP:
            return None

        my_knights = board.pieces(chess.KNIGHT, board.turn)
        my_bishops = board.pieces(chess.BISHOP, board.turn)
        if len(my_knights.intersection(board.my_rank())) == 2 \
            and len(my_bishops.intersection(board.my_rank())) < 2: #the last bishop is coming out or the first one moves twice
            return Hint("#2 Develop Knights before bishops")
        return None