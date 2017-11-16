import chess
from hint import Hint

class UnnecessaryPawnMoveHint():
    def __init__(self):
        pass

    def getHint(self, board, move, best_move):
        if not board.in_opening():
            return None
        
        if board.piece_type_at(move.from_square) != chess.PAWN:
            return None

        if board.piece_type_at(best_move.from_square) == chess.PAWN:
            #print("Best move is also with a pawn")
            return None

        if board.is_in_bigcenter(move.to_square):
            #print("Attacking the center")
            return None

        if move.to_square in [chess.B3, chess.G3, chess.B6, chess.G6]:
            #print("Indian")
            return None

        return Hint(4, "Do not make unnecessary pawn moves during opening")
