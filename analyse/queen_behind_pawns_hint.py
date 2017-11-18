import chess
from hint import Hint

class QueenBehindPawnsHint():
    def __init__(self):
        pass

    def getHint(self, board, move, best_move):
        if not board.in_opening():
            return None
        if board.piece_type_at(move.from_square) != chess.QUEEN:
            return None

        first_pawn_rank = board.first_pawn_rank_in_file(chess.square_file(move.to_square))
        queen_rank = chess.square_rank(move.to_square)

        if board.turn == chess.WHITE and first_pawn_rank < queen_rank or \
            board.turn == chess.BLACK and first_pawn_rank > queen_rank:
            return Hint(7, "Place the Queen behind the line of friendly pawns during opening")
        
        return None