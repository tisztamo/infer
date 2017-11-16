import chess
from hint import Hint

class DevelopAllPiecesHint():
    def __init__(self):
        pass

    def getHint(self, board, move, best_move):
        if board.fullmove_number < 4 or not board.in_opening():
            #print("Not in opening" + str(board.fullmove_number))
            return None
        
        if board.piece_type_at(move.from_square) == chess.KING:
            #print("Moving with the king")
            return None

        move_from = chess.SquareSet.from_square(move.from_square)
        if len(move_from.intersection(board.my_rank())) > 0:
            #print("Moving from my rank")
            return None

        best_move_from = chess.SquareSet.from_square(best_move.from_square)
        if len(best_move_from.intersection(board.my_rank())) == 0:
            #print("The best move is similar: " + str(best_move))
            return None

        #print("The best move: " + str(best_move))
        return Hint(1, "Rapidly develop all pieces")
