import chess
import log

logger = log.getLogger("engine.strength")

class StrengthManager:
    def __init__(self, engine_):
        self.engine = engine_
    
    def double_prob_accounts_for_cp(self, board=None):
        if board is None:
            board = self.engine.current_board
        return 150 * (1 - self.game_stage(board))

    def game_stage(self, board=None):
        if board is None:
            board = self.engine.current_board
        return 1 - float(self.material_value(board)) / 78

    def material_value(self, board=None):
        if board is None:
            board = self.engine.current_board
        values = [0, 1, 3, 3, 5, 9, 0]
        sum = 0
        for square in chess.SquareSet(chess.BB_ALL):
            piece = board.piece_type_at(square)
            if piece:
                sum += values[piece]
        return sum