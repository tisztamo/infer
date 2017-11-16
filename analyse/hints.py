from same_piece_twice_hint import SamePieceTwiceHint
from develop_knights_first_hint import DevelopKnightsFirstHint
from develop_all_pieces_hint import DevelopAllPiecesHint
from unnecessary_pawn_move_hint import UnnecessaryPawnMoveHint
from unnecessary_check_hint import UnnecessaryCheckHint

DEFAULT_HINTS = [SamePieceTwiceHint(), DevelopKnightsFirstHint(), DevelopAllPiecesHint(),
 UnnecessaryPawnMoveHint(), UnnecessaryCheckHint()]

class Hinter:

    def __init__(self):
        self.hints = list(DEFAULT_HINTS)

    def getHints(self, board, move, best_move):
        hints = [hint.getHint(board, move, best_move) for hint in self.hints]
        return filter(None, hints)

    def getHint(self, board, move, best_move):
        hints = self.getHints(board, move, best_move)
        if len(hints) > 0:
            return hints[0]
        else:
            return None