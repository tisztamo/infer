from same_piece_twice_hint import SamePieceTwiceHint
from develop_knights_first_hint import DevelopKnightsFirstHint

DEFAULT_HINTS = [SamePieceTwiceHint(), DevelopKnightsFirstHint()]

class Hinter:

    def __init__(self):
        self.hints = list(DEFAULT_HINTS)

    def getHints(self, board, move):
        hints = [hint.getHint(board, move) for hint in self.hints]
        return filter(None, hints)

    def getHint(self, board, move):
        hints = self.getHints(board, move)
        if len(hints) > 0:
            return hints[0]
        else:
            return None