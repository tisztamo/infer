from hint import Hint

class UnnecessaryCheckHint():
    def __init__(self):
        pass

    def getHint(self, board, move, best_move):
        if not board.in_opening():
            return None
        b = board.copy()
        b.push(move)
        if not b.is_check():
            return None
        b = board.copy()
        b.push(best_move)
        if not b.is_check():
            return Hint(5, "Don't check if not necessary")
        return None