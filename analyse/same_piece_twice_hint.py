from hint import Hint

class SamePieceTwiceHint():
    def __init__(self):
        pass

    #TODO check for forced moves!
    def getHint(self, board, move, best_move):
        if len(board.move_stack) < 2 or not board.in_opening():
            return None
        my_last_move = board.move_stack[-2]
        if my_last_move.to_square == move.from_square:
            return Hint("#3 Don't move the same piece twice during opening")
        return None