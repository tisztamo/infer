import chess, chess.uci
import numpy as np
import log
import input
import inference

logger = log.getLogger("engine")

BACK_ENGINE_EXE = "../stockfish-8-linux/Linux/stockfish_8_x64_modern"
BEAM_SIZE = 5
label_strings = input.load_labels()

class Engine:
    def __init__(self, back_engine_exe=BACK_ENGINE_EXE):
        self.back_engine = None
        self._initBackEngine(back_engine_exe)

    def _initBackEngine(self, back_engine_exe):
        self.back_engine = chess.uci.popen_engine(back_engine_exe)
        self.info_handler = chess.uci.InfoHandler()
        self.back_engine.info_handlers.append(self.info_handler)
        self.back_engine.uci()
        logger.info("Opened back-engine " + self.back_engine.name)


    def evaluateStatic(self, board):
        """Returns a chess.uci.Score with the static evaluation of the move"""
        self.back_engine.position(board.copy())
        self.back_engine.go(depth=2)
        return self.info_handler.info["score"][1]

    def evaluate(self, board):
        """Calculates the value in centipawns of the board
         relative to the side to move"""
        score = self.evaluateStatic(board)
        if score.cp is None:
            return -100000 if score.mate < 0 else 100000
        return score.cp

    def search(self, board, depth=5):
        if depth == 0:
            return "Pass", self.evaluate(board), "Pass"
        predictions = inference.predict(board.fen())
        candidates = np.argpartition(predictions, -20)[-20:]
        candidates = candidates[np.argsort(-predictions[candidates])]
        moves = (label_strings[idx] for idx in candidates)
        move_counter = 0
        best_score = -10000000
        best_move = None
        ponder = None
        for move in moves:
            try:
                board.push_uci(move)
            except:
                logger.warn("Illegal move: " + str(move))
                continue
            ponder, score, _ = self.search(board, depth - 1)
            score = -score
            #logger.info(str(move) + ": " +str(score))
            if score > best_score:
                best_move = move
                best_score = score
            board.pop()
            move_counter += 1
            if move_counter >= BEAM_SIZE:
                break
        return best_move, best_score, ponder


    def bestMove(self, board):
        """Returns the best move in UCI notation, the value of the board after that move and the ponder move"""
        board = board.copy()
        current_score = self.evaluate(board)
        move, new_score, ponder = self.search(board)
        logger.info(move + ": from " + str(current_score) + " to " + str(new_score))
        return move, new_score, ponder


def main():
    e = Engine()
    b = chess.Board()
    best_move, score, ponder = e.bestMove(b)
    logger.info("Best move:" + best_move + "(" + str(score) + ") ponder " + ponder)

if __name__ == "__main__":
    main()
