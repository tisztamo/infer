import chess, chess.uci
import numpy as np
import log
import input
import inference

logger = log.getLogger("engine")

BACK_ENGINE_EXE = "../stockfish-8-linux/Linux/stockfish_8_x64_modern"
label_strings = input.load_labels()

class Engine:
    def __init__(self, back_engine_exe=BACK_ENGINE_EXE):
        self._initBackEngine(back_engine_exe)

    def _initBackEngine(self, back_engine_exe):
        self.back_engine = chess.uci.popen_engine(back_engine_exe)
        self.back_engine.uci()
        logger.info("Opened back-engine " + self.back_engine.name)


    def evaluate(self, board):
        """Calculates the value in centipawns of the board
         from the viewpoint of the player who will move next"""
        return 0

    def bestMove(self, board):
        """Returns the best move in UCI notation and the value of the board after that move"""
        predictions = inference.predict(board.fen())
        return label_strings[np.argmax(predictions)]


def main():
    e = Engine()
    b = chess.Board()
    logger.info("Best move:" + e.bestMove(b))

if __name__ == "__main__":
    main()
