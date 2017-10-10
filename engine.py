import random
import chess, chess.uci
import numpy as np
import log
import input
import inference

logger = log.getLogger("engine")

BACK_ENGINE_EXE = "../stockfish-8-linux/Linux/stockfish_8_x64_modern"
BACK_ENGINE_DEPTH = 5
EVAL_RANDOMNESS = 10
BEAM_SIZES = [0, 5, 8, 10]

STALEMATE = -100000 #Smaller than the smallest possible score

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
        move, ponder = self.back_engine.go(depth=BACK_ENGINE_DEPTH)
        score = self.info_handler.info["score"][1]
        if score.cp is not None:
            score = chess.uci.Score(score.cp + random.randint(-EVAL_RANDOMNESS, EVAL_RANDOMNESS), None)        
        if move is not None:
            move = move.uci()
        if ponder is not None:
            ponder = ponder.uci()
        return move, score, ponder

    def evaluate(self, board):
        """Calculates the value in centipawns of the board
         relative to the side to move"""
        move, score, ponder = self.evaluateStatic(board)
        if score.cp is None:
            mate_distance = abs(score.mate) if abs(score.mate) <= BACK_ENGINE_DEPTH else BACK_ENGINE_DEPTH
            mate_val = 10000 + (BACK_ENGINE_DEPTH - mate_distance) * 1000
            ret_score = -mate_val if score.mate < 0 else mate_val 
            return move, ret_score, ponder
        return move, score.cp, ponder

    def search(self, board, depth=3, try_move=None):
        if depth == 0:
            move, score, ponder = self.evaluate(board)
            return move, score, ponder, "Pass"
        beam_size = BEAM_SIZES[depth]
        predictions = inference.predict(board.fen())
        candidates = np.argpartition(predictions, -20)[-20:]
        candidates = candidates[np.argsort(-predictions[candidates])]
        if try_move is not None:
            candidates = np.insert(candidates, 0, [label_strings.index(try_move)])
        moves = (label_strings[idx] for idx in candidates)
        move_counter = 0
        best_score = STALEMATE
        best_move = None
        ponder = None
        best_ponder = None
        best_ponder_ponder = None
        for move in moves:
            try:
                board.push_uci(move)
            except:
                logger.debug("Illegal move: " + str(move))
                continue
            ponder, score, ponder_ponder, _ = self.search(board, depth - 1)
            score = 0 if score == STALEMATE else -score - 1
            #logger.info(str(depth) + str(move) + ": " +str(score))
            if score > best_score + 10:
                best_move = move
                best_score = score
                best_ponder = ponder
                best_ponder_ponder = ponder_ponder
            board.pop()
            move_counter += 1
            if move_counter >= beam_size:
                break
        return best_move, best_score, best_ponder, best_ponder_ponder


    def bestMove(self, board, try_move):
        """Returns the best move in UCI notation, the value of the board after that move, the ponder move and my anticipated nex move (ponderponder)"""
        board = board.copy()
        static_move, current_score, static_ponder = self.evaluate(board)
        move, new_score, ponder, ponder_ponder = self.search(board, try_move=try_move)
        logger.info(str(move) + ": from " + str(current_score) + " to " + str(new_score) + " ponder " + str(ponder) + " ponderponder " + str(ponder_ponder))
        return move, new_score, ponder, ponder_ponder


def main():
    e = Engine()
    b = chess.Board()
    best_move, score, ponder, ponder_ponder = e.bestMove(b)
    logger.info("Best move:" + best_move + "(" + str(score) + ") ponder " + ponder)

if __name__ == "__main__":
    main()
