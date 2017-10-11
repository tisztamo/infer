import random
import chess, chess.uci
import numpy as np
import log
import input, inference, pgnwriter

logger = log.getLogger("engine")

ENGINE_NAME="Turk Development"
BACK_ENGINE_EXE = "../stockfish-8-linux/Linux/stockfish_8_x64_modern"
BACK_ENGINE_DEPTH = 1
MATE_VAL=  10000 + BACK_ENGINE_DEPTH * 1000
EVAL_RANDOMNESS = 10
BEAM_SIZES = [0, 10, 12]

STALEMATE_SCORE = -20

label_strings = input.load_labels()

class Engine:
    def __init__(self, back_engine_exe=BACK_ENGINE_EXE):
        self.back_engine = None
        self.try_move = None
        self.current_board = chess.Board()
        self._initBackEngine(back_engine_exe)
        self.newGame()
        self.name = ENGINE_NAME

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

    def search(self, board, depth=2, try_move=None):
        STALEMATE = -100000 #Smaller than the smallest possible score
        if board.is_checkmate():
            print("Checkmate found at depth", depth)
            if board.turn == chess.WHITE and board.result == "1-0" or board.turn == chess.BLACK and board.result == "0-1":
                score = MATE_VAL
            else:
                score = -MATE_VAL
            return None, score, None, None

        if depth == 0:
            move, score, ponder = self.evaluate(board)
            return move, score, ponder, None
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
            score = STALEMATE_SCORE if score == STALEMATE else -score - 1
            if try_move is not None and move_counter == 0:
                logger.info("try: " + str(try_move) + ", d: " + str(depth) + " " + str(move) + ": " +str(score))
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

    def newGame(self):
        pgnwriter.write_history(self.current_board, self)
        self.current_board = chess.Board()

    def move(self, board):
        if board.fullmove_number > 1:
            last_move = board.peek()
            self.current_board.push(last_move)
        if board.board_fen() != self.current_board.board_fen():
            if self.current_board.fullmove_number > 1:
                logger.error("Incosistent board, dropping the old one " + self.current_board.fen() + " for " + board.fen())
            self.current_board = board.copy()

        self.color = self.current_board.turn

        move, score, ponder, self.try_move = self.bestMove(board, self.try_move)
        if move is not None:
            self.current_board.push_uci(move)
        return move, score, ponder


def main():
    e = Engine()
    b = chess.Board()
    ponder_ponder = None
    best_move, score, ponder, ponder_ponder = e.move(b, ponder_ponder)
    logger.info("Best move:" + best_move + "(" + str(score) + ") ponder " + ponder)

if __name__ == "__main__":
    main()
