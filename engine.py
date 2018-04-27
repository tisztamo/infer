import math, time
import random
import chess, chess.uci
import tensorflow as tf
import numpy as np
import log
import input, inference, strength
import flags

FLAGS = flags.FLAGS

logger = log.getLogger("engine")

ENGINE_NAME="Infer Development"
MATE_VAL =  20000 #-1000 for every move down to 10000 where it stops. If mate is further than 10 plies, score is 10000

EVAL_RANDOMNESS = 0
STALEMATE_SCORE = -20 #For White

BACK_ENGINE_THREADS = 1
TIME_PER_INFER = 150
policy_bias_cp = 0

label_strings, _ = input.load_labels()

class CandidateMove:
    INVALID_APPEAL = -1000

    def __init__(self, engine, uci, probability, cp_score = None, ponder = None):
        self.engine = engine
        self.uci = uci
        self.probability = probability
        self.appeal = self.INVALID_APPEAL
        self.ponder = ponder
        self.ponder_ponder = None
        self.lost_score = None
        self.set_cp_score(cp_score)

    def set_cp_score(self, cp_score):
        self.cp_score = cp_score
    
    def calculate_appeal(self, best_score):
        if self.probability is None or self.cp_score is None:
            self.appeal = self.INVALID_APPEAL
            return
        prob = max(self.probability, 0.0001)
        self.appeal = prob * pow(2.0, float(self.cp_score - best_score) / self.engine.strengthManager.double_prob_accounts_for_cp())
        self.lost_score = best_score - self.cp_score
        return self.appeal

class Engine:
    def __init__(self):
        self.back_engine = None
        self.current_board = chess.Board()
        self.newGame()
        self.name = ENGINE_NAME
        self.strengthManager = strength.StrengthManager(self)
        self.my_time = 0
        self.opponent_time = 0
        self.movestogo = 0

    @staticmethod
    def cp_score(uci_score_or_res_pred):
        if FLAGS.use_back_engine == "false":
            return uci_score_or_res_pred
        else:
            if uci_score_or_res_pred.cp is None:
                mate_distance = min(abs(uci_score_or_res_pred.mate), 10)
                mate_val = 10000 + (10 - mate_distance) * 1000
                ret_score = -mate_val if uci_score_or_res_pred.mate < 0 else mate_val 
                return ret_score
            return uci_score_or_res_pred.cp

    def initBackEngine(self):
        if FLAGS.use_back_engine == "false":
            return
        self.back_engine = chess.uci.popen_engine(FLAGS.back_engine_exe)
        self.info_handler = chess.uci.InfoHandler()
        self.back_engine.info_handlers.append(self.info_handler)
        self.back_engine.uci()
        self.back_engine.setoption({"Threads": BACK_ENGINE_THREADS})
        logger.info("Opened back-engine " + self.back_engine.name)

    def get_times(self, board, my_side, time_proportion):
        if my_side is None:
            my_side = board.turn
        if my_side == chess.WHITE:
            wtime = self.my_time * time_proportion
            btime = self.opponent_time * time_proportion
        else:
            wtime = self.opponent_time * time_proportion
            btime = self.my_time * time_proportion
        return wtime, btime

    def evaluateStatic(self, board, time_proportion, my_side, back_engine_depth):
        """Returns a move (uci), score (chess.uci.Score), ponder (uci) triplet with the static evaluation of the move"""
        if FLAGS.use_back_engine == "false":
            predicted_result = inference.predict_result(board)
            score = np.sum(np.multiply(predicted_result, [-1000, -20, 1000]))
            return None, score, None
        else:
            self.back_engine.position(board.copy())
            wtime, btime = self.get_times(board, my_side, time_proportion)
            logger.info("back engine eval on " + board.fen() + ", wtime=" + str(wtime) + ", btime=" + str(btime))
            move, ponder = self.back_engine.go(depth=back_engine_depth, movestogo=self.movestogo, wtime=wtime, btime=btime)
            score = self.info_handler.info["score"][1]
            if score.cp is not None:
                score = chess.uci.Score(score.cp + random.randint(-EVAL_RANDOMNESS, EVAL_RANDOMNESS), None)
            if move is not None:
                move = move.uci()
            if ponder is not None:
                ponder = ponder.uci()
            if move is None:
                logger.error("Got None from back engine!")
            return move, score, ponder

    def evaluate(self, board, time_proportion = 0.55, my_side=None, back_engine_depth=None):
        """Calculates the value in centipawns of the board
         relative to the side to move"""
        self.eval_count += 1
        if board.is_game_over(False):
            if board.is_checkmate():
                if board.turn == chess.WHITE and board.result == "1-0" or board.turn == chess.BLACK and board.result == "0-1":
                    score = MATE_VAL
                else:
                    score = -MATE_VAL
            else:
                score = STALEMATE_SCORE if board.turn == chess.WHITE else -STALEMATE_SCORE
            return None, score, None
        move, score, ponder = self.evaluateStatic(board, time_proportion, my_side, back_engine_depth)
        return move, self.cp_score(score), ponder

    def candidate_idxs(self, board):
        predictions = inference.predict_move(board.fen())
        candidates = np.argpartition(predictions, -20)[-20:]
        candidates = candidates[np.argsort(-predictions[candidates])]
        return candidates, predictions[candidates]

    def candidates(self, board):
        moves, probs = self.candidate_idxs(board)
        retval = [CandidateMove(self, label_strings[move], probs[idx]) for idx, move in enumerate(moves)]
        return retval

    def policyCandidate(self, board):
        preds = inference.predict_move(board.fen())
        argmax = np.argmax(preds, 0)
        return CandidateMove(self, inference.label_strings[argmax], preds[argmax])

    def bestMove(self, board):
        """Returns the best move in UCI notation, the value of the board after that move, the ponder move and my anticipated nex move (ponderponder)"""
        board = board.copy()
        self.eval_count = 0
        ts=time.time()
        back_engine_uci, pre_score, static_ponder = self.evaluate(board)
        policy_move = self.policyCandidate(board)
        selected_move = CandidateMove(self, back_engine_uci, 0)
        if policy_move.uci != back_engine_uci:
            board.push_uci(policy_move.uci)
            _, post_score, _2 = self.evaluate(board, my_side = board.turn)
            post_score = -post_score
            logger.info("Back engine: " + str(back_engine_uci) + "(" + str(pre_score) + "), policy: " + policy_move.uci + "(" + str(post_score) + ")")
            if (pre_score < post_score + random.randint(int(0.7 * policy_bias_cp), int(1.3 * policy_bias_cp))):
                selected_move = policy_move
                logger.info("Preferring policy move.")

        logger.info("Selected move: " + str(selected_move.uci))
        return selected_move

    def newGame(self):
        self.current_board = chess.Board()

    def parse_uciparams(self, params):
        my_color = self.current_board.turn

        if "wtime" in params:
            wtime = int(params["wtime"])
            btime = int(params["btime"])
            if my_color == chess.WHITE:
                self.my_time, self.opponent_time = wtime, btime
            else:
                self.my_time, self.opponent_time = btime, wtime

        if "movestogo" in params:
            self.movestogo = int(params["movestogo"])
            self.my_time = self.my_time - self.movestogo * TIME_PER_INFER
            if self.my_time < 1:
                logger.error("Not enough time, adjust TIME_PER_INFER!")
                self.my_time = 1
        else:
            self.movestogo = None


    def move(self, board, params):
        if board.fullmove_number > 1 and len(board.move_stack) > 0:
            last_move = board.peek()
            self.current_board.push(last_move)
        if board.board_fen() != self.current_board.board_fen():
            if self.current_board.fullmove_number > 1:
                logger.error("Incosistent board, dropping the old one " + self.current_board.fen() + " for " + board.fen())
            self.current_board = board.copy()

        if params is not None:
            self.parse_uciparams(params)

        self.color = self.current_board.turn

        if FLAGS.play_first_intuition != "false":
            candidates = self.candidates(board)
            self.print_candidate_moves(candidates)
            move = candidates[0]
        else:
            move = self.bestMove(board)
        if move.uci is not None:
            try:
                self.current_board.push_uci(move.uci)
            except:
                logger.error("Illegal move generated: " + str(move))
                move.uci = list(self.current_board.legal_moves)[0].uci()
        return move

    def print_candidate_moves(self, candidate_moves):
        try:
            moves = sorted(candidate_moves, key=lambda c: -c["prob"])
        except:
            moves = candidate_moves
        for move in moves:
            try:
                print("Candidate", move.uci, move.probability, move.cp_score, move.lost_score, move.appeal)
            except:
                print(move)


def main(unused_argv):
    e = Engine()
    e.initBackEngine()
    b = chess.Board()
    move = e.move(b)
    logger.info("Best move:" + str(move.uci) + "(" + str(move.cp_score) + ") ponder " + str(move.ponder))

if __name__ == "__main__":
    tf.app.run()
