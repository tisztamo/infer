import math, time
import random
import chess, chess.uci
import tensorflow as tf
import numpy as np
import log
import input, inference, strength

tf.app.flags.DEFINE_string('play_first_intuition', 'false',
                           'Play the raw output of the policy network')
tf.app.flags.DEFINE_string('use_back_engine', 'true',
                           'Whether to use external engine for static (leaf) evaluation')
tf.app.flags.DEFINE_string('back_engine_exe', '../stockfish-8-linux/Linux/stockfish_8_x64_modern',
                           'External engine executable')
tf.app.flags.DEFINE_string('back_engine_depth', '6',
                           'External engine search depth')
tf.app.flags.DEFINE_string('search_depth', '3',
                           'Search depth')

FLAGS = tf.app.flags.FLAGS

logger = log.getLogger("engine")

ENGINE_NAME="Turk Development"
MATE_VAL =  20000 #-1000 for every move down to 10000 where it stops. If mate is further than 10 plies, score is 10000

BEAM_SIZES = [0, 12, 12, 12, 12, 12, 12]
MAX_BLUNDER = 250
EVAL_RANDOMNESS = 10
STALEMATE_SCORE = -20 #For White

BACK_ENGINE_THREADS = 4

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
        self.try_move = None
        self.current_board = chess.Board()
        self.newGame()
        self.name = ENGINE_NAME
        self.strengthManager = strength.StrengthManager(self)

    @staticmethod
    def cp_score(uci_score_or_res_pred):
        if FLAGS.use_back_engine == "false":
            return uci_score_or_res_pred * 1000
        else:
            if uci_score_or_res_pred.cp is None:
                mate_distance = min(abs(uci_score_or_res_pred.mate), 10)
                mate_val = 10000 + (10 - mate_distance) * 1000
                ret_score = -mate_val if uci_score_or_res_pred.mate < 0 else mate_val 
                return ret_score
            return uci_score_or_res_pred.cp

    def initBackEngine(self):
        self.back_engine = chess.uci.popen_engine(FLAGS.back_engine_exe)
        self.info_handler = chess.uci.InfoHandler()
        self.back_engine.info_handlers.append(self.info_handler)
        self.back_engine.uci()
        self.back_engine.setoption({"Threads": BACK_ENGINE_THREADS})
        logger.info("Opened back-engine " + self.back_engine.name)

    def evaluateStatic(self, board, back_engine_depth=None):
        """Returns a move (uci), score (chess.uci.Score), ponder (uci) triplet with the static evaluation of the move"""
        if FLAGS.use_back_engine == "false":
            return None, inference.predict_result(board), None
        else:
            if back_engine_depth is None:
                back_engine_depth = FLAGS.back_engine_depth
            self.back_engine.position(board.copy())
            move, ponder = self.back_engine.go(depth=back_engine_depth)
            score = self.info_handler.info["score"][1]
            if score.cp is not None:
                score = chess.uci.Score(score.cp + random.randint(-EVAL_RANDOMNESS, EVAL_RANDOMNESS), None)
            if move is not None:
                move = move.uci()
            if ponder is not None:
                ponder = ponder.uci()
            return move, score, ponder

    def evaluate(self, board, back_engine_depth=None):
        """Calculates the value in centipawns of the board
         relative to the side to move"""
        self.eval_count += 1
        if board.is_game_over(True):
            if board.is_checkmate():
                if board.turn == chess.WHITE and board.result == "1-0" or board.turn == chess.BLACK and board.result == "0-1":
                    score = MATE_VAL
                else:
                    score = -MATE_VAL
            else:
                score = STALEMATE_SCORE if board.turn == chess.WHITE else -STALEMATE_SCORE
            return None, score, None
        move, score, ponder = self.evaluateStatic(board, back_engine_depth)
        return move, self.cp_score(score), ponder

    def candidate_idxs(self, board, try_move=None):
        predictions = inference.predict_move(board.fen())
        candidates = np.argpartition(predictions, -20)[-20:]
        candidates = candidates[np.argsort(-predictions[candidates])]
        if try_move is not None:
            trymove_idx = label_strings.index(try_move)
            found = np.where(candidates==trymove_idx)
            candidates = np.delete(candidates, found)
            candidates = np.insert(candidates, 0, [trymove_idx])
        return candidates, predictions[candidates]

    def candidates(self, board, try_move=None):
        moves, probs = self.candidate_idxs(board, try_move)
        retval = [CandidateMove(self, label_strings[move], probs[idx]) for idx, move in enumerate(moves)]
        return retval

    def alpha_beta_search(self, board, depth, alpha, beta, max_min=1):
        if board.is_game_over(claim_draw=True):
            if board.is_checkmate():
                if board.turn == chess.WHITE and board.result == "1-0" or board.turn == chess.BLACK and board.result == "0-1":
                    score = MATE_VAL * max_min
                else:
                    score = -MATE_VAL * max_min
            else:
                score = STALEMATE_SCORE * max_min
            return CandidateMove(self, None, None, score)
        if depth == 0:
            move, score, ponder = self.evaluate(board)
            return CandidateMove(self, move, None, score * max_min, ponder)
        beam_size = BEAM_SIZES[depth]
        candidates = self.candidates(board)
        move_counter = 0
        selected_candidate = None
        for candidate in candidates:
            if move_counter == 0:
                selected_candidate = candidate
            move_counter += 1
            if move_counter >= beam_size or candidate.probability < 0.5:
                break
            move = candidate.uci
            try:
                board.push_uci(move)
            except:
                #logger.debug("Illegal move: " + str(move))
                continue
        
            ponder_candidate = self.alpha_beta_search(board, depth - 1, alpha, beta, -max_min)
            #print(ponder_candidate)
            board.pop()

            score = ponder_candidate.cp_score                
            if score is None:
                print("YYYYY")
                candidate.score = max_min * MATE_VAL
                return candidate
            candidate.set_cp_score(score)
            candidate.ponder = ponder_candidate.uci
            candidate.ponder_ponder = ponder_candidate.ponder
            
            if max_min == 1:
                if score >= beta:
                    candidate.set_cp_score(beta)
                    #print(depth, "Beta cutoff after ", move_counter, candidate.uci)
                    return candidate #fail hard beta-cutoff
                if score > alpha:
                    #print(candidate.uci, score, alpha)
                    alpha = score #alpha acts like max in MiniMax
                    selected_candidate = candidate
            else:
                if score <= alpha:
                    candidate.set_cp_score(alpha)
                    #print(depth, "Alpha cutoff after ", move_counter, candidate.uci)
                    return candidate
                if score < beta:
                    beta = score
                    selected_candidate = candidate


        if selected_candidate.cp_score is None:
            print("Selected candidate's cp score is None")
            print(board)
            print(depth, selected_candidate)
            for move in board.legal_moves:
                if move.uci() == selected_candidate.uci:
                    return selected_candidate
            return CandidateMove(self, move.uci(), None, None)

        return selected_candidate
        

    def beam_search(self, board, depth=1, try_move=None):
        STALEMATE = -100000 #Smaller than the smallest possible score
        if board.is_checkmate():
            if board.turn == chess.WHITE and board.result == "1-0" or board.turn == chess.BLACK and board.result == "0-1":
                score = MATE_VAL
            else:
                score = -MATE_VAL
            return CandidateMove(self, None, None, score)

        if depth == 0:
            move, score, ponder = self.evaluate(board)
            return CandidateMove(self, move, None, score, ponder)
        beam_size = BEAM_SIZES[depth]
        candidates = self.candidates(board, try_move)
        move_counter = 0
        for candidate in candidates:
            move = candidate.uci
            try:
                board.push_uci(move)
            except:
                #logger.debug("Illegal move: " + str(move))
                continue

            ponder_candidate = self.beam_search(board, depth - 1)
            score = STALEMATE_SCORE if ponder_candidate.cp_score == STALEMATE else -ponder_candidate.cp_score - 1
            candidate.set_cp_score(score)
            candidate.ponder = ponder_candidate.uci
            candidate.ponder_ponder = ponder_candidate.ponder

            board.pop()
            move_counter += 1
            if move_counter >= beam_size:
                break

        try:
            searched_candidates = [c for c in candidates if c.cp_score is not None]
            max_score = max([c.cp_score for c in searched_candidates])
            accepted_candidates = [c for c in searched_candidates if c.cp_score > max_score - MAX_BLUNDER]
        except:
            max_score = -MATE_VAL
            accepted_candidates = []
        best = None
        best_appeal = -100000
        for candidate in accepted_candidates:
            candidate.calculate_appeal(max_score)
            if candidate.appeal > best_appeal:
                best = candidate
                best_appeal = candidate.appeal

        if best is None:
            logger.error("No candidate found for " + board.fen())
            self.print_candidate_moves(candidates)
            move, score, ponder = self.evaluate(board)
            return CandidateMove(self, move, 0.0, score, ponder)
        if depth == 1:
            #self.print_candidate_moves(candidates)
            print(best.appeal, best.probability, max_score - best.cp_score)
            pass
        return best


    def bestMove(self, board, try_move):
        """Returns the best move in UCI notation, the value of the board after that move, the ponder move and my anticipated nex move (ponderponder)"""
        board = board.copy()
        self.eval_count = 0
        ts=time.time()
        static_move, pre_score, static_ponder = self.evaluate(board)
        move = self.beam_search(board, depth=1, try_move=None)
        #move = self.alpha_beta_search(board, int(FLAGS.search_depth), -math.inf, math.inf)
        logger.info(str(move.uci) + ": from " + str(pre_score) + " to " + str(move.cp_score) + " ponder " + str(move.ponder) + " ponderponder " + str(move.ponder_ponder) + " prob: " + str(move.probability) + " appeal: " + str(move.appeal))
        print(self.eval_count, "nodes, nps:", float(self.eval_count) / (time.time() - ts))
        return move

    def newGame(self):
        self.current_board = chess.Board()

    def move(self, board):
        if board.fullmove_number > 1 and len(board.move_stack) > 0:
            last_move = board.peek()
            self.current_board.push(last_move)
        if board.board_fen() != self.current_board.board_fen():
            if self.current_board.fullmove_number > 1:
                logger.error("Incosistent board, dropping the old one " + self.current_board.fen() + " for " + board.fen())
            self.current_board = board.copy()

        self.color = self.current_board.turn

        if FLAGS.play_first_intuition != "false":
            candidates = self.candidates(board)
            self.print_candidate_moves(candidates)
            move = candidates[0]
        else:
            move = self.bestMove(board, self.try_move)
        self.try_move = move.ponder_ponder
        if move.uci is not None:
            try:
                self.current_board.push_uci(move.uci)
            except:
                logger.error("Illegal move generated", str(move))
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
    b = chess.Board()
    move = e.move(b)
    logger.info("Best move:" + str(move.uci) + "(" + str(move.cp_score) + ") ponder " + str(move.ponder))

if __name__ == "__main__":
    tf.app.run()
