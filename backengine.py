import time
import chess
import chess.pgn
import chess.uci

import flags
import input

FLAGS = flags.FLAGS

class PV:
    def __init__(self, moves, score):
        self.moves = moves
        self.score = score

class InfoHandler(chess.uci.InfoHandler):
    def __init__(self):
        super(InfoHandler, self).__init__()
        self.clear()

    def clear(self):
        self.current_pvs = [PV(None, -input.MATE_CP_SCORE) for i in range(input.MULTIPV)]
        self.current_depth = 0
        self.current_pv_idx = 0
    
    def depth(self, val):
        self.current_depth = val

    def score(self, cp, mate, lowerbound, upperbound):
        if cp is None and mate == 0:
            return -input.MATE_CP_SCORE
        self.current_pvs[self.current_pv_idx].score = input.cp_score(chess.uci.Score(cp, mate))
        #print("cp: " + str(cp) + " mate: " + str(mate) + " score: " + str(self.current_pvs[self.current_pv_idx].score))
        super(InfoHandler, self).score(cp, mate, lowerbound, upperbound)

    def multipv(self, num):
        self.current_pv_idx = num - 1

    def pv(self, moves):
        self.current_pvs[self.current_pv_idx].moves = moves
        super(InfoHandler, self).pv(moves)


class BackEngine:
    def __init__(self):
        self.depth = 0 if FLAGS.disable_cp != "false" else int(FLAGS.eval_depth)
        if self.depth > 0:
            self.uci_engine = chess.uci.popen_engine(FLAGS.engine_exe)
            self.info_handler = InfoHandler()
            self.uci_engine.info_handlers.append(self.info_handler)
            self.uci_engine.uci()
            self.uci_engine.setoption({"MultiPV": str(input.MULTIPV)})
            self.name = self.uci_engine.name
            self.last_command = None
        self.last_board = None


    def start_evaluate_board(self, board, game = None, move = None):
        self.start_ts = time.time()
        if self.depth > 0:
            self.uci_engine.isready()
            if self.last_command is not None:
                self.last_command.result()
            #self.uci_engine.ucinewgame()
            #self.uci_engine.isready()
            self.uci_engine.position(board)
            self.info_handler.clear()
            self.last_command = self.uci_engine.go(depth=9, async_callback=True)#movetime=200
        self.last_board = board.copy()
        self.last_game = game
        self.last_move = move

    def is_evaluation_available(self):
        if self.depth == 0:
            return True
        return self.last_command is not None and self.last_command.done()

    def wait_for_evaluation(self):
        if self.depth == 0 or self.last_command is None:
            return False
        self.last_command.result()
        return True
        
    def get_evaluation_result(self):
        """ Returns (board, score in cp, best move, ponder move) from the external engine"""
        if self.depth == 0 or not self.last_command:
            return self.last_board, 0, None, 0, self.last_game, self.last_move
        best_move, ponder_move = self.last_command.result()
        used_time = time.time() - self.start_ts
        board = self.last_board
        self.last_command = None
        self.last_board = None
        score = engine.Engine.cp_score(self.info_handler.info["score"][1])
        return board, score, best_move, self.last_game, self.last_move


    def get_multipv_result(self):
        """ Returns (board, move, game, pvs) where pvs is an array of PV """
        if self.depth == 0 or not self.last_command:
            return self.last_board, self.last_move, self.last_game, []
        best_move, ponder_move = self.last_command.result()
        used_time = time.time() - self.start_ts
        board = self.last_board
        self.last_command = None
        self.last_board = None
        print("Depth: " + str(self.info_handler.current_depth))
        for i in range(input.MULTIPV):
            if self.info_handler.current_pvs[i].moves is not None:
                print(self.info_handler.current_pvs[i].moves[0].uci() + ": " + str(self.info_handler.current_pvs[i].score))
        return board, self.last_move, self.last_game, self.info_handler.current_pvs
