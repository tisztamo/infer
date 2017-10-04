from __future__ import division
from __future__ import print_function

import os
import time
from os import listdir
from os.path import isfile, join
import codecs

import numpy as np
import tensorflow as tf
import chess
import chess.pgn
import chess.uci

import input

tf.app.flags.DEFINE_string('train_dir', '../data/train/',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_dir', '../data/validation/',
                           'Validation data directory')
tf.app.flags.DEFINE_string('output_dir', '../data/',
                           'Output data directory')
tf.app.flags.DEFINE_string('eval_depth', '20',
                           'Depth to eval position using the external engine')
tf.app.flags.DEFINE_string('engine_exe', '../stockfish-8-linux/Linux/stockfish_8_x64',
                           'UCI engine executable')

FLAGS = tf.app.flags.FLAGS
labels = []

class InfoHandler(chess.uci.InfoHandler):
    def __init__(self):
        super(InfoHandler, self).__init__()
        self.current_best_move = None
        self.current_depth = 0
        self.scores = [0] * 40
    
    def depth(self, val):
        self.current_depth = val

    def score(self, cp, mate, lowerbound, upperbound):
        self.current_score = chess.uci.Score(cp, mate)
        self.scores[self.current_depth] = self.current_score
        super(InfoHandler, self).score(cp, mate, lowerbound, upperbound)

    def pv(self, moves):
        if self.current_best_move != moves[0]:
            self.current_best_move = moves[0]
            self.complexity = self.current_depth
            #print("Found new best", self.current_best_move, "at", self.current_depth, self.current_score.cp)
        super(InfoHandler, self).pv(moves)


class Engine:
    def __init__(self):
        self.uci_engine = chess.uci.popen_engine(FLAGS.engine_exe)
        self.info_handler = InfoHandler()
        self.uci_engine.info_handlers.append(self.info_handler)
        self.uci_engine.uci()
        self.uci_engine.setoption({"MultiPV": "1"})
        self.last_command = None
        self.last_board = None

    def start_evaluate_board(self, board):
        depth = int(FLAGS.eval_depth)
        self.uci_engine.ucinewgame()
        self.uci_engine.position(board)
        self.start_ts = time.time()
        self.last_command = self.uci_engine.go(depth=depth, async_callback=True)
        self.last_board = board.copy()

    def is_evaluation_available(self):
        return self.last_command.done()

    def get_evaluation_result(self):
        """ Returns (board, score in cp, best move, ponder move) from the external engine"""
        if not self.last_command:
            return None, None, None
        best_move, ponder_move = self.last_command.result()
        used_time = time.time() - self.start_ts
        board = self.last_board
        self.last_command = None
        self.last_board = None
        score = input.cp_score(self.info_handler.info["score"][1])
        #print(board.fen(), best_move, score, "complexity:", self.info_handler.complexity, "time:", used_time, "ratio:", self.info_handler.complexity / used_time)
        return board, score, best_move, self.info_handler.complexity


def filter_game(game):
    return True
    try:
        white_is_comp = (game.headers["WhiteIsComp"].lower() == "yes")
    except:
        white_is_comp = False
    try:
        black_is_comp = (game.headers["BlackIsComp"].lower() == "yes")
    except:
        black_is_comp = False
    return not white_is_comp and not black_is_comp

def load_labels():
    with open(FLAGS.labels_file) as f:
        return f.readline().strip().split(" ")


def init_engines(num=4):
    retval = []
    for i in range(num):
        retval.append(Engine())
    return retval


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float32_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes (string) features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(board, move, game, cp_score, complexity, best_move):
    game_headers = game.headers
    sixlayer_rep = input.encode_board(board)
    move_uci = move.uci()
    #time_control = game_headers["TimeControl"].split("+")
   # base_time = int(time_control[0])
    #try:
#        time_increment = int(time_control[1])
 #   except:
  #      time_increment = 0
    
    # white_is_comp = False
    # black_is_comp = False
    # try:
    #     white_is_comp = (game.headers["WhiteIsComp".lower()] == "yes")
    #     black_is_comp = (game.headers["BlackIsComp".lower()] == "yes")
    # except:
    #     pass
    
    # white_elo = -1
    # black_elo = -1
    # try:
    #     white_elo = int(game_headers["WhiteElo"])
    #     black_elo = int(game_headers["BlackElo"])
    # except:
    #     pass

    try:
        ply_count = int(game_headers["PlyCount"])
    except:
        ply_count = -1

    result = game_headers["Result"]
    result_code = 0
    if result == "1-0":
        result_code = 1
    elif result == "0-1":
        result = -1
    example = tf.train.Example(features=tf.train.Features(feature={
        'board/sixlayer': tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(sixlayer_rep))),
        'board/cp_score': _int64_feature(int(cp_score)),
        'board/complexity': _int64_feature(complexity),
        'board/best_uci': _bytes_feature(tf.compat.as_bytes(best_move.uci())),
        #'game/basetime': _int64_feature(base_time),
        #'game/time_increment': _int64_feature(time_increment),
        'game/total_ply_count': _int64_feature(ply_count),
        'game/result': _int64_feature(result_code),
        'move/halfmove_clock_before': _int64_feature(board.halfmove_clock),
        'move/uci': _bytes_feature(tf.compat.as_bytes(move_uci)),
        'move/label': _int64_feature(labels.index(move_uci))
        }))
    return example


def _process_pgn_file(filename, writer, engines):
    num_filtered_out = 0
    num_processed = 0
    num_moves = 0
    engine_idx = 0
    engine = engines[engine_idx]

    with codecs.open(filename, encoding="latin-1") as f:#utf-8-sig
        while True:
            game = chess.pgn.read_game(f)
            if not game:
                return num_processed, num_filtered_out, num_moves
            if not filter_game(game):
                num_filtered_out += 1
                continue
            board = game.board()
            for move in game.main_line():
                engine = engines[engine_idx]
                if num_moves >= len(engines):
                    try:
                        result_found = False
                        while not result_found:
                            if engine.is_evaluation_available:
                                result_found = True
                            else:
                                engine_idx = (engine_idx + 1) % len(engines)
                                engine = engines[engine_idx]

                        last_board, score, best_move, complexity = engine.get_evaluation_result()
                        example = _convert_to_example(last_board, move, game, score, complexity, best_move)
                        writer.write(example.SerializeToString())
                    except Exception as x:
                        print("Error while processing:", x)

                num_moves += 1
                board.push(move)
                engine.start_evaluate_board(board)
                engine_idx = (engine_idx + 1) % len(engines)

            num_processed += 1
            if num_processed % 100 == 1:
                print("Processed", num_processed, "games, dropped", num_filtered_out, ",", num_moves, "moves total.")


def _process_dataset(dataset_name, directory, engines):
    global labels
    pgn_files = [f for f in listdir(directory) if isfile(join(directory, f)) and f.endswith("pgn")]
    print("Found", len(pgn_files), "pgn files in", directory)

    num_shards = len(pgn_files)
    shard = 0

    for filename in pgn_files:
        shard += 1
        output_filename = '%s-%.5d-of-%.5d' % (dataset_name, shard, num_shards)
        print("Processing", filename, "into", output_filename)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)
        processed, dropped, moves = _process_pgn_file(join(directory, filename), writer, engines)
        print("Dropped", dropped, "games, processed", processed, ", containing", moves, "halfmoves.")
        writer.close()
        

def main(unused_argv):
    global labels
    print('Saving results to %s' % FLAGS.output_dir)
    labels = load_labels()

    engines = init_engines(7)

    #_process_dataset('validation', FLAGS.validation_dir)
    _process_dataset('train', FLAGS.train_dir, engines)

if __name__ == '__main__':
  tf.app.run()