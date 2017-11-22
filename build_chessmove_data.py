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
import log

import input, engine

tf.app.flags.DEFINE_string('train_dir', '../data/train/',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_dir', '../data/validation/',
                           'Validation data directory')
tf.app.flags.DEFINE_string('output_dir', '../data/',
                           'Output data directory')
tf.app.flags.DEFINE_string('eval_depth', '0',
                           'Depth to eval position using the external engine')
tf.app.flags.DEFINE_string('engine_exe', '../stockfish-8-linux/Linux/stockfish_8_x64',
                           'UCI engine executable')
tf.app.flags.DEFINE_string('skip_games', '0',
                           'Skip the first N games')
tf.app.flags.DEFINE_string('filter_player', '',
                           'Process only moves of the given player, or omit the player if the option starts with "-"')
tf.app.flags.DEFINE_string('omit_draws', 'true',
                           'Omit games that ended in a draw')

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


class EvalEngine:
    def __init__(self):
        self.depth = 0 if FLAGS.disable_cp != "false" else int(FLAGS.eval_depth)
        if self.depth > 0:
            self.uci_engine = chess.uci.popen_engine(FLAGS.engine_exe)
            self.info_handler = InfoHandler()
            self.uci_engine.info_handlers.append(self.info_handler)
            self.uci_engine.uci()
            self.uci_engine.setoption({"MultiPV": "1"})
            self.last_command = None
        self.last_board = None


    def start_evaluate_board(self, board, game, move):
        self.start_ts = time.time()
        if self.depth > 0:
            self.uci_engine.isready()
            self.uci_engine.ucinewgame()
            self.uci_engine.position(board)
            self.last_command = self.uci_engine.go(depth=self.depth, async_callback=True)
        self.last_board = board.copy()
        self.last_game = game
        self.last_move = move

    def is_evaluation_available(self):
        if self.depth == 0:
            return True
        return self.last_command is not None and self.last_command.done()

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
        #print(board.fen(), self.last_move, best_move, score, "complexity:", self.info_handler.complexity, "time:", used_time, "ratio:", self.info_handler.complexity / used_time)
        return board, score, best_move, self.info_handler.complexity, self.last_game, self.last_move


def filter_game(game):
    try:
        white_is_comp = (game.headers["WhiteIsComp"].lower() == "yes")
    except:
        white_is_comp = False
    try:
        black_is_comp = (game.headers["BlackIsComp"].lower() == "yes")
    except:
        black_is_comp = False
    try:
        event = game.headers["Event"]
    except:
        return False
    if FLAGS.omit_draws == "true" and game.headers["Result"] not in ["1-0", "0-1"]:
        if game.headers["Result"] != "1/2-1/2":
            print("Dropping game with result", game.headers["Result"])
        return False
    return not white_is_comp and not black_is_comp and event.find("960") == -1

def filter_move(board, move, player):
    if FLAGS.filter_player == "":
        return True
    if FLAGS.filter_player.startswith("-"):
        return not FLAGS.filter_player[1:] == player
    else:
        return FLAGS.filter_player == player


def init_engines(num=4):
    retval = []
    for i in range(num):
        retval.append(EvalEngine())
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


def _convert_to_example(board, move, game, cp_score, complexity, best_move, player):
    game_headers = game.headers
    sixlayer_rep = input.encode_board(board)
    move_uci = move.uci()
    result = game_headers["Result"]
    result_code = 0
    if result == "1-0":
        result_code = 1
    elif result == "0-1":
        result_code = -1

    if board.turn == chess.BLACK:
        move_uci = input.sideswitch_label(move_uci)
        cp_score = -cp_score
        result_code = -result_code

    try:
        ply_count = int(game_headers["PlyCount"])
    except:
        ply_count = -1


    feature_desc = {
        'board/sixlayer': tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(sixlayer_rep))),
        'board/fen': _bytes_feature(tf.compat.as_bytes(board.fen())),
        #'board/complexity': _int64_feature(complexity),
        #'board/best_uci': _bytes_feature(tf.compat.as_bytes(best_move.uci())),
        #'game/basetime': _int64_feature(base_time),
        #'game/time_increment': _int64_feature(time_increment),
        #'game/total_ply_count': _int64_feature(ply_count),
        'game/result': _int64_feature(result_code),
        'move/player': _int64_feature(input.hash_32(player)),
        'move/uci': _bytes_feature(tf.compat.as_bytes(move_uci)),
        'move/label': _int64_feature(labels.index(move_uci))
    }
    if FLAGS.disable_cp != "false" or int(FLAGS.eval_depth) > 0:
        feature_desc["board/cp_score/" + FLAGS.eval_depth] = _int64_feature(int(cp_score))

    example = tf.train.Example(features=tf.train.Features(feature=feature_desc))
    return example


def _open_pgn_file(filename):
    f = codecs.open(filename, encoding="latin-1")#utf-8-sig
    skip = int(FLAGS.skip_games)
    while skip > 0:
        line = f.readline()
        if line.startswith("[Event "):
            skip -= 1
            if skip == 0:
                while line.strip() != "":
                    line = f.readline()
    return f

def _process_pgn_file(filename, writer, engines):
    num_filtered_out = 0
    num_processed = 0
    num_moves = 0
    engine_idx = 0
    engine = engines[engine_idx]

    with _open_pgn_file(filename) as f:
        while True:
            game = chess.pgn.read_game(f)
            if not game:
                return num_processed, num_filtered_out, num_moves
            if not filter_game(game):
                num_filtered_out += 1
                continue
            players = [game.headers["White"], game.headers["Black"]]
            board = game.board()
            for move in game.main_line():
                if not board.is_valid():
                    print("Invalid board:", board.fen())
                    print("Skipping whole game", game.headers)
                    break
                engine = engines[engine_idx]
                try:
                    result_found = False
                    tried_engines = 0
                    while not result_found and tried_engines < len(engines):
                        tried_engines += 1
                        if engine.is_evaluation_available():
                            result_found = True
                        else:
                            engine_idx = (engine_idx + 1) % len(engines)
                            engine = engines[engine_idx]

                    if result_found:
                        last_board, score, best_move, complexity, last_game, last_move = engine.get_evaluation_result()
                        player = players[0 if last_board.turn == chess.WHITE else 1]
                        example = _convert_to_example(last_board, last_move, last_game, score, complexity, best_move, player)
                        writer.write(example.SerializeToString())
                except Exception as x:
                    print("Error while processing:", x)

                player = players[0 if board.turn == chess.WHITE else 1]
                if filter_move(board, move, player):
                    num_moves += 1
                    engine.start_evaluate_board(board, game, move)
                    engine_idx = (engine_idx + 1) % len(engines)
                else:
                    print("Dropped move by " + player)
                board.push(move)

            num_processed += 1
            if num_processed % 10 == 1:
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
        time.sleep(3)
        

def main(unused_argv):
    global labels
    print('Saving results to %s' % FLAGS.output_dir)
    labels, _ = input.load_labels()

    engines = init_engines(3)

    _process_dataset('validation', FLAGS.validation_dir, engines)
    _process_dataset('train', FLAGS.train_dir, engines)

if __name__ == '__main__':
  tf.app.run()