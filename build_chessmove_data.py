from __future__ import division
from __future__ import print_function

import os, re
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

import input

tf.app.flags.DEFINE_string('train_dir', '../data/train/',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_dir', '../data/validation/',
                           'Validation data directory')
tf.app.flags.DEFINE_string('output_dir', '../data/',
                           'Output data directory')
tf.app.flags.DEFINE_string('skip_games', '0',
                           'Skip the first N games')
tf.app.flags.DEFINE_string('filter_player', '',
                           'Process only moves of the given player, or omit the player if the option starts with "-"')
tf.app.flags.DEFINE_string('filter_black', 'true',
                           'Drop moves of black')
tf.app.flags.DEFINE_string('skip_plies', '0',
                           'Drop the first N plies')
tf.app.flags.DEFINE_string('omit_draws', 'true',
                           'Omit games that ended in a draw')
tf.app.flags.DEFINE_string('omit_whitewins', 'false',
                           'Omit games with result 1-0')
tf.app.flags.DEFINE_string('omit_blackwins', 'false',
                           'Omit games with result 0-1')

FLAGS = tf.app.flags.FLAGS
labels = []

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
    result = game.headers["Result"]
    if FLAGS.omit_draws == "true" and result == "1/2-1/2" or \
       FLAGS.omit_whitewins == "true" and result == "1-0" or \
       FLAGS.omit_blackwins == "true" and result == "0-1":
        return False
    return not white_is_comp and not black_is_comp and event.find("960") == -1

def filter_move(board, move, player, ply_idx):
    if ply_idx < int(FLAGS.skip_plies):
        return False

    if board.turn == chess.BLACK and FLAGS.filter_black == "true":
        return False

    if FLAGS.filter_player == "":
        return True
    if FLAGS.filter_player.startswith("-"):
        return not FLAGS.filter_player[1:] == player
    else:
        return FLAGS.filter_player == player

def get_evaluation(comment):
    """ Variations handled:
     {-3.80/16 216s}
     {(Qa4+) +2.67/12 77s}
     {+0.00/1 0s}
     {+355.34/0 0s}
     {book}
     {book 0s}
     {0s}
    """
    comment = comment.strip()
    if comment in [None, "", "book", "book 0s", "0s"]:
        return None
    match = re.search("([+-]\d+\.\d\d)/(\d+)", comment)
    if match is None:
        #print("No score found in", comment)
        return None
    score = float(match.group(1))
    depth = match.group(2)
    if int(depth) <= 0:
        return None
    if abs(score) < 15.0:
        return score * 100.0
    return None 


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


def _convert_to_example(board, move, game, cp_score, player):
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

    feature_desc = {
        'board/bitboard': tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(sixlayer_rep))),
        'board/fen': _bytes_feature(tf.compat.as_bytes(board.fen())),
        'game/result': _int64_feature(result_code),
        'move/player': _int64_feature(input.hash_32(player)),
        'move/uci': _bytes_feature(tf.compat.as_bytes(move_uci)),
        'move/label': _int64_feature(labels.index(move_uci)),
        'board/cp_score': _int64_feature(int(cp_score))
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_desc))
    return example

def main_line(game):
    """Yields the moves of the main line starting in this node."""
    node = game
    idx = 0
    while node.variations:
        node = node.variations[0]
        yield node.move, node.comment, idx
        idx += 1 

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

def _process_pgn_file(filename, writer):
    num_filtered_out = 0
    num_processed = 0
    num_moves = 0
    num_dropped_moves = 0

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
            for move, comment, ply_idx in main_line(game):
                if not board.is_valid():
                    print("Invalid board:", board.fen())
                    print("Skipping whole game", game.headers)
                    break
                try:
                    player = players[0 if board.turn == chess.WHITE else 1]
                    if filter_move(board, move, player, ply_idx):
                        num_moves += 1
                        score = get_evaluation(comment)
                        if score is not None or FLAGS.disable_cp == "true":
                            example = _convert_to_example(board, move, game, score, player)
                            writer.write(example.SerializeToString())                        
                        else:
                            num_dropped_moves += 1
                    else:
                        num_dropped_moves += 1
                        #print("Dropped move by " + player)
                except Exception as x:
                    print("Error while processing:", x)

                board.push(move)

            num_processed += 1
            if num_processed % 10 == 1:
                print("Processed", num_processed, "games, dropped", num_filtered_out, ",", num_moves, "moves. Dropped ", num_dropped_moves, "moves.")


def _process_dataset(dataset_name, directory):
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
        processed, dropped, moves = _process_pgn_file(join(directory, filename), writer)
        print("Dropped", dropped, "games, processed", processed, ", containing", moves, "halfmoves.")
        writer.close()
        time.sleep(3)
        

def main(unused_argv):
    global labels
    print('Saving results to %s' % FLAGS.output_dir)
    labels, _ = input.load_labels()

    _process_dataset('validation', FLAGS.validation_dir)
    _process_dataset('train', FLAGS.train_dir)

if __name__ == '__main__':
  tf.app.run()