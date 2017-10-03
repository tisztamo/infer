from __future__ import division
from __future__ import print_function

import os
from os import listdir
from os.path import isfile, join
import codecs

import numpy as np
import tensorflow as tf
import chess
import chess.pgn

import input#TODO: different directory!

tf.app.flags.DEFINE_string('train_dir', '../data/train/',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_dir', '../data/validation/',
                           'Validation data directory')
tf.app.flags.DEFINE_string('output_dir', '../data/',
                           'Output data directory')

FLAGS = tf.app.flags.FLAGS

labels = []

def filter_game(game):
    """
    Decides if a pgn-parsed game should be used
    [Event "FICS rated standard game"]
    [Site "FICS freechess.org"]
    [FICSGamesDBGameNo "232718660"]
    [White "TogaII"]
    [Black "wundtsapawnintime"]
    [WhiteElo "2585"]
    [BlackElo "1720"]
    [WhiteIsComp "Yes"]
    [TimeControl "900+5"]
    [Date "2009.08.31"]
    [Time "22:42:00"]
    [WhiteClock "0:15:00.000"]
    [BlackClock "0:15:00.000"]
    [ECO "C62"]
    [PlyCount "22"]
    [Result "1-0"]
    """
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


def _convert_to_example(board, move, game):
    global labels
    game_headers = game.headers
    sixlayer_rep = input.encode_board(board)
    move_uci = move.uci()
    time_control = game_headers["TimeControl"].split("+")
    base_time = int(time_control[0])
    try:
        time_increment = int(time_control[1])
    except:
        time_increment = 0
    
    white_is_comp = False
    black_is_comp = False
    try:
        white_is_comp = (game.headers["WhiteIsComp".lower()] == "yes")
        black_is_comp = (game.headers["BlackIsComp".lower()] == "yes")
    except:
        pass
    
    white_elo = -1
    black_elo = -1
    try:
        white_elo = int(game_headers["WhiteElo"])
        black_elo = int(game_headers["BlackElo"])
    except:
        pass

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
        'board/fen': _bytes_feature(tf.compat.as_bytes(board.fen())),
        'board/sixlayer': tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(sixlayer_rep))),
        'game/event': _bytes_feature(tf.compat.as_bytes(game_headers["Event"])),
        'game/white': _bytes_feature(tf.compat.as_bytes(game_headers["White"])),
        'game/black': _bytes_feature(tf.compat.as_bytes(game_headers["Black"])),
        'game/white_elo': _int64_feature(white_elo),
        'game/black_elo': _int64_feature(black_elo),
        'game/white_is_comp': _int64_feature(white_is_comp),
        'game/black_is_comp': _int64_feature(black_is_comp),
        'game/basetime': _int64_feature(base_time),
        'game/time_increment': _int64_feature(time_increment),
        'game/total_ply_count': _int64_feature(ply_count),
        'game/result': _int64_feature(result_code),
        'move/halfmove_clock_before': _int64_feature(board.halfmove_clock),
        'move/fullmove_number_before': _int64_feature(board.fullmove_number),
        'move/uci': _bytes_feature(tf.compat.as_bytes(move_uci)),
        'move/label': _int64_feature(labels.index(move_uci))}))
    return example


def _process_pgn_file(filename, writer):
    num_filtered_out = 0
    num_processed = 0
    num_moves = 0
    with codecs.open(filename, encoding="utf-8-sig") as f:
        while True:
            game = chess.pgn.read_game(f)
            if not game:
                return num_processed, num_filtered_out, num_moves
            if not filter_game(game):
                num_filtered_out += 1
                continue
            board = game.board()
            for move in game.main_line():
                example = _convert_to_example(board, move, game)
                writer.write(example.SerializeToString())
                board.push(move)
                num_moves += 1
            num_processed += 1


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
        

def main(unused_argv):
    global labels
    print('Saving results to %s' % FLAGS.output_dir)

    labels = load_labels()

    #_process_dataset('validation', FLAGS.validation_dir)
    _process_dataset('train', FLAGS.train_dir)

if __name__ == '__main__':
  tf.app.run()