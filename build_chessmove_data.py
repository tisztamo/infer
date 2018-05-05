from __future__ import division
from __future__ import print_function
import os
import time,random
from operator import attrgetter
import traceback
from os import listdir
from os.path import isfile, join
import codecs

import numpy as np
import math
import tensorflow as tf
import chess
import chess.pgn
import chess.uci
import log

import flags, input, engine, backengine

FLAGS = flags.FLAGS
labels = []

def filter_game(game):
    try:
        event = game.headers["Event"]
        white_elo = int(game.headers["WhiteElo"])
        black_elo = int(game.headers["BlackElo"])
    except:
        return False

    if white_elo < 2200 or black_elo < 2200:
        return False

    if FLAGS.omit_draws == "true" and game.headers["Result"] not in ["1-0", "0-1"]:
        if game.headers["Result"] != "1/2-1/2":
            print("Dropping game with result", game.headers["Result"])
        return False
    return event.find("960") == -1

def filter_move(board, move, player):
    if FLAGS.prune_opening == "true":
        move_idx = board.fullmove_number - 1
        if move_idx < random.randint(0, 12):
            return False
    if FLAGS.filter_player == "":
        return True
    else:
        if FLAGS.filter_player.startswith("-"):
            return not FLAGS.filter_player[1:] == player
        else:
            return FLAGS.filter_player == player

def parse_eval_comment(move):
    pass

def init_engines(num=4):
    retval = []
    for i in range(num):
        retval.append(backengine.BackEngine())
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


def _convert_to_example(board, move, game, player, pvs):
    game_headers = game.headers
    layer_rep = input.encode_board(board)
    move_uci = move.uci()
    result = game_headers["Result"]
    result_code = 0
    if result == "1-0":
        result_code = 1
    elif result == "0-1":
        result_code = -1

    if board.turn == chess.BLACK:
        move_uci = input.sideswitch_label(move_uci)
        result_code = -result_code
        for pv in pvs:
            if pv.moves is not None:
                pv.moves[0] = chess.Move.from_uci(input.sideswitch_label(pv.moves[0].uci()))

    try:
        ply_count = int(game_headers["PlyCount"])
    except:
        ply_count = -1


    feature_desc = {
        'board/twelvelayer': tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(layer_rep))),
        'board/fen': _bytes_feature(tf.compat.as_bytes(board.fen())),
        'game/result': _int64_feature(result_code),
        'move/player': _int64_feature(input.hash_32(player)),
        'move/uci': _bytes_feature(tf.compat.as_bytes(move_uci)),
        'move/label': _int64_feature(labels.index(move_uci))
    }
    if False:
        print("-------------- " + board.fen())
        print(board)
        print("turn:" + str(board.turn) +  "player: " + player)
        print("move: " + move.uci() + ", normalized: " + move_uci)
        print("Result: " + result + ", normalized code: " + str(result_code))
        print(layer_rep)

    max_score = max(pvs, key = attrgetter('score')).score
    for i in range(input.MULTIPV):
        if pvs[i].moves is None:
            rel_score = 0
            move = input.encode_move(None)
        else:
            move = input.encode_move(pvs[i].moves[0])
            score_diff = pvs[i].score - max_score
            rel_score =  1 + np.tanh(score_diff * 0.005)  
            #print("Move: " + pvs[i].moves[0].uci() + ", Score: " + str(pvs[i].score) + ", diff: " + str(score_diff) + " Rel score: " + str(rel_score))
            #print(move)
        feature_desc["eval/pv" + str(i) + "/movelayers"] =  tf.train.Feature(float_list=tf.train.FloatList(value=np.ravel(move))),
        feature_desc["eval/pv" + str(i) + "/cp_score"] =  _int64_feature(pvs[i].score)
        feature_desc["eval/pv" + str(i) + "/rel_score"] =  _float32_feature(rel_score)

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
    num_dropped_moves = 0
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

                    if not result_found:
                        result_found = engine.wait_for_evaluation()

                    if result_found:
                        last_board, last_move, last_game, pvs = engine.get_multipv_result()
                        #print("Best move: " + str(pvs[0].moves[0]) + " for board " + last_board.fen()) 
                        #print(last_board)
                        player = players[0 if last_board.turn == chess.WHITE else 1]
                        example = _convert_to_example(last_board, last_move, last_game, player, pvs)
                        writer.write(example.SerializeToString())
                except Exception as x:
                   print("Error while processing:", x)
                   traceback.print_exc()

                player = players[0 if board.turn == chess.WHITE else 1]
                if filter_move(board, move, player):
                    num_moves += 1
                    engine.start_evaluate_board(board, game, move)
                    engine_idx = (engine_idx + 1) % len(engines)
                else:
                    num_dropped_moves += 1
                board.push(move)

            num_processed += 1
            if num_processed % 10 == 1:
                print("Processed", num_processed, "games, dropped", num_filtered_out, ";", num_moves, "moves, dropped ", num_dropped_moves)


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
        output_file = os.path.join(FLAGS.data_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)
        processed, dropped, moves = _process_pgn_file(join(directory, filename), writer, engines)
        print("Dropped", dropped, "games, processed", processed, ", containing", moves, "halfmoves.")
        writer.close()
        time.sleep(3)
        

def main(unused_argv):
    global labels
    print('Saving results to %s' % FLAGS.data_dir)
    labels, _ = input.load_labels()

    engines = init_engines(7)

    #_process_dataset('validation', FLAGS.validation_dir, engines)
    _process_dataset('train', FLAGS.train_dir, engines)

if __name__ == '__main__':
  tf.app.run()