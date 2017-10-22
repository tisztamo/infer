import os, fnmatch, zlib
import tensorflow as tf
import tensorflow.contrib
import numpy as np
import chess

tf.app.flags.DEFINE_string('data_dir', '/mnt/red/train/humanlike/preprocessed/',
                           'Preprocessed training data directory')
tf.app.flags.DEFINE_string('labels_file', 'labels.txt',
                           'List of all labels (uci move notation)')
tf.app.flags.DEFINE_string('logdir', '/mnt/red/train/humanlike/logdir',
                           'Directory to store network parameters and training logs')
tf.app.flags.DEFINE_string('disable_cp', 'true',
                           'Do not load of cp_score field from the tfrecord data files')

FLAGS = tf.app.flags.FLAGS
BATCH_SIZE = 16
MATE_CP_SCORE = 20000

def find_files(directory, pattern):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def encode_board(board):
    rep = np.zeros((6, 8, 8), dtype=float)
    occupied_white = board.occupied_co[chess.WHITE]
    pawns = board.pawns
    knights = board.knights
    bishops = board.bishops
    rooks = board.rooks
    queens = board.queens
    kings = board.kings
    for col in range(0, 8):
        for row in range(0, 8):
            mask = chess.BB_SQUARES[row * 8 +col]
            if pawns & mask:
                rep[0, col, row] = 1 if occupied_white & mask else -1
            elif knights & mask:
                rep[1, col, row] = 1 if occupied_white & mask else -1
            elif bishops & mask:
                rep[2, col, row] = 1 if occupied_white & mask else -1
            elif rooks & mask:
                rep[3, col, row] = 1 if occupied_white & mask else -1
            elif queens & mask:
                rep[4, col, row] = 1 if occupied_white & mask else -1
            elif kings & mask:
                rep[5, col, row] = 1 if occupied_white & mask else -1
    return rep

def hash_32(str):
    return zlib.adler32(str)

def load_labels():
    with open(FLAGS.labels_file) as f:
        return f.readline().strip().split(" ")

def _parse_example(example_proto):
    features = {
        "board/sixlayer": tf.FixedLenFeature([384], tf.float32),
        "move/turn": tf.FixedLenFeature((), tf.int64),
        "move/player": tf.FixedLenFeature((), tf.int64),
        "move/label": tf.FixedLenFeature((), tf.int64)
    }
    if FLAGS.disable_cp == "false":
        features["board/cp_score/"] = tf.FixedLenFeature((), tf.int64)

    parsed_features = tf.parse_single_example(example_proto, features)
    cp_score = parsed_features["board/cp_score/"] if FLAGS.disable_cp == "false" else 0
    cp_score = tf.cast(cp_score, tf.float32)
    board = tf.reshape(parsed_features["board/sixlayer"], (6, 8, 8))
    board = tf.transpose(board, perm=[1, 2, 0])
    return (board, parsed_features["move/turn"], parsed_features["move/player"]), parsed_features["move/label"], cp_score

def inputs(filenames, shuffle=True):
    dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_example)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(BATCH_SIZE)#.repeat()
    return dataset
