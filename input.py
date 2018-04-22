import os, fnmatch, zlib
import tensorflow as tf
import tensorflow.contrib
import numpy as np
import chess
import flags

FLAGS = flags.FLAGS
BATCH_SIZE = 256
MATE_CP_SCORE = 20000

device = None

if FLAGS.gpu != "true":
    device = "/cpu:0"

def find_files(directory, pattern):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def encode_board(board):
    rep = np.zeros((12, 8, 8), dtype=float)
    occupied_white = board.occupied_co[chess.WHITE]
    pawns = board.pawns
    knights = board.knights
    bishops = board.bishops
    rooks = board.rooks
    queens = board.queens
    kings = board.kings
    for col in range(0, 8):
        for row in range(0, 8):
            mask = chess.BB_SQUARES[row * 8 + col]
            if pawns & mask:
                rep[0, col, row] = 1 if occupied_white & mask else 0
                rep[6, col, row] = 1 - rep[0, col, row]
            elif knights & mask:
                rep[1, col, row] = 1 if occupied_white & mask else 0
                rep[7, col, row] = 1 - rep[1, col, row]
            elif bishops & mask:
                rep[2, col, row] = 1 if occupied_white & mask else 0
                rep[8, col, row] = 1 - rep[2, col, row]
            elif rooks & mask:
                rep[3, col, row] = 1 if occupied_white & mask else 0
                rep[9, col, row] = 1 - rep[3, col, row]
            elif queens & mask:
                rep[4, col, row] = 1 if occupied_white & mask else 0
                rep[10, col, row] = 1 - rep[4, col, row]
            elif kings & mask:
                rep[5, col, row] = 1 if occupied_white & mask else 0
                rep[11, col, row] = 1 - rep[5, col, row]

    if board.turn == chess.BLACK:
        rep = np.array([rep[6], rep[7], rep[8], rep[9], rep[10], rep[11], rep[0], rep[1], rep[2], rep[3], rep[4], rep[5]])
        rep = np.flip(rep, 2)

    return rep


def decode_board(encoded_board):
    board = chess.Board()
    for col in range(0, 8):
        for row in range(0, 8):
            square = chess.square(col, row)
            piece = None

            if encoded_board[col, row, 0] >= 0.5:
                 piece = chess.Piece(chess.PAWN, chess.WHITE)
            elif encoded_board[col, row, 6] >= 0.5:
                 piece = chess.Piece(chess.PAWN, chess.BLACK)
            elif encoded_board[col, row, 1] >= 0.5:
                 piece = chess.Piece(chess.KNIGHT, chess.WHITE)
            elif encoded_board[col, row, 7] >= 0.5:
                 piece = chess.Piece(chess.KNIGHT, chess.BLACK)
            elif encoded_board[col, row, 2] >= 0.5:
                 piece = chess.Piece(chess.BISHOP, chess.WHITE)
            elif encoded_board[col, row, 8] >= 0.5:
                 piece = chess.Piece(chess.BISHOP, chess.BLACK)
            elif encoded_board[col, row, 3] >= 0.5:
                 piece = chess.Piece(chess.ROOK, chess.WHITE)
            elif encoded_board[col, row, 9] >= 0.5:
                 piece = chess.Piece(chess.ROOK, chess.BLACK)
            elif encoded_board[col, row, 4] >= 0.5:
                 piece = chess.Piece(chess.QUEEN, chess.WHITE)
            elif encoded_board[col, row, 10] >= 0.5:
                 piece = chess.Piece(chess.QUEEN, chess.BLACK)
            elif encoded_board[col, row, 5] >= 0.5:
                 piece = chess.Piece(chess.KING, chess.WHITE)
            elif encoded_board[col, row, 11] >= 0.5:
                piece = chess.Piece(chess.KING, chess.BLACK)
            board.set_piece_at(square, piece)
    return board

def hash_32(str):
    return zlib.adler32(str.encode('utf-8')) & 0xffffffff

def sideswitch_label(label_str):
    retval = list(label_str)
    retval[1] = str(9 - int(retval[1]))
    retval[3] = str(9 - int(retval[3]))
    return "".join(retval)


def load_labels():
    with open(FLAGS.labels_file) as f:
        labels = f.readline().strip().split(" ")
        switch_indexer = [0] * len(labels)
        for idx, label in enumerate(labels):
            switched_label = sideswitch_label(label)
            switched_idx = labels.index(switched_label)
            switch_indexer[idx] = switched_idx
        return labels, switch_indexer

def _parse_example(example_proto):
    features = {
        "board/twelvelayer": tf.FixedLenFeature([768], tf.float32),
        "move/player": tf.FixedLenFeature((), tf.int64),
        "move/label": tf.FixedLenFeature((), tf.int64),
        "game/result": tf.FixedLenFeature((), tf.int64)
    }
    if FLAGS.disable_cp == "false":
        features["board/cp_score/"] = tf.FixedLenFeature((), tf.int64)

    parsed_features = tf.parse_single_example(example_proto, features)
    #cp_score = parsed_features["board/cp_score/"] if FLAGS.disable_cp == "false" else 0
    #cp_score = tf.cast(cp_score, tf.float32)
    board = tf.reshape(parsed_features["board/twelvelayer"], (12, 8, 8))
    board = tf.transpose(board, perm=[1, 2, 0])
    return (board, parsed_features["move/player"]), parsed_features["move/label"], parsed_features["game/result"]

def inputs(filenames, shuffle=True):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_example)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=50000)
    dataset = dataset.batch(BATCH_SIZE)
    if FLAGS.repeat_dataset != "false":
        dataset = dataset.repeat()
    return dataset

if __name__ == "__main__":
    board = chess.Board()
    board.push_uci("e2e4")
    encode_board(board)
