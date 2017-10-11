import numpy as np
import tensorflow as tf
import tensorflow.contrib
import random
import input
import model
import chess

FLAGS = tf.app.flags.FLAGS

with tf.device('/cpu:0'):
    board = tf.placeholder(tf.float32, shape=[1, 8, 8, 6])
    extrainfo = tf.placeholder(tf.float32, shape=[1])
    example = [board, extrainfo]
    logits, _ = model.model(example)

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    sess = tf.Session(config=config)
    checkpoint = tf.train.get_checkpoint_state(FLAGS.logdir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print ("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("No checkpoint found in logdir.")

def predict(fen):
    global logits, prediction, sess, label_strings, board
    _board = chess.Board(fen)
    encoded_board = input.encode_board(_board)
    encoded_board = [np.transpose(encoded_board, (1, 2, 0))]
    feed_dict = {board: encoded_board, extrainfo: [_board.halfmove_clock]}
    predictions = sess.run([logits], feed_dict=feed_dict)
    legal_moves = []
    for move in _board.legal_moves:
        legal_moves.append(move.uci())
    legal_labels = [int(mov in legal_moves) for mov in label_strings]
    return (predictions[0] * legal_labels)[0]

label_strings = input.load_labels()


#preds = predict("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
#for idx in range(len(label_strings)):
#    if preds[idx] != 0:
#        print(label_strings[idx], preds[idx])
