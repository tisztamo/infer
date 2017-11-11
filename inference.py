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
    turn = tf.placeholder(tf.float32, shape=[1])
    player = tf.placeholder(tf.float32, shape=[1])
    example = [board, turn, player]
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

def predict(fen, player_="?"):
    global logits, prediction, sess, label_strings, board
    board_ = chess.Board(fen)
    #print(board_)
    encoded_board = input.encode_board(board_)
    encoded_board = [np.transpose(encoded_board, (1, 2, 0))]
    turn_ = 1.0# if board_.turn == chess.WHITE else 0.0
    player_hash = 0.1#float(input.hash_32(player_))
    feed_dict = {board: encoded_board, turn: [turn_], player: [player_hash]}
    predictions = sess.run([logits], feed_dict=feed_dict)
    legal_moves = []
    for move in board_.legal_moves:
        legal_moves.append(move.uci())
    legal_labels = [int(mov in legal_moves) for mov in label_strings]
    preds = predictions[0] if board_.turn == chess.WHITE else np.array([predictions[0][0][switch_indexer]])
    retval = (preds * legal_labels)[0]
    return retval

label_strings, switch_indexer = input.load_labels()


def main():
    for i in range(1):
        preds=predict("rnbqkbnr/ppppp1pp/5p2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        argmax = np.argmax(preds, 0)
        print("Best: " + label_strings[argmax] + " " + str(preds[argmax]))
    #preds = predict("r4r1R/pb2bkp1/4p3/3p1p1q/1ppPnB2/2P1P3/PPQ2PP1/2K4R w - - 0 22", "Karpov, Anatoly")
    #preds = predict("r3k2r/p7/Bp2p3/2pqPnpp/3p4/P7/1PP2PPP/R2QR1K1 w kq - 2 19", "Karpov, Anatoly")
    argmax = np.argmax(preds, 0)
    print("Best: " + label_strings[argmax] + " " + str(preds[argmax]))
    candidates = np.argpartition(preds, -20)[-20:]
    candidates = candidates[np.argsort(-preds[candidates])]

    for idx in range(len(candidates)):
        print(label_strings[candidates[idx]], preds[candidates[idx]])

if __name__ == "__main__":
    main()
