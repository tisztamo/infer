import numpy as np
import tensorflow as tf
import tensorflow.contrib
import random
import input
import model
import chess

tf.app.flags.DEFINE_string('gpu', 'true',
                           'Use the GPU')

FLAGS = tf.app.flags.FLAGS

if FLAGS.gpu == "true":
    device = None
else:
    device = "/cpu:0"

with tf.device(device):
    board = tf.placeholder(tf.float32, shape=[1, 8, 8, 6])
    player = tf.placeholder(tf.float32, shape=[1])
    example = [board, player]

    features, _ = model.feature_extractor(example)
    logits, _ = model.policy_model(example, features)
    result_prediction, _ = model.result_model(example, features)

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

def predict_move(fen, player_="?"):
    board_ = chess.Board(fen)
    encoded_board = input.encode_board(board_)
    encoded_board = [np.transpose(encoded_board, (1, 2, 0))]
    player_hash = float(input.hash_32(player_))
    feed_dict = {board: encoded_board, player: [player_hash]}
    move_preds = sess.run([logits], feed_dict=feed_dict)
    legal_moves = []
    for move in board_.legal_moves:
        legal_moves.append(move.uci())
    legal_labels = [int(mov in legal_moves) for mov in label_strings]
    preds = move_preds[0][0] if board_.turn == chess.WHITE else np.array([move_preds[0][0][switch_indexer]][0])
    predicted_moves = (preds * legal_labels)
    return predicted_moves

def predict_result(board_, player_="?"):
    encoded_board = input.encode_board(board_)
    encoded_board = [np.transpose(encoded_board, (1, 2, 0))]
    player_hash = float(input.hash_32(player_))
    feed_dict = {board: encoded_board, player: [player_hash]}
    result_pred = sess.run([result_prediction], feed_dict=feed_dict)
    return result_pred[0][0][0]

label_strings, switch_indexer = input.load_labels()


def main(unused_argv):
    #preds, result_pred = predict("r4r1R/pb2bkp1/4p3/3p1p1q/1ppPnB2/2P1P3/PPQ2PP1/2K4R w - - 0 22")
    #preds, result_pred = predict("r4r1R/pb2bkp1/4p3/3p1p1R/1ppPnB2/2P1P3/PPQ2PP1/2K5 b - - 0 22")
    result_pred = predict_result(chess.Board("r1bqkb1r/2ppn1pp/ppn2p2/3Pp3/2B1P3/2N2N2/PPP2PPP/R1BQK2R b KQkq - 0 7"))
    print("Result:", result_pred)
    # preds = predict_move("rnbqkbnr/ppppp1pp/5p2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    # print(preds)
    # argmax = np.argmax(preds, 0)
    # print("Best: " + label_strings[argmax] + " " + str(preds[argmax]))
    # candidates = np.argpartition(preds, -20)[-20:]
    # candidates = candidates[np.argsort(-preds[candidates])]

    # for idx in range(len(candidates)):
    #     print(label_strings[candidates[idx]], preds[candidates[idx]])

if __name__ == "__main__":
    tf.app.run()
