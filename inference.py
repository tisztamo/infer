import math
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib
#from tensorflow.python import debug as tf_debug
import random
import chess
import input
import model
from experiments.evaluate import siamese_model 

FLAGS = tf.app.flags.FLAGS

with tf.device(input.device):
    board = tf.placeholder(tf.float32, shape=[1, 8, 8, 12])
    player = tf.placeholder(tf.float32, shape=[1])
    example = [board, player]

    features = model.feature_extractor(example)
    logits = model.policy_model(example, features)
    if FLAGS.predict_result != "false":
        result_prediction = model.result_model(example, features)
    
    move_vars = tf.trainable_variables()

    if FLAGS.siamese_eval != "false":
        siamese_input, predicted_score_diff, siamese_vars = siamese_model.score_diff_predictor(features.shape)

    move_saver = tf.train.Saver(var_list = move_vars)
    siamese_saver = tf.train.Saver(var_list = siamese_vars)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    sess = tf.Session(config=config)#tf_debug.LocalCLIDebugWrapperSession(...)
    checkpoint = tf.train.get_checkpoint_state(FLAGS.logdir)
    if checkpoint and checkpoint.model_checkpoint_path:
        move_saver.restore(sess, checkpoint.model_checkpoint_path)
        print ("Successfully loaded move predictor:", checkpoint.model_checkpoint_path)
    else:
        print("No move checkpoint found in logdir.")

    checkpoint = tf.train.get_checkpoint_state(FLAGS.logdir + "/siamese/")
    if checkpoint and checkpoint.model_checkpoint_path:
        siamese_saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded siamese net:", checkpoint.model_checkpoint_path)
    else:
        print("No siamese checkpoint found.")


def create_feed(board_, player_="?"):
    encoded_board = input.encode_board(board_)
    encoded_board = [np.transpose(encoded_board, (1, 2, 0))]
    player_hash = float(input.hash_32(player_))
    return {board: encoded_board, player: [player_hash]}

def predict_move(fen, player_="?"):
    board_ = chess.Board(fen)
    feed_dict = create_feed(board_, player_)
    move_preds = sess.run([logits], feed_dict=feed_dict)
    legal_moves = []
    for move in board_.legal_moves:
        legal_moves.append(move.uci())
    legal_labels = [int(mov in legal_moves) for mov in label_strings]
    preds = move_preds[0][0] if board_.turn == chess.WHITE else np.array([move_preds[0][0][switch_indexer]][0])
    predicted_moves = (preds * legal_labels)
    return predicted_moves

def predict_result(board_, player_="?"):
    feed_dict = create_feed(board_, player_)
    result_pred = sess.run([result_prediction], feed_dict=feed_dict)
    return result_pred[0][0]

def predict_eval(board_):
    b = board_.copy()
    feed_dict = create_feed(b)
    v_features1 = sess.run(features, feed_dict=feed_dict)

    b.turn = chess.WHITE if b.turn == chess.BLACK else chess.BLACK
    feed_dict = create_feed(b)
    v_features2 = sess.run(features, feed_dict=feed_dict)

    v_siamese_input = np.concatenate([v_features1, v_features2], 1)
    v_predicted_score_diff = sess.run(predicted_score_diff, feed_dict={siamese_input:v_siamese_input})
    return v_predicted_score_diff[0][0] * 1500.0

def visualize_layer(board_):
    feed_dict = create_feed(board_)
    for op in tf.get_default_graph().get_operations():
        if op.name.find("Relu") != -1:
            tensor_to_vis = tf.get_default_graph().get_tensor_by_name(op.name + ":0")
            if  len(tensor_to_vis.shape) == 4:
                print(op.name)
                units = sess.run(tensor_to_vis, feed_dict=feed_dict)
                filters = units.shape[3]
                plt.figure(1, figsize=(40,40))
                n_columns = 16
                n_rows = math.ceil(filters / n_columns) + 1
                for i in range(filters):
                    plt.subplot(n_rows, n_columns, i+1)
                    plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
                plt.savefig(op.name + ".png")
                plt.clf()  

label_strings, switch_indexer = input.load_labels()


def main(unused_argv):
    #preds, result_pred = predict("r4r1R/pb2bkp1/4p3/3p1p1q/1ppPnB2/2P1P3/PPQ2PP1/2K4R w - - 0 22")
    #preds, result_pred = predict("r4r1R/pb2bkp1/4p3/3p1p1R/1ppPnB2/2P1P3/PPQ2PP1/2K5 b - - 0 22")
    #result_pred = predict_result(chess.Board("r1bqkbB1/2ppn1p1/ppP2p1p/4p3/4P3/2N2N2/PPP2PPP/R1BQK2R b KQq - 0 9"))
    #print("Result:", result_pred)
    b = chess.Board("r5k1/6b1/2p2pp1/q4b1p/1ppP3P/5P2/1PPQ2PN/1K1R3R w - - 3 24")
    b.turn=chess.BLACK
    eval_pred = predict_eval(b)
    print(b)
    print("Eval:", eval_pred)
    #visualize_layer(chess.Board("r1bqkbB1/2ppn1p1/ppP2p1p/4p3/4P3/2N2N2/PPP2PPP/R1BQK2R b KQq - 0 9"))
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
