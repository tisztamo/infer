import math
from operator import attrgetter
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib
#from tensorflow.python import debug as tf_debug
import random
import input
import model
import chess
import flags
import backengine

FLAGS = flags.FLAGS

with tf.device(input.device):
    board = tf.placeholder(tf.float32, shape=[1, 8, 8, 12])
    movelayers = tf.placeholder(tf.float32, shape=[1, 8, 8, 2 * input.MULTIPV])
    movescores = tf.placeholder(tf.float32, shape=[1, input.MULTIPV])
    result = tf.placeholder(tf.float32, shape=[1])
    example = [board, movelayers, movescores, result]

    features = model.feature_extractor(example)
    logits = model.policy_model(example, features)

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    sess = tf.Session(config=config)#tf_debug.LocalCLIDebugWrapperSession(...)
    checkpoint = tf.train.get_checkpoint_state(FLAGS.logdir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print ("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("No checkpoint found in logdir.")

def create_feed(board_, pvs, result_=1):
    encoded_board = input.encode_board(board_)
    encoded_board = [np.transpose(encoded_board, (1, 2, 0))]

    if board_.turn == chess.BLACK:
        for pv in pvs:
            if pv.moves is not None:
                pv.moves[0] = chess.Move.from_uci(input.sideswitch_label(pv.moves[0].uci()))

    movelayers_ = []
    movescores_ = []
    max_score = max(pvs, key = attrgetter('score')).score
    for i in range(input.MULTIPV):
        if pvs[i].moves is None:
            rel_score = 0
            movelayer = input.encode_move(None)
        else:
            movelayer = input.encode_move(pvs[i].moves[0])
            score_diff = pvs[i].score - max_score
            rel_score =  1 + np.tanh(score_diff * 0.005)  
            print(pvs[i].moves[0].uci() + ": " +str(pvs[i].score) + ", " + str(score_diff) + ", " + str(rel_score))

        movelayer = np.transpose(movelayer, axes=[1, 2, 0])
        movelayer = np.multiply(movelayer, rel_score)
        #print("movelayer:")
        #print(movelayer.shape)
        #print(movelayer)
        movelayers_.append(movelayer)

        movescore = rel_score
        movescores_.append(movescore)

    movelayers_concat = np.concatenate(movelayers_, axis=2)
    #print("movelayers:")
    #print(movelayers_concat.shape)
    #print(movelayers_concat)
    movescores_concat = np.stack(movescores_)
    print("movescores:")
    print(movescores_concat)

    return {board: encoded_board, movelayers: [movelayers_concat], movescores: [movescores_concat], result: [result_]}

def predict_move(fen, pvs, result_ = 1, player_="?"):
    board_ = chess.Board(fen)
    feed_dict = create_feed(board_, pvs, result_)
    move_preds = sess.run([logits], feed_dict=feed_dict)
    legal_moves = []
    for move in board_.legal_moves:
        legal_moves.append(move.uci())
    legal_labels = [int(mov in legal_moves) for mov in label_strings]
    preds = move_preds[0][0] if board_.turn == chess.WHITE else np.array([move_preds[0][0][switch_indexer]][0])
    predicted_moves = (preds * legal_labels)
    return predicted_moves

# def visualize_layer(board_):
#     feed_dict = create_feed(board_)
#     for op in tf.get_default_graph().get_operations():
#         if op.name.find("Relu") != -1:
#             tensor_to_vis = tf.get_default_graph().get_tensor_by_name(op.name + ":0")
#             if  len(tensor_to_vis.shape) == 4:
#                 print(op.name)
#                 units = sess.run(tensor_to_vis, feed_dict=feed_dict)
#                 filters = units.shape[3]
#                 plt.figure(1, figsize=(40,40))
#                 n_columns = 16
#                 n_rows = math.ceil(filters / n_columns) + 1
#                 for i in range(filters):
#                     plt.subplot(n_rows, n_columns, i+1)
#                     plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
#                 plt.savefig(op.name + ".png")
#                 plt.clf()  

label_strings, switch_indexer = input.load_labels()


def main(unused_argv):
    pass
    #preds, result_pred = predict("r4r1R/pb2bkp1/4p3/3p1p1q/1ppPnB2/2P1P3/PPQ2PP1/2K4R w - - 0 22")
    #preds, result_pred = predict("r4r1R/pb2bkp1/4p3/3p1p1R/1ppPnB2/2P1P3/PPQ2PP1/2K5 b - - 0 22")
    #result_pred = predict_result(chess.Board("r1bqkbB1/2ppn1p1/ppP2p1p/4p3/4P3/2N2N2/PPP2PPP/R1BQK2R b KQq - 0 9"))
    #print("Result:", result_pred)
    #visualize_layer(chess.Board("r1bqkbB1/2ppn1p1/ppP2p1p/4p3/4P3/2N2N2/PPP2PPP/R1BQK2R b KQq - 0 9"))
    #preds = predict_move("6k1/5p1p/4n3/4Pp2/3p2P1/2r2N1K/7P/4R3 w - - 1 46",
    #[backengine.PV([chess.Move.from_uci("e1a1")], 0), backengine.PV([chess.Move.from_uci("e1d1")], -100), backengine.PV([chess.Move.from_uci("g4f5")], -200), backengine.PV([chess.Move.from_uci("h3g2")], 0)], 1)
    preds = predict_move("r4r1R/pb2bkp1/4p3/3p1p1R/1ppPnB2/2P1P3/PPQ2PP1/2K5 b - - 0 22",
    [backengine.PV([chess.Move.from_uci("f8h8")], 0), backengine.PV([chess.Move.from_uci("b4c3")], 0), backengine.PV([chess.Move.from_uci("f8c8")], -0), backengine.PV([chess.Move.from_uci("e4f6")], -00)], 1)
    argmax = np.argmax(preds, 0)
    print("Best: " + label_strings[argmax] + " " + str(preds[argmax]))
    candidates = np.argpartition(preds, -10)[-10:]
    candidates = candidates[np.argsort(-preds[candidates])]

    for idx in range(len(candidates)):
        print(label_strings[candidates[idx]], preds[candidates[idx]])


if __name__ == "__main__":
    tf.app.run()
