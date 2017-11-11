import tensorflow as tf
import numpy as np
import chess
import model
import input

FLAGS = tf.app.flags.FLAGS

label_strings, _ = input.load_labels()

with tf.device('/cpu:0'):
    board = tf.placeholder(tf.float32, shape=[1, 8, 8, 6])
    turn = tf.placeholder(tf.float32, shape=[1])
    #player = tf.placeholder(tf.float32, shape=[1])
    label = tf.placeholder(tf.int64, shape=[1])
    example = [board, turn]
    features, _ = model.feature_extractor([board])
    logits, _ = model.model(example, features)
    
    onehot_labels = tf.one_hot(label, model.NUM_LABELS, dtype=tf.float32)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=onehot_labels))

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


def normalize_board(encoded_board):
    THRESHOLD = 0.9
    retval = np.zeros_like(encoded_board)
    for i in range(encoded_board.shape[0]):
        for j in range(encoded_board.shape[1]):
            for k in range(encoded_board.shape[2]):
                if encoded_board[i, j, k] < -THRESHOLD:
                    retval[i, j, k] = -1
                elif encoded_board[i, j, k] > THRESHOLD:
                    retval[i, j, k] = 1
    return retval


def generate_board(start_fen):
    current_board = input.encode_board(chess.Board(start_fen))
    print_board(current_board)
    current_board = np.transpose(current_board, (1, 2, 0))
    gradients = tf.gradients(loss, board)
    move = label_strings.index("g1f3")
    print("XXXXXXXXX")
    for i in range(100):
        grad_v = sess.run(gradients, feed_dict = {
            board: [current_board],
            turn: [1],
            label: [0]})[0][0]
        current_board += grad_v / (20 * np.abs(grad_v).mean()+1e-7)
        print_board(normalize_board(np.transpose(current_board, (2, 0, 1))))
    
    current_board = np.transpose(current_board, (2, 0, 1))
    return normalize_board(current_board)

def print_board(encoded_board):
    board = input.decode_board(encoded_board)
    print(board)
    print(board.fen())


def main():
    board = generate_board(chess.Board().fen())
    print_board(board)
    

if __name__ == "__main__":
    main()

