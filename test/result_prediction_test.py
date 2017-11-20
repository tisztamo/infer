import chess
import tensorflow as tf
import inference
import engine

tf.app.flags.DEFINE_string('csv_in', 'test/result_prediction_test.csv',
                           'Path to input CSV file containing fens and evaluations')

FLAGS = tf.app.flags.FLAGS

def main(unused_argv):
    FLAGS.use_back_engine = "true"
    e = engine.Engine()
    e.initBackEngine()
    with open(FLAGS.csv_in) as csv_in:
        print("fen,best_move,move,evaluation,result_prediction")
        line = csv_in.readline()
        while True:
            line = csv_in.readline()
            if line is None or line == "":
                break
            words = line.split(",")
            fen = words[0].strip()
            board = chess.Board(fen)
            best_move = words[1].strip()
            move = words[2].strip()
            known_eval = 0
            try:
                known_eval = float(words[3].strip())
            except:
                m_, score, p_ = e.evaluateStatic(board)
                known_eval = e.cp_score(score)
                if board.turn == chess.BLACK:
                    known_eval = -known_eval
            result_pred = inference.predict_result(board)
            if board.turn == chess.BLACK:
                result_pred = - result_pred
            print(fen + "," + best_move + "," + move + "," + str(known_eval) + "," + str(result_pred))

if __name__ == "__main__":
    tf.app.run()
