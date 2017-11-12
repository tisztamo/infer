import codecs
import tensorflow as tf
import chess, chess.pgn

import log, engine

# Usage: python analyse.py --pgn_in=PGN_FILE

tf.app.flags.DEFINE_string('pgn_in', 'game.pgn',
                           'Path to pgn file containing games to analyse')
tf.app.flags.DEFINE_string('encoding', 'latin-1',
                           'input pgn encoding (latin-1 or utf-8-sig)')

FLAGS = tf.app.flags.FLAGS

logger = log.getLogger("analyse")

def _open_pgn_file(filename):
    f = codecs.open(filename, encoding=FLAGS.encoding)
    return f

def analyse_game(game, turk):
    board = game.board()
    node = game
    while len(node.variations) > 0:
        next_node = node.variations[0]
        move = next_node.move
        turk_move = turk.bestMove(board, try_move=move.uci())
        if turk_move is not None and turk_move.uci != move.uci():
            node.add_variation(chess.Move.from_uci(turk_move.uci))
        board.push(move)
        node = next_node

    return game

def main(unused_argv):
    with _open_pgn_file(FLAGS.pgn_in) as pgn_in:
        turk = engine.Engine()
        turk.initBackEngine()
        game = chess.pgn.read_game(pgn_in)
        analysed = analyse_game(game, turk)
        print(analysed)
        exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
        pgn_string = analysed.accept(exporter)
        print(pgn_string)

if __name__ == '__main__':
  tf.app.run()