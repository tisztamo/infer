from __future__ import print_function
import codecs
import tensorflow as tf
import chess, chess.pgn

import log, engine
from analyse import hints, insight_board

# Usage: python -m analyse/analyser --pgn_in=PGN_FILE

tf.app.flags.DEFINE_string('pgn_in', 'game.pgn',
                           'Path to pgn file containing games to analyse')
tf.app.flags.DEFINE_string('pgn_out', 'analysed.pgn',
                           'Output file')
tf.app.flags.DEFINE_string('encoding', 'utf-8-sig',
                           'input pgn encoding (latin-1 or utf-8-sig)')

FLAGS = tf.app.flags.FLAGS

logger = log.getLogger("analyse")

hinter = hints.Hinter()

def _open_pgn_file(filename):
    f = codecs.open(filename, encoding=FLAGS.encoding)
    return f

def comment_on_move(node, board, move, score, turk_move):
    if score < turk_move.cp_score - 100:
        #, chess.Move.from_uci(turk_move.ponder)
        hint_move = chess.Move.from_uci(turk_move.uci)
        hint = hinter.getHint(board, move, hint_move)
        hint_text = ""
        if hint is not None:
            hint_text = " Hint: " + hint.text
        node.variations[0].comment = "Mistake (" + str(score - turk_move.cp_score) + ")" + hint_text
        node.variations = node.variations[:1]
        variation = node.add_line([hint_move])

def analyse_game(game, turk):
    board = insight_board.from_board(game.board())
    node = game
    i = 1
    while len(node.variations) > 0:
        print(game.headers["White"] + " vs " + game.headers["Black"] + ": " + str(i), end="\r")
        i = i + 1
        next_node = node.variations[0]
        move = next_node.move
        turk_move = turk.bestMove(board, try_move=move.uci())
        next_board = board.copy()
        next_board.push(move)

        if turk_move is not None and turk_move.uci != move.uci():
            _move, score, _ponder = turk.evaluate(next_board)#TODO harmonize depth with turk.bestMove
            score = -score
            comment_on_move(node, board, move, score, turk_move)

        node = next_node
        board.push(move)

    return game

def main(unused_argv):
    with _open_pgn_file(FLAGS.pgn_in) as pgn_in:
        turk = engine.Engine()
        turk.initBackEngine()
        game = chess.pgn.read_game(pgn_in)
        analysed = analyse_game(game, turk)
        exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
        pgn_string = analysed.accept(exporter)
        print(pgn_string)
        with codecs.open(FLAGS.pgn_out, "w", encoding=FLAGS.encoding) as pgn_out:
            pgn_out.write(pgn_string)
            pgn_out.write("\n")

if __name__ == '__main__':
  tf.app.run()