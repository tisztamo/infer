import unittest, random
from mock import patch
import chess, chess.uci
import numpy as np
import tensorflow as tf
from analyse import hints, insight_board

tf.app.flags.DEFINE_string('analyse_test_csv', 'test/analyse_test.csv',
                           'Path to CSV file containing analyser test data')

class TestHints(unittest.TestCase):
    def setUp(self):
        self.hinter = hints.Hinter()

    def tearDown(self):
        pass

    def create_board(self, descriptor):
        words = descriptor.split(" ")
        if len(words[0]) < 10:#moves
            board = chess.Board()
            for move in words:
                board.push_san(move)
        else:
            board = chess.Board(descriptor)
        return insight_board.from_board(board)


    def test_from_csv(self):
        with open(tf.app.flags.FLAGS.analyse_test_csv, "r") as csv:
            csv.readline()
            for line in csv.readlines():
                print(line)
                words = line.split(",")
                board = self.create_board(words[0])
                print(board.fen())
                move = board.parse_san(words[1])
                best_move = board.parse_san(words[2])
                expected_hints = [int(hint) for hint in words[3].split(" ")]
                got_hints = self.hinter.getHints(board, move, best_move)
                got_hint_ids = [hint.id for hint in got_hints]
                self.assertEqual(expected_hints, got_hint_ids)

if __name__ == '__main__':
    unittest.main()