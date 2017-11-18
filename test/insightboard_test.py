import unittest, random
from mock import patch
import chess, chess.uci
import numpy as np
import tensorflow as tf
from analyse import hints, insight_board

class TestHints(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_first_pawn_rank_in_file(self):
        board = insight_board.from_board(chess.Board())
        self.assertEqual(board.first_pawn_rank_in_file(0), 1)
        self.assertEqual(board.first_pawn_rank_in_file(4), 1)
        self.assertEqual(board.first_pawn_rank_in_file(5, chess.BLACK), 6)
        board.push_uci("e2e4")
        self.assertEqual(board.first_pawn_rank_in_file(4), 6)#Black!
        board.push_uci("d7d6")
        self.assertEqual(board.first_pawn_rank_in_file(4), 3)#White
        self.assertEqual(board.first_pawn_rank_in_file(3, chess.BLACK), 5)
        self.assertEqual(board.first_pawn_rank_in_file(4, chess.WHITE), 3)
        

if __name__ == '__main__':
    unittest.main()