import unittest, random
from mock import patch
import chess, chess.uci
import engine, strength

class TestStrengthManager(unittest.TestCase):
    def setUp(self):
        self.engine = engine.Engine()
        self.manager = strength.StrengthManager(engine)

    def tearDown(self):
        pass

    def test_material_value(self):
        board = chess.Board(fen="1q1r2k1/2p2pp1/2P2n1p/QPbp1P2/8/6P1/R3RP1P/4B1K1 w - - 3 33")
        value = self.manager.material_value(board)
        self.assertEqual(value, 53)

    def test_game_stage(self):
        board = chess.Board()
        self.assertAlmostEqual(self.manager.game_stage(board), 0.0)

        board = chess.Board(fen="8/3r4/q6K/1p3p2/2p3k1/8/8/8 w - - 10 63")
        self.assertAlmostEqual(self.manager.game_stage(board), 0.78205128)

    def test_double_prob_accounts_for_cp(self):
        board = chess.Board()
        self.assertAlmostEqual(self.manager.double_prob_accounts_for_cp(board), strength.DOUBLE_PROB_ACCOUNTS_FOR_CP)

        board = chess.Board(fen="8/3r4/q6K/1p3p2/2p3k1/8/8/8 w - - 10 63")
        self.assertAlmostEqual(self.manager.double_prob_accounts_for_cp(board), 0.21794872 * strength.DOUBLE_PROB_ACCOUNTS_FOR_CP, 5)
        