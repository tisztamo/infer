import unittest
from mock import patch
import chess, chess.uci
import numpy as np
import engine

class TestEngine(unittest.TestCase):
    def setUp(self):
        self.engine = engine.Engine()

    def tearDown(self):
        pass

    def test_cp_score(self):
        score = chess.uci.Score(-10, 0)
        self.assertEqual(engine.Engine.cp_score(score), -10)

        score = chess.uci.Score(5000, 0)
        self.assertEqual(engine.Engine.cp_score(score), 5000)

        score = chess.uci.Score(None, 10)
        self.assertEqual(engine.Engine.cp_score(score), 10000)

        score = chess.uci.Score(None, 11)
        self.assertEqual(engine.Engine.cp_score(score), 10000)

        score = chess.uci.Score(None, 1)
        self.assertEqual(engine.Engine.cp_score(score), 19000)

        score = chess.uci.Score(None, 0)
        self.assertEqual(engine.Engine.cp_score(score), 20000)

    def sample_prediction(self):
        preds = [0, 9] + [0] * 98 + [ 10, 0, 8, 7, 6, 5, 4, 3, 2, 1] + [0] * 90 + [11, 12, 13, 14, 15, 16, 17, 18, 19, 20] + [0] * 800
        return np.array(preds)

    @patch('inference.predict')
    def test_candidates(self, predict_mock):
        predict_mock.return_value = self.sample_prediction()
        candidates = self.engine.candidates(chess.Board())
        gt = [209, 208, 207, 206, 205, 204, 203, 202, 201, 200, 100, 1, 102, 103, 104, 105, 106, 107, 108, 109]
        np.testing.assert_array_equal(candidates, gt)

        candidates = self.engine.candidates(chess.Board(), try_move="a1a2")
        np.testing.assert_array_equal(candidates, [0] + gt)

        candidates = self.engine.candidates(chess.Board(), try_move="a1a3")
        gt.remove(1)
        gt = [1] + gt
        np.testing.assert_array_equal(candidates, gt)


if __name__ == '__main__':
    unittest.main()