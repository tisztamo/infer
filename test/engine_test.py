import unittest, random
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
    def test_candidate_idxs(self, predict_mock):
        predict_mock.return_value = self.sample_prediction()
        candidates, probs = self.engine.candidate_idxs(chess.Board())
        gt = [209, 208, 207, 206, 205, 204, 203, 202, 201, 200, 100, 1, 102, 103, 104, 105, 106, 107, 108, 109]
        np.testing.assert_array_equal(candidates, gt)

        candidates, probs = self.engine.candidate_idxs(chess.Board(), try_move="a1a2")
        np.testing.assert_array_equal(candidates, [0] + gt)

        candidates, probs = self.engine.candidate_idxs(chess.Board(), try_move="a1a3")
        gt.remove(1)
        gt = [1] + gt
        np.testing.assert_array_equal(candidates, gt)

    def test_appeal_higher_prob_accounts_score_loss(self):
        dobule_prob_accountsfor_cp = engine.DOUBLE_PROB_ACCOUNTS_FOR_CP
        for i in range(1000):
            base_prob = random.random() * 15
            best_score = random.randint(-5000, 5000)
            c1 = engine.CandidateMove("e2e4", base_prob, best_score)
            c2 = engine.CandidateMove("d2d4", 2 * base_prob, best_score - dobule_prob_accountsfor_cp)
            c3 = engine.CandidateMove("g1f3", 4 * base_prob, best_score - 2 * dobule_prob_accountsfor_cp)
            #print(c1.cp_score, c1.probability, ":", c2.cp_score, c2.probability, ":", c3.cp_score, c3.probability)
            c1_appeal = c1.calculate_appeal(best_score)
            self.assertAlmostEqual(c1_appeal, c2.calculate_appeal(best_score))
            self.assertAlmostEqual(c1_appeal, c3.calculate_appeal(best_score))

if __name__ == '__main__':
    unittest.main()