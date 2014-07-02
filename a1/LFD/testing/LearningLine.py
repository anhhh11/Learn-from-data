import unittest
from a1.LFD.LearningLine import LearningLine
import numpy as np
class LearningLineTest(unittest.TestCase):
    def setUp(self):
        self.ll = LearningLine(0,1)
        self.new_line = LearningLine(2,-3)
        self.test_points = np.array([
            [0,2],
            [2,0],
            [1,1],
            [2,2]
        ])
        self.test_results = np.array([1,-1,0,0])

    def test_get_betas(self):
        self.assertTrue(
            np.array_equiv(
                self.ll.get_betas(),
                np.array([0,1])
            )
        )

    def test_classify_one_point(self):
        for i in xrange(self.test_points.shape[0]):
            self.assertEqual(self.ll.classify(self.test_points[i,:]),
                             self.test_results[i])

    def test_classify_multiple_points(self):
        self.assertTrue(np.all(self.ll.classify(self.test_points)==self.test_results))

    def test_missclassify_same_line(self):
        new_line = LearningLine(0,1)
        self.assertTrue(
            np.array_equiv(self.ll.missclassify(new_line,
                                                self.test_points),
                           np.arange(4)))

    def test_missclassify_diff_line(self):
        new_line = LearningLine(0,2)
        test_point = np.array([1,1])
        self.assertFalse(
            np.array_equiv(self.ll.missclassify(new_line,
                                                self.test_points),
                           np.arange(4)))
    def test_adjust_inplace(self):
        w = self.ll.get_weight()
        self.ll.adjust_inplace(np.array([1,2,3]))
        w_adjusted = self.ll.get_weight()
        self.assertFalse(
            np.array_equiv(w,w_adjusted)
        )

    def test_adjust_not_inplace(self):
        w = self.ll.get_weight()
        new_line = self.ll.adjust(np.array([1,2,3]))
        w_same_line = self.ll.get_weight()
        w_adjusted = new_line.get_weight()

        self.assertTrue(
            np.array_equiv(w,w_same_line)
        )

        self.assertFalse(
            np.array_equiv(w_same_line,w_adjusted)
        )

    def test_missclassify_proportion(self):
        self.assertGreater(2,self.ll.missclassify_proportion(self.new_line,
                                                              self.test_points))
