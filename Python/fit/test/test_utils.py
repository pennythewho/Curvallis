import unittest as ut
import numpy as np
from .. import utils

class TestFitUtils(ut.TestCase):
    def setUp(self):
        self.m = 10
        self.knots = np.linspace(0, 1, self.m + 1)
    def tearDown(self):
        pass

    def test_get_num_ctrlpts(self):
        for p in range(int(self.m/2)):
            n = self.m - p - 1
            self.assertEqual(n + 1, utils.get_num_ctrlpts(p, self.knots))

    def test_get_knotspan_start_idx_u0_simpleknots(self):
        u = self.knots[0]
        self.assertEqual(0, utils.get_knotspan_start_idx(self.knots, u))

    def test_get_knotspan_start_idx_u0_clampedknots(self):
        u = self.knots[0]
        knots = np.append(np.repeat(u, 2), self.knots)
        self.assertEqual(2, utils.get_knotspan_start_idx(knots, u))

    def test_get_knotspan_start_idx_below(self):
        u = self.knots[0] - 1
        self.assertEqual(-1, utils.get_knotspan_start_idx(self.knots, u))

    def test_get_knotspan_start_idx_above(self):
        u = self.knots[-1] + 1
        self.assertEqual(-1, utils.get_knotspan_start_idx(self.knots, u))

    def test_get_knotspan_start_idx_single_knot(self):
        self.assertEqual(-1, utils.get_knotspan_start_idx([1], 1))

    def test_get_knotspan_start_idx_twoknots(self):
        self.assertEqual(0, utils.get_knotspan_start_idx([0,1], .5))

    def test_get_knotspan_start_idx_um_simpleknots(self):
        u = self.knots[-1]
        self.assertEqual(self.knots.size - 2, utils.get_knotspan_start_idx(self.knots, u))

    def test_get_knotspan_start_idx_um_clampedknots(self):
        u = self.knots[-1]
        knots = np.append(self.knots, np.repeat(u, 2))
        self.assertEqual(len(knots)-4, utils.get_knotspan_start_idx(self.knots, u))

    def test_get_knotspan_start_idx_inneru_simpleknots(self):
        u = (self.knots[3]+self.knots[4])/2
        self.assertEqual(3, utils.get_knotspan_start_idx(self.knots, u))

    def test_get_knotspan_start_idx_inneru_multipleknots(self):
        u = (self.knots[3] + self.knots[4]) / 2
        knots = np.insert(self.knots, 3, self.knots[3])
        self.assertEqual(4, utils.get_knotspan_start_idx(knots, u))

    def test_is_clamped_end_knot_simple_first(self):
        self.assertFalse(utils.is_clamped_end_knot(2, self.knots, self.knots[0]))

    def test_is_clamped_end_knot_unclamped_first(self):
        p=2
        knots = np.insert(self.knots, 0, np.repeat(self.knots[0], p-1))
        self.assertFalse(utils.is_clamped_end_knot(p, knots, knots[0]))

    def test_is_clamped_end_knot_clamped_first(self):
        p=2
        knots = np.insert(self.knots, 0, np.repeat(self.knots[0], p))
        self.assertTrue(utils.is_clamped_end_knot(p, knots, knots[0]))

    def test_is_clamped_end_knot_simple_internal(self):
        self.assertFalse(utils.is_clamped_end_knot(2, self.knots, self.knots[1]))

    def test_is_clamped_end_knot_unclamped_internal(self):
        p=2
        ki = 1
        knots = np.insert(self.knots, ki, np.repeat(self.knots[ki], p-1))
        self.assertFalse(utils.is_clamped_end_knot(p, knots, self.knots[ki]))

    def test_is_clamped_end_knot_clamped_internal(self):
        p=2
        ki = 1
        knots = np.insert(self.knots, ki, np.repeat(self.knots[ki], p))
        self.assertFalse(utils.is_clamped_end_knot(p, knots, self.knots[ki]))

    def test_is_clamped_end_knot_simple_final(self):
        self.assertFalse(utils.is_clamped_end_knot(2, self.knots, self.knots[-1]))

    def test_is_clamped_end_knot_unclamped_final(self):
        p = 2
        knots = np.append(self.knots, np.repeat(self.knots[-1], p-1))
        self.assertFalse(utils.is_clamped_end_knot(p, knots, self.knots[-1]))

    def test_is_clamped_end_knot_clamped_final(self):
        p=2
        knots = np.append(self.knots, np.repeat(self.knots[-1], p))
        self.assertTrue(utils.is_clamped_end_knot(p, knots, self.knots[-1]))

    def test_is_clamped_end_knot_not_a_knot(self):
        u = sum(self.knots[0:2])/2
        self.assertFalse(utils.is_clamped_end_knot(2, self.knots, u))



