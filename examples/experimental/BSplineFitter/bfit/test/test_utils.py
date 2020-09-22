import unittest as ut
import numpy as np
from numpy import testing as nptest
from .. import utils

class TestUtils(ut.TestCase):
    def setUp(self):
        self.m = 10
        self.knots = np.linspace(0, 1, self.m + 1)
    def tearDown(self):
        pass

    def test_is_nondecreasing_increasing(self):
        v = np.arange(4)
        self.assertTrue(utils.is_nondecreasing(v))

    def test_is_nondecreasing_allsame(self):
        v = np.repeat(4, 4)
        self.assertTrue(utils.is_nondecreasing(v))

    def test_is_nondecreasing_nondecreasing(self):
        v = np.array([0,0,1,2,2,3,3])
        self.assertTrue(utils.is_nondecreasing(v))

    def test_is_nondecreasing_onestepdown(self):
        v = np.array([0,0,1,2,np.nextafter(2,1),3,3])
        self.assertFalse(utils.is_nondecreasing(v))

    def test_validate_knots_nondecreasing(self):
        knots = np.array([0,0,1,2,2,3,3])
        utils.validate_knots(knots)

    def test_validate_knots_empty_spans(self):
        knots = np.repeat(4, 4)
        self.assertRaisesRegex(ValueError, "at least one knot span with length", utils.validate_knots, knots=knots)

    def test_validate_knots_decreasing(self):
        knots = np.array([0,0,1,2,np.nextafter(2,1),3,3])
        self.assertRaisesRegex(ValueError, "not non-decreasing", utils.validate_knots, knots=knots)

    def test_get_multiplicity_simple(self):
        knots = np.arange(4)
        self.assertEqual(1, utils.get_multiplicity(knots, 0))
        self.assertEqual(1, utils.get_multiplicity(knots, 1))
        self.assertEqual(1, utils.get_multiplicity(knots, 2))
        self.assertEqual(1, utils.get_multiplicity(knots, 3))

    def test_get_multiplicity_multiplefirstknot(self):
        knots = np.array([1,1,2,3,4])
        self.assertEqual(2, utils.get_multiplicity(knots, 1))
        self.assertEqual(1, utils.get_multiplicity(knots, 2))
        self.assertEqual(1, utils.get_multiplicity(knots, 3))
        self.assertEqual(1, utils.get_multiplicity(knots, 4))

    def test_get_multiplicity_multiplelastknot(self):
        knots = np.array([1,2,3,4,4])
        self.assertEqual(1, utils.get_multiplicity(knots, 1))
        self.assertEqual(1, utils.get_multiplicity(knots, 2))
        self.assertEqual(1, utils.get_multiplicity(knots, 3))
        self.assertEqual(2, utils.get_multiplicity(knots, 4))

    def test_get_multiplicity_multipleendknots(self):
        knots = np.array([1,1,2,3,4,4])
        self.assertEqual(2, utils.get_multiplicity(knots, 1))
        self.assertEqual(1, utils.get_multiplicity(knots, 2))
        self.assertEqual(1, utils.get_multiplicity(knots, 3))
        self.assertEqual(2, utils.get_multiplicity(knots, 4))

    def test_get_multiplicity_multipleinternalknots(self):
        knots = np.array([1,2,2,3,3,4])
        self.assertEqual(1, utils.get_multiplicity(knots, 1))
        self.assertEqual(2, utils.get_multiplicity(knots, 2))
        self.assertEqual(2, utils.get_multiplicity(knots, 3))
        self.assertEqual(1, utils.get_multiplicity(knots, 4))

    def test_get_multiplicity_invalidknotsequence(self):
        knots = np.array([0,1,np.nextafter(1,0),2,3])
        self.assertRaises(ValueError, utils.get_multiplicity, knots=knots, knot=1)

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
        self.assertEqual(0, utils.get_knotspan_start_idx([0, 1], .5))

    def test_get_knotspan_start_idx_um_simpleknots(self):
        u = self.knots[-1]
        self.assertEqual(self.knots.size - 2, utils.get_knotspan_start_idx(self.knots, u))

    def test_get_knotspan_start_idx_um_clampedknots(self):
        u = self.knots[-1]
        knots = np.append(self.knots, np.repeat(u, 2))
        self.assertEqual(len(knots) - 4, utils.get_knotspan_start_idx(self.knots, u))

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

    def test_is_function_nonzero_zero_length_span(self):
        p = 2
        knots = np.array([0, 0, 0, .3, .5, .5, .6, 1, 1, 1])
        self.assertFalse(utils.is_function_nonzero(p, knots, 0, 0))
        self.assertFalse(utils.is_function_nonzero(p, knots, 0, 0))

    def test_is_function_nonzero_zero_below_range(self):
        p = 2
        knots = np.array([0, 0, 0, .3, .5, .5, .6, 1, 1, 1])
        # N_4,2 should be nonzero only on [u_4, u_7)
        for iks in range(4):
            self.assertFalse(utils.is_function_nonzero(p, knots, 4, iks), 'Incorrect for iks={0}'.format(iks))

    def test_is_function_nonzero_empty_span_in_range(self):
        p = 2
        knots = np.array([0, 0, 0, .3, .5, .5, .6, 1, 1, 1])
        # N_4,2 should be nonzero on [u_4, u_7) but [u_4,u_5) is a zero-length span
        self.assertFalse(utils.is_function_nonzero(p, knots, 4, 4))

    def test_is_function_nonzero_nonzero_in_range(self):
        p = 2
        knots = np.array([0, 0, 0, .3, .5, .5, .6, 1, 1, 1])
        # N_4,2 should be nonzero on [u_4, u_7) but [u_4,u_5) is a zero-length span
        for iks in range(5,7):
            self.assertTrue(utils.is_function_nonzero(p, knots, 4, iks), 'Incorrect for iks={0}'.format(iks))

    def test_is_function_nonzero_zero_above_range(self):
        p = 2
        knots = np.array([0, 0, 0, .3, .5, .5, .6, 1, 1, 1])
        # N_2,2 should be nonzero only on [u_2,u_5)
        for iks in range(5, len(knots)):
            self.assertFalse(utils.is_function_nonzero(p, knots, 2, iks), 'Incorrect for iks={0}'.format(iks))

    def test_is_function_nonzero_lastknot_multiple(self):
        p = 2
        knots = np.array([0, 0, 0, .3, .5, .5, .6, 1, 1, 1])
        n = 6
        self.assertTrue(utils.is_function_nonzero(p, knots, n, n))
        for iks in range(n+1, len(knots)):
            self.assertFalse(utils.is_function_nonzero(p, knots, n, iks), 'Incorrect for iks={0}'.format(iks))

    def test_get_last_knotspan_last_span_empty(self):
        knots = [0,1,2,3,4,4]
        self.assertEqual(3, utils.get_last_knotspan(knots))

    def test_get_last_knotspan_onespan(self):
        knots = [0,1]
        self.assertEqual(0, utils.get_last_knotspan(knots))

    def test_get_last_knotspan_invalidknotsequence(self):
        knots = np.repeat(0,4)
        self.assertRaises(ValueError, utils.get_last_knotspan, knots=knots)

    def test_get_last_knotspan_lastspan(self):
        knots = list(range(4))
        # the last index will be 3, but that is the end of the knot span that begins at index 2
        self.assertEqual(2, utils.get_last_knotspan(knots))

    def test_is_in_knotspan_below_span(self):
        self.assertFalse(utils.is_in_knotspan(.1, (.2, .3), True))
        self.assertFalse(utils.is_in_knotspan(.1, (.2, .3), False))

    def test_is_in_knotspan_above_span(self):
        self.assertFalse(utils.is_in_knotspan(.4, (.2, .3), True))
        self.assertFalse(utils.is_in_knotspan(.4, (.2, .3), False))

    def test_is_in_knotspan_equal_to_lower(self):
        self.assertTrue(utils.is_in_knotspan(.2, (.2, .3), True))
        self.assertTrue(utils.is_in_knotspan(.2, (.2, .3), False))

    def test_is_in_knotspan_halfway(self):
        self.assertTrue(utils.is_in_knotspan(.25, (.2, .3), True))
        self.assertTrue(utils.is_in_knotspan(.25, (.2, .3), False))

    def test_is_in_knotspan_equal_to_upper_lastknot(self):
        self.assertTrue(utils.is_in_knotspan(.3, (.2, .3), True))

    def test_is_in_knotspan_equal_to_upper_notlastknot(self):
        self.assertFalse(utils.is_in_knotspan(.3, (.2, .3), False))

    def test_find_sites_in_span_inlastspan(self):
        knots = [0,0,0,.3,.5,.5,.6,1,1,1]
        sites = np.linspace(0,1,11)
        iks=6
        nptest.assert_array_equal([6,7,8,9,10], utils.find_sites_in_span(knots, 6, sites))

    def test_find_sites_in_span_emptyknotindex(self):
        knots = [0, 0, 0, .3, .5, .5, .6, 1, 1, 1]
        sites = np.linspace(0, 1, 11)
        for iks in (np.diff(knots) == 0).nonzero()[0]:
            self.assertEqual(0, utils.find_sites_in_span(knots, iks, sites).size)

    def test_find_sites_in_span_outsideknots(self):
        knots = [0, 0, 0, .3, .5, .5, .6, 1, 1, 1]
        sites = np.linspace(2, 3, 11)
        for iks in range(len(knots)):
            self.assertEqual(0, utils.find_sites_in_span(knots, iks, sites).size)

    def test_find_sites_in_span_firstknot(self):
        knots = [0, 0, 0, .3, .5, .5, .6, 1, 1, 1]
        sites = np.linspace(0, 1, 11)
        self.assertEqual(0, utils.find_sites_in_span(knots, 0, sites).size)
        self.assertEqual(0, utils.find_sites_in_span(knots, 1, sites).size)
        nptest.assert_array_equal([0,1,2], utils.find_sites_in_span(knots, 2, sites))

    def test_find_sites_in_span_sitesequaltoknots(self):
        knots = np.arange(4)
        nptest.assert_array_equal([0], utils.find_sites_in_span(knots, 0, knots))
        nptest.assert_array_equal([1], utils.find_sites_in_span(knots, 1, knots))
        nptest.assert_array_equal([2,3], utils.find_sites_in_span(knots, 2,  knots))





