import unittest as ut
import numpy as np
from numpy import testing as nptest
from .. import fit

class TestFit(ut.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_validate_knots_and_data_increasing_knots(self):
        p = 2
        knots = np.array([0, 0, 0, .3, .5, .4, .6, 1, 1, 1])
        x = np.arange(10, 11)
        self.assertRaises(ValueError, fit.validate_knots_and_data, p=p, knots=knots, x=x)

    def test_validate_knots_and_data_too_few_knots(self):
        p = 2
        knots = np.arange(2)
        x = np.arange(10,11)
        self.assertRaisesRegex(ValueError, r'There must be at least 3 knots', fit.validate_knots_and_data, p=p, knots=knots, x=x)

    def test_validate_knots_and_data_increasing_x(self):
        p = 2
        knots = np.array([0, 0, 0, .3, .5, .5, .6, 1, 1, 1])
        x = [.1, .2, .1, .3]
        self.assertRaisesRegex(ValueError, r'non-decreasing', fit.validate_knots_and_data, p=p, knots=knots, x=x)

    def test_validate_knots_and_data_low_x(self):
        p = 2
        knots = np.array([0, 0, 0, .3, .5, .5, .6, 1, 1, 1])
        x = [-1]
        self.assertRaisesRegex(ValueError, r'outside the knot', fit.validate_knots_and_data, p=p, knots=knots, x=x)

    def test_validate_knots_and_data_high_x(self):
        p = 2
        knots = np.array([0, 0, 0, .3, .5, .5, .6, 1, 1, 1])
        x = [2]
        self.assertRaisesRegex(ValueError, r'outside the knot', fit.validate_knots_and_data, p=p, knots=knots, x=x)

    def test_validate_knots_and_data_too_many_site_repeats(self):
        p = 2
        knots = np.array([0, 0, 0, .3, .5, .5, .6, 1, 1, 1])
        x = np.repeat(.5,p+1)
        self.assertRaisesRegex(ValueError, r'repeated at most', fit.validate_knots_and_data, p=p, knots=knots, x=x)

    def test_validate_knots_and_data_first_knot_low_multiplicity(self):
        p = 2
        knots = np.array([0, 0, .3, .5, .5, .6, 1, 1, 1])
        x = np.linspace(0,1,11)
        self.assertRaisesRegex(ValueError, r'first and last knots does not have multiplicity', fit.validate_knots_and_data, p=p, knots=knots, x=x)

    def test_validate_knots_and_data_first_knot_high_multiplicity(self):
        p = 2
        knots = np.array([0, 0, 0, 0, .3, .5, .5, .6, 1, 1, 1])
        x = np.linspace(0,1,11)
        self.assertRaisesRegex(ValueError, r'first and last knots does not have multiplicity', fit.validate_knots_and_data, p=p, knots=knots, x=x)

    def test_validate_knots_and_data_last_knot_low_multiplicity(self):
        p = 2
        knots = np.array([0, 0, 0, .3, .5, .5, .6, 1, 1])
        x = np.linspace(0,1,11)
        self.assertRaisesRegex(ValueError, r'first and last knots does not have multiplicity', fit.validate_knots_and_data, p=p, knots=knots, x=x)

    def test_validate_knots_and_data_last_knot_high_multiplicity(self):
        p = 2
        knots = np.array([0, 0, 0, .3, .5, .5, .6, 1, 1, 1, 1])
        x = np.linspace(0,1,11)
        self.assertRaisesRegex(ValueError, r'first and last knots does not have multiplicity', fit.validate_knots_and_data, p=p, knots=knots, x=x)

    def test_validate_knots_and_data_fail_schoenberg_whitney(self):
        p = 2
        x = [5, 6, 8, 10, 40]
        knots = [5, 5, 5, 12, 40, 40, 40]
        self.assertRaisesRegex(ValueError, r'Schoenberg-Whitney', fit.validate_knots_and_data, p=p, knots=knots, x=x)

    def test_validate_knots_and_data_knots_less_than_lensites_plus_p(self):
        p = 2
        x = [5, 6, 8, 10, 40]
        knots = [5, 5, 5, 8, 40, 40, 40]
        self.assertIsNone(fit.validate_knots_and_data(p, knots, x))

    def test_validate_knots_and_data_knots_lensites_plus_p(self):
        p = 2
        x = [5, 6, 8, 10, 40]
        knots = [5, 5, 5, 8, 12, 40, 40, 40]
        self.assertIsNone(fit.validate_knots_and_data(p, knots, x))

    def test_validate_knots_and_data_knots_more_than_lensites_plus_p(self):
        p = 2
        x = [5, 6, 8, 10, 40]
        knots = [5, 5, 5, 8, 12, 30, 40, 40, 40]
        self.assertIsNone(fit.validate_knots_and_data(p, knots, x))

    def test_get_default_interior_knots_p2(self):
        p = 2
        x = [5, 6, 8, 10, 40]
        nptest.assert_array_equal([7.0, 9.0], fit.get_default_interior_knots(p,x))

    def test_get_default_interior_knots_p3(self):
        p = 3
        x = [5, 6, 8, 10, 40, 100]
        nptest.assert_array_almost_equal([8, 58/3], fit.get_default_interior_knots(p,x))

    def test_augment_knots_bad_internal_knots(self):
        p = 2
        x = [5, 6, 8, 10, 40]
        iknots = [12]
        self.assertRaises(ValueError, fit.augment_knots, p=p, x=x, iknots=iknots)

    def test_augment_knots_interior_includes_end(self):
        p = 2
        x = [5, 6, 8, 10, 40]
        iknots = [5, 12]
        self.assertRaises(ValueError, fit.augment_knots, p=p, x=x, iknots=iknots)

    def test_augment_knots_first_last_with_correct_multiplicity(self):
        p = 2
        x = [5, 6, 8, 10, 40]
        iknots = [8, 12]
        nptest.assert_array_equal([5,5,5,8,12,40,40,40], fit.augment_knots(p, iknots, x))

    def test_get_default_knots_p2(self):
        p = 2
        x = [5, 6, 8, 10, 40]
        nptest.assert_array_equal([5,5,5,7,9,40,40,40], fit.get_default_knots(p,x))

    def test_get_default_knots_p3(self):
        p = 3
        x = [5, 6, 8, 10, 40]
        nptest.assert_array_almost_equal([5, 5, 5, 5, 8, 40, 40, 40, 40], fit.get_default_knots(p, x))












