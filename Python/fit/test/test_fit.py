import unittest as ut
import numpy as np
from numpy.random import random_sample
from numpy import testing as nptest
from .. import fit

class TestFit(ut.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_validate_knots_and_data_too_few_knots(self):
        p = 2
        knots = np.arange(2)
        x = np.arange(0, 1, 11)
        self.assertRaisesRegex(ValueError, r'There must be at least 5 knots', fit.validate_knots_and_data, p=p,
                               knots=knots, x=x)

    def test_validate_knots_and_data_decreasing_knots(self):
        p = 2
        knots = [0, 0, 0, .3, .5, .4, .6, 1, 1, 1]
        x = np.linspace(0,1, 11)
        self.assertRaises(ValueError, fit.validate_knots_and_data, p=p, knots=knots, x=x)

    def test_validate_knots_and_data_decreasing_x(self):
        p = 2
        knots = [0, 0, 0, .3, .5, .5, .6, 1, 1, 1]
        x = [0, .1, .2, .3, .5, .4, .6, .7, .8, .9, 1]
        self.assertIsNone(fit.validate_knots_and_data(p, knots, x))

    def test_validate_knots_and_data_low_x(self):
        p = 2
        knots = [0, 0, 0, .3, .5, .5, .6, 1, 1, 1]
        x = [-1]
        self.assertRaisesRegex(ValueError, r'outside the knot', fit.validate_knots_and_data, p=p, knots=knots, x=x)

    def test_validate_knots_and_data_high_x(self):
        p = 2
        knots = [0, 0, 0, .3, .5, .5, .6, 1, 1, 1]
        x = [2]
        self.assertRaisesRegex(ValueError, r'outside the knot', fit.validate_knots_and_data, p=p, knots=knots, x=x)

    def test_validate_knots_and_sites_no_data_in_nonzero_span(self):
        p = 2
        knots = [0, 0, 0, .3, .5, .5, .6, 1, 1, 1]
        x = np.linspace(.3, 1, 8)
        self.assertRaisesRegex(ValueError, r'no constraining data', fit.validate_knots_and_data, p=p, knots=knots, x=x)

    def test_validate_knots_and_sites_one_site_per_basis_function(self):
        p = 2
        knots = np.array([0, 0, 0, .3, .5, .5, .6, 1, 1, 1])
        x = [.1, .3, .6]
        self.assertIsNone(fit.validate_knots_and_data(p, knots, x))

    def test_validate_knots_and_sites_last_span_only_has_lastknot(self):
        p = 2
        knots = np.array([0, 0, 0, .3, .5, .5, .6, 1, 1, 1])
        x = [.1, .5, 1]
        self.assertIsNone(fit.validate_knots_and_data(p, knots, x))

    def test_validate_knots_and_sites_excess_multiplicity(self):
        p = 2
        knots = np.array([0, 0, 0, .3, .5, .5, .5, .5, .6, 1, 1, 1])
        x = np.linspace(0,1,11)
        self.assertRaisesRegex(ValueError, r'multiplicity greater than 3', fit.validate_knots_and_data,
                               p=p, knots=knots, x=x)

    def test_get_default_interior_knots_p2(self):
        p = 2
        x = [5, 6, 8, 10, 40]
        nptest.assert_array_equal([7.0, 9.0], fit.get_default_interior_knots(p,x))

    def test_get_default_interior_knots_p3(self):
        p = 3
        x = [5, 6, 8, 10, 40, 100]
        nptest.assert_array_almost_equal([8, 58/3], fit.get_default_interior_knots(p,x))

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

    def test_get_spline_quadratic_noisy_parabola_no_regularization(self):
        p = 2
        x = np.linspace(0, 10, 21)
        y = np.poly1d([3, 2, 1])(x) + 1e-2*random_sample(len(x))
        knots = [0, 0, 0, 2.5, 5, 7.5, 10, 10, 10]
        bsp = fit.get_spline(p, knots, x, y)
        nptest.assert_array_equal(knots, bsp._knots)
        nptest.assert_array_equal(p, bsp._degree)
        self.assertEqual(6, len(bsp._coefs))
        nptest.assert_array_almost_equal([1.003, 3.506, 46.006, 126.007, 243.506, 321.002], bsp._coefs, decimal=2)

    def test_get_spline_quadratic_ignores_regularization_above_p(self):
        p = 2
        x = np.linspace(0, 10, 21)
        y = np.poly1d([3, 2, 1])(x) + 1e-2 * random_sample(len(x))
        knots = [0, 0, 0, 2.5, 5, 7.5, 10, 10, 10]
        bsp = fit.get_spline(p, knots, x, y)
        bspd = fit.get_spline(p, knots, x, y, minimize_d3_x=[5,6])
        nptest.assert_array_equal(bsp._coefs, bspd._coefs)

    def test_get_spline_quadratic_with_d1_regularization(self):
        p = 2
        x = [.197, 1, 3, 7, 20, 27, 39]
        y = [.5, .59, .73, 1.1, 2.1, 2.2, 1.7]
        min_d1_x = [.197, 27]
        knots = fit.augment_knots(2, [.5, 1, 5, 10, 30], x)
        bsp = fit.get_spline(p, knots, x, y)
        bspr = fit.get_spline(p, knots, x, y, minimize_d1_x=min_d1_x)
        nptest.assert_array_equal(knots, bsp._knots)
        nptest.assert_array_equal(p, bsp._degree)
        self.assertEqual(8, bsp._coefs.size)
        self.assertEqual(8, bspr._coefs.size)
        self.assertRaises(AssertionError, nptest.assert_array_almost_equal, bsp._coefs, bspr._coefs)

    def test_get_spline_cubic_with_d1d2_regularization(self):
        p = 3
        x = [.197, 1, 3, 7, 20, 27, 39, 197]
        y = [.5, .59, .73, 1.1, 2.1, 2.2, 1.7, 1.4]
        min_d1_x = [.197, 27, 197]
        min_d2_x = np.linspace(40, 197)
        int_knots = [.5, 1, 5, 10, 30]
        knots1 = fit.augment_knots(2, int_knots, x)
        bsp = fit.get_spline(p, knots1, x, y)
        bsp1r = fit.get_spline(p, knots1, x, y, minimize_d1_x=min_d1_x)
        knots2 = fit.augment_knots(2, np.concatenate((int_knots, np.linspace(40, 195, 20))), x)
        bsp2r = fit.get_spline(p, knots2, x, y, minimize_d1_x=min_d1_x, minimize_d2_x=min_d2_x)
        nptest.assert_array_equal(knots2, bsp2r._knots)
        self.assertRaises(AssertionError, nptest.assert_array_almost_equal, bsp1r._coefs, bsp2r._coefs)
        # from matplotlib import pyplot as plt
        # plt.plot(x, y, '*',label='data')
        # pltx = np.linspace(x[0],x[-1],2000)
        # plt.plot(pltx,bsp(pltx),label='fit - no regularization')
        # plt.plot(pltx,bsp1r(pltx),label='fit with d1 regularization')
        # plt.plot(pltx,bsp2r(pltx),label='fit with d1 and d2 regularization')
        # plt.legend()
        # plt.show(block=True)
















