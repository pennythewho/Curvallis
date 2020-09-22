import unittest as ut
import numpy as np
from numpy.random import random_sample
from numpy import testing as nptest
from .. import fit, basis_function as bf

class TestFit(ut.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def _get_noisy(self, y, d=2):
        """ adds random noise at decimal corresponding to d less than the order of magnitude of std(abs(y))
        e.g. if std(abs(y)) ~ .01 and d=1, random element will be on order of .001
        """
        return y + (random_sample(len(y)) - 0.5) * 10 ** (np.floor(np.log10(np.std(abs(y)))) - d)

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

    def test__get_weighted_matrix_weight_1_scalar(self):
        mat = np.array([[3, 5, 7], [2, 4, 6], [11, 13, 17]])
        w = 1
        nptest.assert_array_equal(mat, fit._get_weighted_matrix(w, mat))

    def test__get_weighted_matrix_scalar_weight(self):
        mat = np.array([[3, 5, 7], [2, 4, 6], [11, 13, 17]])
        w = 3
        expected = np.array([[9, 15, 21], [6, 12, 18], [33, 39, 51]])
        nptest.assert_array_equal(expected, fit._get_weighted_matrix(w, mat))

    def test__get_weighted_matrix_weight_1_vector(self):
        mat = np.array([[3, 5, 7], [2, 4, 6], [11, 13, 17]])
        w = [1, 1, 1]
        nptest.assert_array_equal(mat, fit._get_weighted_matrix(w, mat))

    def test__get_weighted_matrix_single_weight_vector(self):
        mat = np.array([[3, 5, 7], [2, 4, 6], [11, 13, 17]])
        w = [3, 3, 3]
        expected = np.array([[9, 15, 21], [6, 12, 18], [33, 39, 51]])
        nptest.assert_array_equal(expected, fit._get_weighted_matrix(w, mat))

    def test__get_weighted_matrix_varied_weights(self):
        mat = np.array([[3, 5, 7], [2, 4, 6], [11, 13, 17]])
        w = [5, 4, 3]
        expected = np.array([[15, 25, 35], [8, 16, 24], [33, 39, 51]])
        nptest.assert_array_equal(expected, fit._get_weighted_matrix(w, mat))

    def test__get_weighted_matrix_too_few_weights(self):
        mat = np.array([[3, 5, 7], [2, 4, 6], [11, 13, 17]])
        w = [5, 4]
        self.assertRaisesRegex(ValueError, 'Wrong number of weights', fit._get_weighted_matrix, wt=w, mat=mat)

    def test_get_spline_quadratic_no_weight_no_regularization_no_noise(self):
        p = 2
        x = np.linspace(0, 10, 21)
        knots = fit.augment_knots(p, [2, 4, 6, 8], x)
        expected_coefs = [1, 4, 1, 5, 9, 2, 6]
        A = bf.get_collocation_matrix(p, knots, x)
        y = A @ expected_coefs
        bsp = fit.get_spline(p, knots, x, y)
        self.assertEqual(p, bsp._degree)
        nptest.assert_array_equal(knots, bsp._knots)
        self.assertEqual(len(expected_coefs), bsp._coefs.size)
        nptest.assert_array_almost_equal(expected_coefs, bsp._coefs)

    def test_get_spline_cubic_no_weight_no_regularization_no_noise(self):
        p = 3
        x = np.linspace(0, 10, 21)
        knots = fit.augment_knots(p, [2, 4, 6, 8], x)
        expected_coefs = [1, 4, 1, 5, 9, 2, 6, 5]
        A = bf.get_collocation_matrix(p, knots, x)
        y = A @ expected_coefs
        bsp = fit.get_spline(p, knots, x, y)
        self.assertEqual(p, bsp._degree)
        nptest.assert_array_equal(knots, bsp._knots)
        self.assertEqual(len(expected_coefs), bsp._coefs.size)
        nptest.assert_array_almost_equal(expected_coefs, bsp._coefs)

    def test_get_spline_quadratic_no_weight_no_regularization_noisy(self):
        p = 2
        x = np.linspace(0, 10, 21)
        knots = fit.augment_knots(p, [2, 4, 6, 8], x)
        expected_coefs = [1, 4, 1, 5, 9, 2, 6]
        A = bf.get_collocation_matrix(p, knots, x)
        y = self._get_noisy(A @ expected_coefs)
        bsp = fit.get_spline(p, knots, x, y)
        nptest.assert_array_almost_equal(expected_coefs, bsp._coefs, decimal=1)

    def test_get_spline_scalar_weight_no_regularization_noisy(self):
        p = 2
        x = np.linspace(0, 10, 21)
        knots = fit.augment_knots(p, [2, 4, 6, 8], x)
        expected_coefs = [1, 4, 1, 5, 9, 2, 6]
        A = bf.get_collocation_matrix(p, knots, x)
        y = self._get_noisy(A @ expected_coefs)
        bsp = fit.get_spline(p, knots, x, y, w=3)
        nptest.assert_array_almost_equal(expected_coefs, bsp._coefs, decimal=1)

    def test_get_spline_no_weight_ignore_regularization_with_high_derivative(self):
        p = 2
        x = np.linspace(0, 10, 21)
        knots = fit.augment_knots(p, [2, 4, 6, 8], x)
        expected_coefs = [1, 4, 1, 5, 9, 2, 6]
        A = bf.get_collocation_matrix(p, knots, x)
        y = self._get_noisy(A @ expected_coefs)
        bsp = fit.get_spline(p, knots, x, y)
        bspir = fit.get_spline(p, knots, x, y, minimize_d3_x=np.linspace(4, 6, 21))
        nptest.assert_array_almost_equal(expected_coefs, bsp._coefs, decimal=1)
        nptest.assert_array_equal(bsp._coefs, bspir._coefs)

    def test_get_spline_no_weight_d1_regularization(self):
        p = 2
        x = np.linspace(0, 10, 21)
        knots = fit.augment_knots(p, [3, 5, 7], x)
        p2 = np.poly1d([3, -14, 8])
        ideal_coefs = np.array([8, -13, -3, 29, 99, 168])     # fit.get_spline(p, knots, x, Y)
        noisy_coefs = np.array([7.8, -13.3, -2.7, 28.8, 98.5, 168.4])
        y = bf.get_collocation_matrix(p, knots, x) @ noisy_coefs
        bspr = fit.get_spline(p, knots, x, y, minimize_d1_x=p2.deriv(1).roots)  # noisy, with valid regularization
        self.assertRaises(AssertionError, nptest.assert_array_almost_equal, noisy_coefs, bspr._coefs, decimal=2)
        # overall, coefs with regularization should be closer to ideal
        self.assertLess(sum(abs(bspr._coefs - ideal_coefs)), sum(abs(noisy_coefs - ideal_coefs)))

    def test_get_spline_no_weight_d2_regularization(self):
        p = 3
        x = np.linspace(0, 10, 21)
        knots = fit.augment_knots(p, [3, 5, 7], x)
        p3 = np.poly1d([3, -14, 8, 7])
        Y = p3(x)
        ideal_coefs = np.array([7., 15., -125/3, 92/3, 1177/3, 1059., 1687.])   # fit.get_spline(p, knots, x, Y)
        noisy_coefs = np.array([6.7, 15.1, -41.8, 30.4, 392.1, 1059., 1687.1])
        y = bf.get_collocation_matrix(p, knots, x) @ noisy_coefs
        bspr = fit.get_spline(p, knots, x, y, minimize_d2_x=[14/9])             # actual root is 14/9
        self.assertRaises(AssertionError, nptest.assert_array_almost_equal, noisy_coefs, bspr._coefs, decimal=2)
        self.assertLess(sum(abs(bspr._coefs - ideal_coefs)), sum(abs(noisy_coefs - ideal_coefs)))

    def test_get_spline_no_weight_d1d2_regularization(self):
        p = 3
        x = np.linspace(0, 10, 21)
        knots = fit.augment_knots(p, [3, 5, 7], x)
        p3 = np.poly1d([3, -14, 8, 7])
        Y = p3(x)
        ideal_coefs = np.array([7., 15., -125 / 3, 92 / 3, 1177 / 3, 1059., 1687.])  # fit.get_spline(p, knots, x, Y)
        noisy_coefs = np.array([6.8, 15.1, -41.8, 30.4, 392.1, 1058.9, 1687.1])
        y = bf.get_collocation_matrix(p, knots, x) @ noisy_coefs
        bsp1r = fit.get_spline(p, knots, x, y, minimize_d1_x=p3.deriv(1).roots)
        bsp2r = fit.get_spline(p, knots, x, y, minimize_d1_x=p3.deriv(1).roots, minimize_d2_x=p3.deriv(2).roots)
        self.assertRaises(AssertionError, nptest.assert_array_almost_equal, bsp1r._coefs, bsp2r._coefs, decimal=3)
        self.assertLess(sum(abs(bsp2r._coefs - ideal_coefs)), sum(abs(noisy_coefs - ideal_coefs)))
        self.assertLess(sum(abs(bsp2r(x) - Y)), sum(abs(bf.get_collocation_matrix(p, knots, x)@noisy_coefs - Y)))

    def test_get_spline_no_weight_scalar_weighted_d1d2_regularization(self):
        p = 3
        x = np.linspace(0, 10, 21)
        knots = fit.augment_knots(p, [3, 5, 7], x)
        p3 = np.poly1d([3, -14, 8, 7])
        Y = p3(x)
        ideal_coefs = np.array([7., 15., -125 / 3, 92 / 3, 1177 / 3, 1059., 1687.])  # fit.get_spline(p, knots, x, Y)
        noisy_coefs = np.array([6.8, 15.1, -41.8, 30.4, 392.1, 1058.9, 1687.1])
        y = bf.get_collocation_matrix(p, knots, x) @ noisy_coefs
        bspr = fit.get_spline(p, knots, x, y, minimize_d1_x=p3.deriv(1).roots, minimize_d2_x=p3.deriv(2).roots)
        bsprw = fit.get_spline(p, knots, x, y, minimize_d1_x=p3.deriv(1).roots, minimize_d1_w=2,
                              minimize_d2_x=p3.deriv(2).roots, minimize_d2_w=1.5)
        self.assertRaises(AssertionError, nptest.assert_array_almost_equal, bsprw._coefs, bspr._coefs, decimal=2)
        # weighting exact data in regularization should improve fit over noisy coefs
        self.assertLess(sum(abs(bsprw(x) - Y)), sum(abs(bspr(x) - Y)))





















