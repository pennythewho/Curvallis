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

    def test__get_weighted_matrix_vector_scalar_weight(self):
        mat = np.arange(5)
        w = 2
        nptest.assert_array_equal(2*mat, fit._get_weighted_matrix(w, mat))

    def test__get_weighted_matrix_vector_vector_weight(self):
        mat = [1, 2, 3]
        w = [2, 3, 4]
        nptest.assert_array_equal([2, 6, 12], fit._get_weighted_matrix(w, mat))

    def test__get_derivative_constraints_minimize_d1_no_weight(self):
        p = 2
        knots = fit.augment_knots(p, [3, 5, 7], [0, 10])
        p2 = np.poly1d([3, -14, 8])
        min_d1_x = p2.deriv(1).roots
        X, A, d = fit._get_derivative_constraints(p, knots, minimize_d1_x=min_d1_x)
        nptest.assert_array_equal(min_d1_x, X)
        nptest.assert_array_equal(bf.get_collocation_matrix(p, knots, min_d1_x, 1), A)
        nptest.assert_array_equal(np.zeros(len(min_d1_x)), d)

    def test__get_derivative_constraints_minimize_d1_scalar_weight(self):
        p = 2
        knots = fit.augment_knots(p, [3, 5, 7], [0, 10])
        p2 = np.poly1d([3, -14, 8])
        min_d1_x = p2.deriv(1).roots
        w = 1.5
        X, A, d = fit._get_derivative_constraints(p, knots, minimize_d1_x=min_d1_x, minimize_d1_w=w)
        nptest.assert_array_equal(min_d1_x, X)
        nptest.assert_array_equal(1.5*bf.get_collocation_matrix(p, knots, min_d1_x, 1), A)
        nptest.assert_array_equal(np.zeros(len(min_d1_x)), d)

    def test__get_derivative_constraints_minimize_d1_vector_weight(self):
        p = 3
        knots = fit.augment_knots(p, [3, 5, 7], [0, 10])
        p3 = np.poly1d([3, 8, 7], r=True)
        min_d1_x = p3.deriv(1).roots
        w = [1.5, 2.5]
        X, A, d = fit._get_derivative_constraints(p, knots, minimize_d1_x=min_d1_x, minimize_d1_w=w)
        nptest.assert_array_equal(min_d1_x, X)
        colloc = bf.get_collocation_matrix(p, knots, min_d1_x, 1)
        for r in range(A.shape[0]):
            nptest.assert_array_equal(colloc[r,:]*w[r], A[r, :])
        nptest.assert_array_equal(np.zeros(len(min_d1_x)), d)

    def test__get_derivative_constraints_minimize_d2_no_weight(self):
        p = 4
        knots = fit.augment_knots(p, [3, 5, 7], [0, 10])
        p4 = np.poly1d([3, 8, 7, 4.5], r=True)
        min_d2_x = p4.deriv(2).roots
        X, A, d = fit._get_derivative_constraints(p, knots, minimize_d2_x=min_d2_x)
        nptest.assert_array_equal(min_d2_x, X)
        nptest.assert_array_equal(bf.get_collocation_matrix(p, knots, min_d2_x, 2), A)
        nptest.assert_array_equal(np.zeros(len(min_d2_x)), d)

    def test__get_derivative_constraints_minimize_d2_scalar_weight(self):
        p = 4
        knots = fit.augment_knots(p, [3, 5, 7], [0, 10])
        p4 = np.poly1d([3, 8, 7, 4.5], r=True)
        min_d2_x = p4.deriv(2).roots
        w = 1.5
        X, A, d = fit._get_derivative_constraints(p, knots, minimize_d2_x=min_d2_x, minimize_d2_w=w)
        nptest.assert_array_equal(min_d2_x, X)
        nptest.assert_array_equal(w*bf.get_collocation_matrix(p, knots, min_d2_x, 2), A)
        nptest.assert_array_equal(np.zeros(len(min_d2_x)), d)

    def test__get_derivative_constraints_minimize_d2_vector_weight(self):
        p = 4
        knots = fit.augment_knots(p, [3, 5, 7], [0, 10])
        p4 = np.poly1d([3, 8, 7, 4.5], r=True)
        min_d2_x = p4.deriv(2).roots
        w = [1.5, 2.5]
        X, A, d = fit._get_derivative_constraints(p, knots, minimize_d2_x=min_d2_x, minimize_d2_w=w)
        nptest.assert_array_equal(min_d2_x, X)
        colloc = bf.get_collocation_matrix(p, knots, min_d2_x, 2)
        for r in range(A.shape[0]):
            nptest.assert_array_equal(colloc[r, :] * w[r], A[r, :])
        nptest.assert_array_equal(np.zeros(len(min_d2_x)), d)

    def test__get_derivative_constraints_minimize_d1d2_one_weighted(self):
        p = 4
        knots = fit.augment_knots(p, [3, 5, 7], [0, 10])
        p4 = np.poly1d([3, 8, 7, 4.5], r=True)
        min_d1_x = p4.deriv(1).roots
        min_d2_x = p4.deriv(2).roots
        w2 = [1.5, 2.5]
        X, A, d = fit._get_derivative_constraints(p, knots, minimize_d1_x=min_d1_x, minimize_d2_x=min_d2_x, minimize_d2_w=w2)
        nptest.assert_array_equal(np.concatenate((min_d1_x, min_d2_x)), X)
        colloc1 = bf.get_collocation_matrix(p, knots, min_d1_x, 1)
        colloc2 = fit._get_weighted_matrix(w2, bf.get_collocation_matrix(p, knots, min_d2_x, 2))
        nptest.assert_array_equal(np.vstack((colloc1, colloc2)), A)
        nptest.assert_array_equal(np.zeros(len(min_d1_x)+len(min_d2_x)), d)

    def test__get_derivative_constraints_minimize_d1d2_scalar_weights(self):
        p = 4
        knots = fit.augment_knots(p, [3, 5, 7], [0, 10])
        p4 = np.poly1d([3, 8, 7, 4.5], r=True)
        min_d1_x = p4.deriv(1).roots
        min_d2_x = p4.deriv(2).roots
        w1 = 1.5
        w2 = 2.5
        X, A, d = fit._get_derivative_constraints(p, knots, minimize_d1_x=min_d1_x, minimize_d1_w=w1, minimize_d2_x=min_d2_x, minimize_d2_w=w2)
        nptest.assert_array_equal(np.concatenate((min_d1_x, min_d2_x)), X)
        colloc1 = w1*bf.get_collocation_matrix(p, knots, min_d1_x, 1)
        colloc2 = w2*bf.get_collocation_matrix(p, knots, min_d2_x, 2)
        nptest.assert_array_equal(np.vstack((colloc1, colloc2)), A)
        nptest.assert_array_equal(np.zeros(len(min_d1_x)+len(min_d2_x)), d)

    def test__get_derivative_constraints_ignore_minimize_high_derivative(self):
        p = 2
        knots = [0, 0, 0, 3, 5, 7, 10, 10, 10]
        X, A, d = fit._get_derivative_constraints(p, knots, minimize_d3_x=[1,2])
        self.assertTrue(X.size == 0)
        self.assertTrue(A.size == 0)
        self.assertTrue(d.size == 0)

    def test__get_derivative_constraints_ignore_weight_if_minimize_missing(self):
        p = 4
        knots = fit.augment_knots(p, [3, 5, 7], [0, 10])
        p4 = np.poly1d([3, 8, 7, 4.5], r=True)
        min_d2_x = p4.deriv(2).roots
        w2 = 2.5
        X, A, d = fit._get_derivative_constraints(p, knots, minimize_d1_w=1.5, minimize_d2_x=min_d2_x, minimize_d2_w=w2)
        nptest.assert_array_equal(min_d2_x, X)
        nptest.assert_array_equal(w2 * bf.get_collocation_matrix(p, knots, min_d2_x, 2), A)
        nptest.assert_array_equal(np.zeros(len(min_d2_x)), d)

    def test__get_derivative_constraints_minimize_d2_ignore_set_weight(self):
        p = 4
        knots = fit.augment_knots(p, [3, 5, 7], [0, 10])
        p4 = np.poly1d([3, 8, 7, 4.5], r=True)
        min_d2_x = p4.deriv(2).roots
        X, A, d = fit._get_derivative_constraints(p, knots, minimize_d2_x=min_d2_x, set_d2_w=2)
        nptest.assert_array_equal(min_d2_x, X)
        nptest.assert_array_equal(bf.get_collocation_matrix(p, knots, min_d2_x, 2), A)
        nptest.assert_array_equal(np.zeros(len(min_d2_x)), d)

    def test__get_derivative_constraints_set_d1_unweighted(self):
        p = 4
        knots = fit.augment_knots(p, [3, 5, 7], [0, 10])
        p4 = np.poly1d([3, 8, 7, 4.5], r=True)
        x = [1,2,3]
        dy = p4.deriv(1)(x)
        set_d1_x = [(a,b) for a,b in zip(x, dy)]
        X, A, d = fit._get_derivative_constraints(p, knots, set_d1_x=set_d1_x)
        nptest.assert_array_equal(x, X)
        nptest.assert_array_equal(bf.get_collocation_matrix(p, knots, x, 1), A)
        nptest.assert_array_equal(dy, d)

    def test__get_derivative_constraints_set_d1_scalar_weight(self):
        p = 4
        knots = fit.augment_knots(p, [3, 5, 7], [0, 10])
        p4 = np.poly1d([3, 8, 7, 4.5], r=True)
        x = [1, 2, 3]
        dy = p4.deriv(1)(x)
        set_d1_x = [(a, b) for a, b in zip(x, dy)]
        w = 2
        X, A, d = fit._get_derivative_constraints(p, knots, set_d1_x=set_d1_x, set_d1_w=w)
        nptest.assert_array_equal(x, X)
        nptest.assert_array_equal(w*bf.get_collocation_matrix(p, knots, x, 1), A)
        nptest.assert_array_equal(w*dy, d)

    def test__get_derivative_constraints_set_d1_vector_weight(self):
        p = 3
        knots = fit.augment_knots(p, [3, 5, 7], [0, 10])
        p3 = np.poly1d([3, 8, 7], r=True)
        d1_x = [1, 2]
        d1_y = p3.deriv(1)(d1_x)
        set_d1_x = [(x, dy) for x, dy in zip(d1_x, d1_y)]
        w = [1.5, 2.5]
        X, A, d = fit._get_derivative_constraints(p, knots, set_d1_x=set_d1_x, set_d1_w=w)
        nptest.assert_array_equal(d1_x, X)
        colloc = bf.get_collocation_matrix(p, knots, d1_x, 1)
        for r in range(A.shape[0]):
            nptest.assert_array_equal(colloc[r, :] * w[r], A[r, :])
            self.assertEqual(d1_y[r]*w[r], d[r])

    def test__get_derivative_constraints_set_d2_ignore_minimize_weight(self):
        p = 3
        knots = fit.augment_knots(p, [3, 5, 7], [0, 10])
        p3 = np.poly1d([3, 8, 7], r=True)
        d2_x = [1, 2]
        d2_y = p3.deriv(2)(d2_x)
        set_d2_x = [(x,dy) for x, dy in zip(d2_x, d2_y)]
        X, A, d = fit._get_derivative_constraints(p, knots, set_d2_x=set_d2_x, minimize_d2_w=2)
        nptest.assert_array_equal(d2_x, X)
        nptest.assert_array_equal(bf.get_collocation_matrix(p, knots, d2_x, 2), A)
        nptest.assert_array_equal(d2_y, d)

    def test__get_derivative_constraints_set_ignore_high_derivative(self):
        p = 3
        knots = fit.augment_knots(p, [3, 5, 7], [0, 10])
        X, A, d = fit._get_derivative_constraints(p, knots, set_d4_x=[(1,2),(3,4)])
        self.assertTrue(X.size == 0)
        self.assertTrue(A.size == 0)
        self.assertTrue(d.size == 0)

    def test__get_derivative_constraints_ignore_weight_if_set_missing(self):
        p = 3
        knots = fit.augment_knots(p, [3, 5, 7], [0, 10])
        X, A, d = fit._get_derivative_constraints(p, knots, set_d1_w=[2, 3])
        self.assertTrue(X.size == 0)
        self.assertTrue(A.size == 0)
        self.assertTrue(d.size == 0)

    def test__get_derivative_constraints_set_d1_ignore_minimize_weight(self):
        p = 4
        knots = fit.augment_knots(p, [3, 5, 7], [0, 10])
        p4 = np.poly1d([3, 8, 7, 4.5], r=True)
        x = [1, 2, 3]
        dy = p4.deriv(1)(x)
        set_d1_x = [(a, b) for a, b in zip(x, dy)]
        X, A, d = fit._get_derivative_constraints(p, knots, set_d1_x=set_d1_x, minimize_d1_w=2)
        nptest.assert_array_equal(x, X)
        nptest.assert_array_equal(bf.get_collocation_matrix(p, knots, x, 1), A)
        nptest.assert_array_equal(dy, d)

    def test__get_derivative_constraints_ignore_set_high_derivative(self):
        p = 2
        knots = [0, 0, 0, 3, 5, 7, 10, 10, 10]
        X, A, d = fit._get_derivative_constraints(p, knots, set_d3_x=[1, 2])
        self.assertTrue(X.size == 0)
        self.assertTrue(A.size == 0)
        self.assertTrue(d.size == 0)

    def test_get_bspline_fit_quadratic_no_weight_no_regularization_no_noise(self):
        p = 2
        x = np.linspace(0, 10, 21)
        knots = fit.augment_knots(p, [2, 4, 6, 8], x)
        expected_coefs = [1, 4, 1, 5, 9, 2, 6]
        A = bf.get_collocation_matrix(p, knots, x)
        y = A @ expected_coefs
        bsp = fit.get_bspline_fit(p, knots, x, y)
        self.assertEqual(p, bsp._degree)
        nptest.assert_array_equal(knots, bsp._knots)
        self.assertEqual(len(expected_coefs), bsp._coefs.size)
        nptest.assert_array_almost_equal(expected_coefs, bsp._coefs)

    def test_get_bspline_fit_cubic_no_weight_no_regularization_no_noise(self):
        p = 3
        x = np.linspace(0, 10, 21)
        knots = fit.augment_knots(p, [2, 4, 6, 8], x)
        expected_coefs = [1, 4, 1, 5, 9, 2, 6, 5]
        A = bf.get_collocation_matrix(p, knots, x)
        y = A @ expected_coefs
        bsp = fit.get_bspline_fit(p, knots, x, y)
        self.assertEqual(p, bsp._degree)
        nptest.assert_array_equal(knots, bsp._knots)
        self.assertEqual(len(expected_coefs), bsp._coefs.size)
        nptest.assert_array_almost_equal(expected_coefs, bsp._coefs)

    def test_get_bspline_fit_quadratic_no_weight_no_regularization_noisy(self):
        p = 2
        x = np.linspace(0, 10, 21)
        knots = fit.augment_knots(p, [2, 4, 6, 8], x)
        expected_coefs = [1, 4, 1, 5, 9, 2, 6]
        A = bf.get_collocation_matrix(p, knots, x)
        y = self._get_noisy(A @ expected_coefs)
        bsp = fit.get_bspline_fit(p, knots, x, y)
        nptest.assert_array_almost_equal(expected_coefs, bsp._coefs, decimal=1)

    def test_get_bspline_fit_scalar_weight_no_regularization_noisy(self):
        p = 2
        x = np.linspace(0, 10, 21)
        knots = fit.augment_knots(p, [2, 4, 6, 8], x)
        expected_coefs = [1, 4, 1, 5, 9, 2, 6]
        A = bf.get_collocation_matrix(p, knots, x)
        y = self._get_noisy(A @ expected_coefs)
        bsp = fit.get_bspline_fit(p, knots, x, y, w=3)
        nptest.assert_array_almost_equal(expected_coefs, bsp._coefs, decimal=1)

    def test_get_bspline_fit_no_weight_ignore_regularization_with_high_derivative(self):
        p = 2
        x = np.linspace(0, 10, 21)
        knots = fit.augment_knots(p, [2, 4, 6, 8], x)
        expected_coefs = [1, 4, 1, 5, 9, 2, 6]
        A = bf.get_collocation_matrix(p, knots, x)
        y = self._get_noisy(A @ expected_coefs)
        bsp = fit.get_bspline_fit(p, knots, x, y)
        bspir = fit.get_bspline_fit(p, knots, x, y, minimize_d3_x=np.linspace(4, 6, 21))
        nptest.assert_array_almost_equal(expected_coefs, bsp._coefs, decimal=1)
        nptest.assert_array_equal(bsp._coefs, bspir._coefs)

    def test_get_bspline_fit_no_weight_d1_regularization(self):
        p = 2
        x = np.linspace(0, 10, 21)
        knots = fit.augment_knots(p, [3, 5, 7], x)
        p2 = np.poly1d([3, -14, 8])
        ideal_coefs = np.array([8, -13, -3, 29, 99, 168])     # fit.get_bspline_fit(p, knots, x, Y)
        noisy_coefs = np.array([7.8, -13.3, -2.7, 28.8, 98.5, 168.4])
        y = bf.get_collocation_matrix(p, knots, x) @ noisy_coefs
        bsp = fit.get_bspline_fit(p, knots, x, y)
        bspr = fit.get_bspline_fit(p, knots, x, y, minimize_d1_x=p2.deriv(1).roots)
        bspwr = fit.get_bspline_fit(p, knots, x, y, minimize_d1_x=p2.deriv(1).roots, minimize_d1_w=1.5)
        self.assertRaises(AssertionError, nptest.assert_array_equal, bsp._coefs, bspr._coefs)
        self.assertRaises(AssertionError, nptest.assert_array_equal, bsp._coefs, bspwr._coefs)
        self.assertRaises(AssertionError, nptest.assert_array_equal, bspr._coefs, bspwr._coefs)

    def test_get_bspline_fit_set_d1_scalar_weight(self):
        p = 2
        x = np.linspace(0, 10, 21)
        knots = fit.augment_knots(p, [3, 5, 7], x)
        p2 = np.poly1d([3, -14, 8])
        ideal_coefs = np.array([8, -13, -3, 29, 99, 168])  # fit.get_bspline_fit(p, knots, x, Y)
        noisy_coefs = np.array([7.8, -13.3, -2.7, 28.8, 98.5, 168.4])
        y = bf.get_collocation_matrix(p, knots, x) @ noisy_coefs
        d1_x = [1, 2, 3]
        d1_y = p2.deriv(1)(d1_x)
        set_d1_x = [(x, dy) for x, dy in zip(d1_x, d1_y)]
        d1_w = 2
        bsp = fit.get_bspline_fit(p, knots, x, y)
        bsps = fit.get_bspline_fit(p, knots, x, y, set_d1_x=set_d1_x)
        bspws = fit.get_bspline_fit(p, knots, x, y, set_d1_x=set_d1_x, set_d1_w=d1_w)
        self.assertRaises(AssertionError, nptest.assert_array_equal, bsp._coefs, bsps._coefs)
        self.assertRaises(AssertionError, nptest.assert_array_equal, bsp._coefs, bspws._coefs)
        self.assertRaises(AssertionError, nptest.assert_array_equal, bsps._coefs, bspws._coefs)

    def test_get_bspline_fit_require_at_least_one_data_point(self):
        p = 3
        knots = fit.augment_knots(p, [3, 5, 7], [0, 10])
        p3 = np.poly1d([3, 4, 5], r=True)
        self.assertRaisesRegex(ValueError, 'at least one', fit.get_bspline_fit, p=p, knots=knots, x=[], y=[], minimize_d1_x=p3.deriv(1).roots)
























