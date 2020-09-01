import unittest as ut
import numpy as np
import numpy.testing as nptest
from numpy.polynomial.polynomial import Polynomial as poly
from .. import basis_function as bf

class TestFitBasisFunction(ut.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_get_basis_function_icp_out_of_range(self):
        m = 2
        knots = np.arange(m+1)
        p = 2
        n = m-p-1
        self.assertRaises(ValueError, bf.get_basis_function, icp=-1, p=p, knots=knots, iks=0)
        self.assertRaises(ValueError, bf.get_basis_function, icp=n+1, p=p, knots=knots, iks=0)

    def test_get_basis_function_iks_out_of_range(self):
        m = 2
        knots = np.arange(m + 1)
        p = 2
        self.assertRaises(ValueError, bf.get_basis_function, icp=0, p=p, knots=knots, iks=-1)
        self.assertRaises(ValueError, bf.get_basis_function, icp=0, p=p, knots=knots, iks=m+1)

    def test_get_basis_function_simple_knots_p2_cp0(self):
        p = 2
        m = 4
        knots = np.arange(m+1)/m
        i = 0
        self.assertTrue(bf.get_basis_function(i, p, knots, 0).has_samecoef(poly((0, 0, 8))))
        self.assertTrue(bf.get_basis_function(i, p, knots, 1).has_samecoef(poly((-1.5, 12, -16))))
        self.assertTrue(bf.get_basis_function(i, p, knots, 2).has_samecoef(poly((4.5, -12, 8))))
        self.assertTrue(bf.get_basis_function(i, p, knots, 3).has_samecoef(poly(0)))

    def test_get_basis_function_simple_knots_p2_cp1(self):
        p = 2
        m = 4
        knots = np.arange(m + 1) / m
        i = 1
        self.assertTrue(bf.get_basis_function(i, p, knots, 0).has_samecoef(poly(0)))
        self.assertTrue(bf.get_basis_function(i, p, knots, 1).has_samecoef(poly((0.5, -4, 8))))
        self.assertTrue(bf.get_basis_function(i, p, knots, 2).has_samecoef(poly((-5.5, 20, -16))))
        self.assertTrue(bf.get_basis_function(i, p, knots, 3).has_samecoef(poly((8, -16, 8))))

    def test_get_basis_function_multiple_knots_p2_cp0(self):
        p = 2
        knots = np.append(np.append(np.repeat(0, p+1), [.3, .5, .5, .6]), np.repeat(1, p+1))
        m = len(knots) - 1
        i = 0
        self.assertTrue(bf.get_basis_function(i, p, knots, 2).has_samecoef(poly((1, -10/3))**2), 'wrong polynomial for icp=0, iks=2')
        for iks in np.append(np.arange(2), np.arange(3, m)):
            self.assertTrue(bf.get_basis_function(i, p, knots, iks).has_samecoef(poly(0)), 'wrong polynomial for icp=0, iks={0}'.format(iks))

    def test_get_basis_function_multiple_knots_p2_cp1(self):
        p = 2
        knots = np.append(np.append(np.repeat(0, p + 1), [.3, .5, .5, .6]), np.repeat(1, p + 1))
        m = len(knots) - 1
        i = 1
        self.assertTrue(bf.get_basis_function(i, p, knots, 2).has_samecoef(20/3*poly((0, 1, -8/3))), 'wrong polynomial for icp=1, iks=2')
        self.assertTrue(bf.get_basis_function(i, p, knots, 3).has_samecoef(2.5*poly((1, -2))**2), 'wrong polynomial for icp=1, iks=3')
        for iks in np.append(np.arange(2), np.arange(4, m)):
            self.assertTrue(bf.get_basis_function(i, p, knots, iks).has_samecoef(poly(0)), 'wrong polynomial for icp=1, iks={0}'.format(iks))

    def test_get_basis_function_multiple_knots_p2_cp2(self):
        p = 2
        knots = np.append(np.append(np.repeat(0, p + 1), [.3, .5, .5, .6]), np.repeat(1, p + 1))
        m = len(knots) - 1
        i = 2
        nptest.assert_almost_equal(bf.get_basis_function(i, p, knots, 2).coef, poly((0, 0, 20/3)).coef, err_msg='wrong polynomial for icp=2, iks=2')
        nptest.assert_almost_equal(bf.get_basis_function(i, p, knots, 3).coef, poly((-3.75, 25, -35)).coef, err_msg='wrong polynomial for icp=2, iks=3')
        for iks in np.append(np.arange(2), np.arange(4, m)):
            self.assertTrue(bf.get_basis_function(i, p, knots, iks).has_samecoef(poly(0)), 'wrong polynomial for icp=2, iks={0}'.format(iks))

    def test_get_basis_function_multiple_knots_p2_cp3(self):
        p = 2
        knots = np.append(np.append(np.repeat(0, p + 1), [.3, .5, .5, .6]), np.repeat(1, p + 1))
        m = len(knots) - 1
        i = 3
        nptest.assert_almost_equal(bf.get_basis_function(i, p, knots, 3).coef, (poly((-1.5, 5))**2).coef, err_msg='wrong polynomial for icp=3, iks=3')
        self.assertTrue(bf.get_basis_function(i, p, knots, 4).has_samecoef(poly(0)), 'wrong polynomial for icp=3, iks=4')  # zero-length span
        nptest.assert_almost_equal(bf.get_basis_function(i, p, knots, 5).coef, (poly((6, -10))**2).coef, err_msg='wrong polynomial for icp=3, iks=5')
        for iks in np.append(np.arange(3), np.arange(6, m)):
            self.assertTrue(bf.get_basis_function(i, p, knots, iks).has_samecoef(poly(0)), 'wrong polynomial for icp=3, iks={0}'.format(iks))

    def test_get_basis_function_multiple_knots_p2_cp4(self):
        p = 2
        knots = np.append(np.append(np.repeat(0, p + 1), [.3, .5, .5, .6]), np.repeat(1, p + 1))
        m = len(knots) - 1
        i = 4
        nptest.assert_almost_equal(bf.get_basis_function(i, p, knots, 5).coef, (20*poly((-2, 7, -6))).coef, err_msg='wrong polynomial for icp=4, iks=5')
        self.assertTrue(bf.get_basis_function(i, p, knots, 6).has_samecoef(5*poly((1, -1))**2), 'wrong polynomial for icp=4, iks=6')
        for iks in np.append(np.arange(5), np.arange(7, m)):
            self.assertTrue(bf.get_basis_function(i, p, knots, iks).has_samecoef(poly(0)), 'wrong polynomial for icp=3, iks={0}'.format(iks))

    def test_get_basis_function_multiple_knots_p2_cp5(self):
        p = 2
        knots = np.append(np.append(np.repeat(0, p + 1), [.3, .5, .5, .6]), np.repeat(1, p + 1))
        m = len(knots) - 1
        i = 5
        nptest.assert_almost_equal(bf.get_basis_function(i, p, knots, 5).coef, poly((5, -20, 20)).coef, err_msg='wrong polynomial for icp=5, iks=5')
        self.assertTrue(bf.get_basis_function(i, p, knots, 6).has_samecoef(poly((-6.25, 17.5, -11.25))), 'wrong polynomial for icp=5, iks=6')
        for iks in np.append(np.arange(5), np.arange(7, m)):
            self.assertTrue(bf.get_basis_function(i, p, knots, iks).has_samecoef(poly(0)), 'wrong polynomial for icp=3, iks={0}'.format(iks))

    def test_get_basis_function_multiple_knots_p2_cp6(self):
        p = 2
        knots = np.append(np.append(np.repeat(0, p + 1), [.3, .5, .5, .6]), np.repeat(1, p + 1))
        m = len(knots) - 1
        i = 6
        nptest.assert_almost_equal(bf.get_basis_function(i, p, knots, 6).coef, poly((2.25, -7.5, 6.25)).coef, err_msg='wrong polynomial for icp=6, iks=6')
        for iks in np.append(np.arange(6), np.arange(7, m)):
            self.assertTrue(bf.get_basis_function(i, p, knots, iks).has_samecoef(poly(0)), 'wrong polynomial for icp=3, iks={0}'.format(iks))

    def test_get_basis_function_multiple_knots_p3_zerolengthspans(self):
        p = 3
        knots = np.array([1, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5])
        m = len(knots) - 1
        n = m-p-1
        # all zero-length spans should have all basis functions equal to 0
        for j in [j for (j, (a, b)) in enumerate(zip(knots, knots[1:])) if a==b]:
            for i in range(n+1):
                self.assertTrue(bf.get_basis_function(i, p, knots, j).has_samecoef(poly(0)))

    def test_get_basis_function_multiple_knots_p3_knotspan1(self):
        p = 3
        knots = np.array([1, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5])
        n = len(knots) - 1 - p - 1
        # N_0,3
        expected = (poly((-1,1))**2)*poly((2,-1))/2 + (poly((-1,1))**2)*poly((3,-1))/4 + (poly((-1,1))**2)*poly((3,-1))/4
        actual = bf.get_basis_function(0,3,knots,1)
        self.assertTrue(actual.has_samecoef(expected), 'N_0,3\nexpected:\t{0}\nactual:\t{1}'.format(expected,actual))
        # N_1,3
        expected = (poly((-1,1))**3)/4
        actual = bf.get_basis_function(1,3,knots,1)
        self.assertTrue(actual.has_samecoef(expected), 'N_1,3\nexpected:\t{0}\nactual:\t{1}'.format(expected,actual))
        # all other basis functions are 0
        for icp in range(2,n+1):
            self.assertTrue(bf.get_basis_function(icp,3,knots,1).has_samecoef(poly(0)))

    def test_get_basis_function_multiple_knots_p3_knotspan2(self):
        p = 3
        knots = np.array([1, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5])
        n = len(knots) - 1 - p - 1
        # N_0,3
        expected = poly((-1,1))*(poly((3,-1))**2)/4 + (poly((3,-1))**2)*poly((-1,1))/4 + (poly((3,-1))**2)*poly((-2,1))/2
        actual = bf.get_basis_function(0, 3, knots, 2)
        self.assertTrue(actual.has_samecoef(expected), 'N_0,3\nexpected:\t{0}\nactual:\t{1}'.format(expected, actual))
        # N_1,3
        expected = (poly((-1,1))**2)*poly((3,-1))/4 + poly((3,-1))*poly((-2,1))*poly((-1,1))/2 + (poly((-2,1))**2)*poly((4,-1))/2
        actual = bf.get_basis_function(1, 3, knots, 2)
        self.assertTrue(actual.has_samecoef(expected), 'N_1,3\nexpected:\t{0}\nactual:\t{1}'.format(expected, actual))
        # N_2,3
        expected = (poly((-2,1))**3)/2
        actual = bf.get_basis_function(2, 3, knots, 2)
        self.assertTrue(actual.has_samecoef(expected), 'N_2,3\nexpected:\t{0}\nactual:\t{1}'.format(expected, actual))
        # all other basis function are 0
        for icp in range(3,n+1):
            self.assertTrue(bf.get_basis_function(icp,3,knots,2).has_samecoef(poly(0)))

    def test_get_basis_function_multiple_knots_p3_knotspan4(self):
        p = 3
        knots = np.array([1, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5])
        n = len(knots) - 1 - p - 1
        iks = 4
        # N_0,3
        self.assertTrue(bf.get_basis_function(0, 3, knots, iks).has_samecoef(poly(0)))
        # N_1,3
        expected = (poly((4,-1))**3)/2
        actual = bf.get_basis_function(1, 3, knots, iks)
        self.assertTrue(actual.has_samecoef(expected), 'N_1,3\nexpected:\t{0}\nactual:\t{1}'.format(expected, actual))
        # N_2,3
        expected = poly((-2,1))*(poly((4,-1))**2)/2 + 2*(poly((4,-1))**2)*poly((-3,1))
        actual = bf.get_basis_function(2, 3, knots, iks)
        self.assertTrue(actual.has_samecoef(expected), 'N_2,3\nexpected:\t{0}\nactual:\t{1}'.format(expected, actual))
        # N_3,3
        expected = 2*(poly((-3,1))**2)*poly((4,-1)) + poly((4,-1))*(poly((-3,1))**2)
        actual = bf.get_basis_function(3, 3, knots, iks)
        self.assertTrue(actual.has_samecoef(expected), 'N_3,3\nexpected:\t{0}\nactual:\t{1}'.format(expected, actual))
        # N_4,3
        expected = poly((-3,1))**3
        actual = bf.get_basis_function(4, 3, knots, iks)
        self.assertTrue(actual.has_samecoef(expected), 'N_4,3\nexpected:\t{0}\nactual:\t{1}'.format(expected, actual))
        # all other basis function are 0
        for icp in range(5, n + 1):
            self.assertTrue(bf.get_basis_function(icp, 3, knots, iks).has_samecoef(poly(0)))

    def test_get_basis_function_multiple_knots_p3_knotspan7(self):
        p = 3
        knots = np.array([1, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5])
        n = len(knots) - 1 - p - 1
        iks = 7
        # N_0,3 ... N_3,3
        for icp in range(4):
            self.assertTrue(bf.get_basis_function(icp, 3, knots, iks).has_samecoef(poly(0)))
        # N_4,3
        expected = poly((5,-1)) ** 3
        actual = bf.get_basis_function(4, 3, knots, iks)
        self.assertTrue(actual.has_samecoef(expected), 'N_4,3\nexpected:\t{0}\nactual:\t{1}'.format(expected, actual))
        # N_5,3
        expected = (poly((5, -1))**2)*poly((-4,1)) + 2*(poly((5,-1))**2)*poly((-4,1))
        actual = bf.get_basis_function(5, 3, knots, iks)
        self.assertTrue(actual.has_samecoef(expected), 'N_5,3\nexpected:\t{0}\nactual:\t{1}'.format(expected, actual))
        # N_6,3
        expected = 2*(poly((-4,1))**2)*poly((5,-1)) + poly((5,-1))*(poly((-4,1))**2)
        actual = bf.get_basis_function(6, 3, knots, iks)
        self.assertTrue(actual.has_samecoef(expected), 'N_6,3\nexpected:\t{0}\nactual:\t{1}'.format(expected, actual))
        # N_7,3
        expected = (poly((-4,1))**3)
        actual = bf.get_basis_function(7, 3, knots, iks)
        self.assertTrue(actual.has_samecoef(expected), 'N_7,3\nexpected:\t{0}\nactual:\t{1}'.format(expected, actual))
        # all the rest should be 0
        for icp in range(8,n+1):
            self.assertTrue(bf.get_basis_function(icp, 3, knots, iks).has_samecoef(poly(0)))

    def test_get_basis_functions_simple(self):
        p = 2
        m = 4
        knots = np.arange(m + 1) / m
        n = m-p-1
        bfmatrix = bf.get_basis_functions(p, knots)
        self.assertEqual((m, n+1), bfmatrix.shape, 'Matrix is wrong shape')
        # N_0,2
        self.assertTrue(bfmatrix[0,0].has_samecoef(poly((0, 0, 8))), 'Incorrect polynomial for N_0,2 on [u_0,u_1)')
        self.assertTrue(bfmatrix[1,0].has_samecoef(poly((-1.5, 12, -16))), 'Incorrect polynomial for N_0,2 on [u_1,u_2)')
        self.assertTrue(bfmatrix[2,0].has_samecoef(poly((4.5, -12, 8))), 'Incorrect polynomial for N_0,2 on [u_3,u_4)')
        self.assertTrue(bfmatrix[3,0].has_samecoef(poly(0)), 'Incorrect polynomial for N_0,2 on [u_3,u_4)')
        # N_1,2
        self.assertTrue(bfmatrix[0,1].has_samecoef(poly(0)), 'Incorrect polynomial for N_1,2 on [u_0,u_1)')
        self.assertTrue(bfmatrix[1,1].has_samecoef(poly((.5, -4, 8))), 'Incorrect polynomial for N_1,2 on [u_1,u_2)')
        self.assertTrue(bfmatrix[2,1].has_samecoef(poly((-5.5, 20, -16))), 'Incorrect polynomial for N_1,2 on [u_2,u_3)')
        self.assertTrue(bfmatrix[3,1].has_samecoef(poly((8, -16, 8))), 'Incorrect polynomial for N_1,2 on [u_3,u_4)')

    def test_get_basis_functions_multipleknots(self):
        p = 2
        knots = np.append(np.append(np.repeat(0, p + 1), [.3, .5, .5, .6]), np.repeat(1, p + 1))
        m = len(knots) - 1
        n = m-p-1
        bfmatrix = bf.get_basis_functions(p, knots)
        self.assertEqual((m, n+1), bfmatrix.shape, 'Matrix is wrong shape')
        # N_0,2
        self.assertTrue(bfmatrix[2,0].has_samecoef(poly((1, -10/3))**2), 'Incorrect polynomial for N_0,2 on [u_2,u_3)')
        for j in np.append(np.arange(2), np.arange(3,m)):
            self.assertTrue(bfmatrix[j,0].has_samecoef(poly(0)), 'Incorrect polynomial for N_0,2 on [u_{0},u_{1})'.format(j, j+1))
        #



