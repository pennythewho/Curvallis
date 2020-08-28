import unittest as ut
import numpy as np
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
        self.assertRaises(ValueError, bf.get_basis_function, icp=-1, degree=p, knots=knots, iks=0)
        self.assertRaises(ValueError, bf.get_basis_function, icp=n+1, degree=p, knots=knots, iks=0)

    def test_get_basis_function_iks_out_of_range(self):
        m = 2
        knots = np.arange(m + 1)
        p = 2
        self.assertRaises(ValueError, bf.get_basis_function, icp=0, degree=p, knots=knots, iks=-1)
        self.assertRaises(ValueError, bf.get_basis_function, icp=0, degree=p, knots=knots, iks=m+1)

    def test_get_basis_function_simple_knots(self):
        p = 2
        m = 4
        knots = np.arange(m+1)/m
        # N_0,p
        assert(bf.get_basis_function(0, p, knots, 0).has_samecoef(poly((0, 0, 8))))
        assert(bf.get_basis_function(0, p, knots, 1).has_samecoef(poly((-1.5, 12, -16))))
        assert(bf.get_basis_function(0, p, knots, 2).has_samecoef(poly((4.5, -12, 8))))
        assert(bf.get_basis_function(0, p, knots, 3).has_samecoef(poly(0)))
        # N_1,p
        assert(bf.get_basis_function(1, p, knots, 0).has_samecoef(poly(0)))
        assert(bf.get_basis_function(1, p, knots, 1).has_samecoef(poly((0.5, -4, 8))))
        assert(bf.get_basis_function(1, p, knots, 2).has_samecoef(poly((-5.5, 20, -16))))
        assert(bf.get_basis_function(1, p, knots, 3).has_samecoef(poly((8, -16, 8))))

