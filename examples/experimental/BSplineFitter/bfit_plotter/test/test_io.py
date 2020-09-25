import unittest as ut
import numpy as np
from numpy import testing as nptest
from .. import io

class TestIo(ut.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test__get_expected_cols_p2_no_weights(self):
        ecols = ['x', 'y', 'd1y', 'd2y']
        nptest.assert_array_equal(ecols, io._get_expected_cols(2, []))

    def test__get_expected_cols_p3_no_weights(self):
        ecols = ['x', 'y', 'd1y', 'd2y', 'd3y']
        nptest.assert_array_equal(ecols, io._get_expected_cols(3, []))

    def test__get_expected_cols_p2_d0_weighted(self):
        ecols = ['x', 'y', 'w', 'd1y', 'd2y']
        nptest.assert_array_equal(ecols, io._get_expected_cols(2, [0]))

    def test__get_expected_cols_p2_d1_weighted(self):
        ecols = ['x', 'y', 'd1y', 'd1w', 'd2y']
        nptest.assert_array_equal(ecols, io._get_expected_cols(2, [1]))

    def test__get_expected_cols_p2_d2_weighted(self):
        ecols = ['x', 'y', 'd1y', 'd2y', 'd2w']
        nptest.assert_array_equal(ecols, io._get_expected_cols(2, [2]))

    def test__get_expected_cols_p2_ignore_high_deriv_weight(self):
        ecols = ['x', 'y', 'd1y', 'd2y']
        nptest.assert_array_equal(ecols, io._get_expected_cols(2, [3]))

    def test__get_expected_cols_p2_d0d1_weighted(self):
        ecols = ['x', 'y', 'w', 'd1y', 'd1w', 'd2y']
        nptest.assert_array_equal(ecols, io._get_expected_cols(2, [0, 1]))

    def test__get_expected_cols_p2_d1d2_weighted(self):
        ecols = ['x', 'y', 'd1y', 'd1w', 'd2y', 'd2w']
        nptest.assert_array_equal(ecols, io._get_expected_cols(2, [1, 2]))

    def test__get_expected_cols_p2_d0d2_weighted(self):
        ecols = ['x', 'y', 'w', 'd1y', 'd2y', 'd2w']
        nptest.assert_array_equal(ecols, io._get_expected_cols(2, [0, 2]))

    def test__get_expected_cols_p2_d0d1d2_weighted(self):
        ecols = ['x', 'y', 'w', 'd1y', 'd1w', 'd2y', 'd2w']
        nptest.assert_array_equal(ecols, io._get_expected_cols(2, [0, 1, 2]))