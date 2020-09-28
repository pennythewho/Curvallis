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

    def test__initialize_output_xy_only(self):
        cols = ['x', 'y']
        out_dict = io._initialize_output(cols)
        self.assertEqual(2, len(out_dict.keys()))
        self.assertTrue('x' in out_dict)
        self.assertEqual(0, out_dict['x'].size)
        self.assertEqual((0,), out_dict['x'].shape)
        self.assertTrue('y' in out_dict)
        self.assertEqual(0, out_dict['y'].size)
        self.assertEqual((0,), out_dict['y'].shape)

    def test__initialize_output_xyw(self):
        cols = ['x', 'y', 'w']
        out_dict = io._initialize_output(cols)
        self.assertEqual(3, len(out_dict.keys()))
        self.assertTrue('x' in out_dict)
        self.assertTrue('y' in out_dict)
        self.assertTrue('w' in out_dict)
        self.assertEqual(0, out_dict['w'].size)
        self.assertEqual((0,), out_dict['w'].shape)

    def test__initialize_output_xy_d1_no_wt(self):
        cols = ['x', 'y', 'd1y']
        out_dict = io._initialize_output(cols)
        self.assertEqual(3, len(out_dict.keys()))
        self.assertTrue('x' in out_dict)
        self.assertTrue('y' in out_dict)
        self.assertTrue('set_d1_x' in out_dict)
        self.assertEqual(0, out_dict['set_d1_x'].size)
        self.assertEqual((0,2), out_dict['set_d1_x'].shape)

    def test__initialize_output_xyw_d2_weighted(self):
        cols = ['x', 'y', 'w', 'd2y', 'd2w']
        out_dict = io._initialize_output(cols)
        self.assertEqual(5, len(out_dict.keys()))
        self.assertTrue('x' in out_dict)
        self.assertTrue('y' in out_dict)
        self.assertTrue('y' in out_dict)
        self.assertTrue('set_d2_x' in out_dict)
        self.assertTrue('set_d2_w' in out_dict)
        self.assertEqual(0, out_dict['set_d2_w'].size)
        self.assertEqual((0,), out_dict['set_d2_w'].shape)

    def test__parse_line_xy(self):
        cols = ['x', 'y']
        line = '3.4, 5.6'
        out = io._parse_line(cols, line)
        self.assertEqual(2, len(out.keys()))
        self.assertEqual(3.4, out['x'])
        self.assertEqual(5.6, out['y'])

    def test__parse_line_xyw(self):
        cols = ['x', 'y', 'w']
        line = '3.4, 5.6, 2'
        out = io._parse_line(cols, line)
        self.assertEqual(3, len(out.keys()))
        self.assertEqual(3.4, out['x'])
        self.assertEqual(5.6, out['y'])
        self.assertEqual(2.0, out['w'])

    def test__parse_line_xy_d1unweighted(self):
        cols = ['x', 'y', 'd1y']
        line = '3.4, 5.6, 2'
        out = io._parse_line(cols, line)
        self.assertEqual(3, len(out.keys()))
        self.assertEqual(3.4, out['x'])
        self.assertEqual(5.6, out['y'])
        self.assertEqual((3.4, 2.0), out['set_d1_x'])

    def test__parse_line_xyw_d2unweighted(self):
        cols = ['x', 'y', 'w', 'd2y']
        line = '3.4, 5.6, 1.5, 0'
        out = io._parse_line(cols, line)
        self.assertEqual(4, len(out.keys()))
        self.assertEqual(3.4, out['x'])
        self.assertEqual(5.6, out['y'])
        self.assertEqual(1.5, out['w'])
        self.assertEqual((3.4, 0.0), out['set_d2_x'])

    def test__parse_line_xy_d1_d2w(self):
        cols = ['x', 'y', 'd1y', 'd2y', 'd2w']
        line = '3.4, 5.6, 0, 1.5, 2'
        out = io._parse_line(cols, line)
        self.assertEqual(5, len(out.keys()))
        self.assertEqual(3.4, out['x'])
        self.assertEqual(5.6, out['y'])
        self.assertEqual((3.4, 0.0), out['set_d1_x'])
        self.assertEqual((3.4, 1.5), out['set_d2_x'])
        self.assertEqual(2.0, out['set_d2_w'])