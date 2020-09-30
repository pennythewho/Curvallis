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

    def test__get_kw_deriv(self):
        self.assertEqual('set_d1_x', io._get_kw('1', False))

    def test__get_kw_deriv_wt(self):
        self.assertEqual('set_d2_w', io._get_kw('2', True))

    def test__get_kw_deriv_multidigit(self):
        self.assertEqual('set_d10_x', io._get_kw('10', False))

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
        self.assertTrue('w' in out_dict)
        self.assertTrue('set_d2_x' in out_dict)
        self.assertTrue('set_d2_w' in out_dict)
        self.assertEqual(0, out_dict['set_d2_w'].size)
        self.assertEqual((0,), out_dict['set_d2_w'].shape)

    def test__initialize_output_xy_d10_weighted(self):
        cols = ['x', 'y', 'd10y', 'd10w']
        out_dict = io._initialize_output(cols)
        self.assertEqual(4, len(out_dict.keys()))
        self.assertTrue('x' in out_dict)
        self.assertTrue('y' in out_dict)
        self.assertTrue('set_d10_x' in out_dict)
        self.assertEqual(0, out_dict['set_d10_x'].size)
        self.assertEqual((0,2), out_dict['set_d10_x'].shape)
        self.assertTrue('set_d10_w' in out_dict)
        self.assertEqual(0, out_dict['set_d10_w'].size)
        self.assertEqual((0,), out_dict['set_d10_w'].shape)

    def test__parse_line_xy(self):
        cols = ['x', 'y']
        line = '3.4, 5.6'
        out = io._parse_line(io._initialize_output(cols), cols, line)
        self.assertEqual(2, len(out.keys()))
        self.assertEqual(3.4, out['x'])
        self.assertEqual(5.6, out['y'])

    def test__parse_line_xy_second_line(self):
        cols = ['x', 'y']
        l1 = '3.4, 5.6'
        out = io._parse_line(io._initialize_output(cols), cols, l1)
        l2 = '4.5,6.7'
        out = io._parse_line(out, cols, l2)
        nptest.assert_array_equal([3.4, 4.5], out['x'])
        nptest.assert_array_equal([5.6, 6.7], out['y'])

    def test__parse_line_xyw(self):
        cols = ['x', 'y', 'w']
        l1 = '3.4, 5.6, 2'
        l2 = '4.5,6.7,1.5'
        out = io._parse_line(io._initialize_output(cols), cols, l1)
        out = io._parse_line(out, cols, l2)
        self.assertEqual(3, len(out.keys()))
        nptest.assert_array_equal([3.4, 4.5], out['x'])
        nptest.assert_array_equal([5.6, 6.7], out['y'])
        nptest.assert_array_equal([2.0, 1.5], out['w'])

    def test__parse_line_xy_d1unweighted(self):
        cols = ['x', 'y', 'd1y']
        l1 = '3.4, 5.6, 2'
        l2 = '4.5, 6.7, 0'
        out = io._parse_line(io._initialize_output(cols), cols, l1)
        out = io._parse_line(out, cols, l2)
        nptest.assert_array_equal([3.4, 4.5], out['x'])
        nptest.assert_array_equal([5.6, 6.7], out['y'])
        nptest.assert_array_equal([(3.4, 2.0), (4.5, 0.0)], out['set_d1_x'])

    def test__parse_line_xyw_d2unweighted(self):
        cols = ['x', 'y', 'w', 'd2y']
        l1 = '3.4, 5.6, 1.5, 0'
        l2 = '4.5, 6.7, 1.5, 0'
        out = io._parse_line(io._initialize_output(cols), cols, l1)
        out = io._parse_line(out, cols, l2)
        nptest.assert_array_equal([3.4, 4.5], out['x'])
        nptest.assert_array_equal([5.6, 6.7], out['y'])
        nptest.assert_array_equal([1.5, 1.5], out['w'])
        nptest.assert_array_equal([(3.4, 0.0), (4.5, 0.0)], out['set_d2_x'])

    def test__parse_line_xy_d1_d2w_incomplete_lines(self):
        cols = ['x', 'y', 'd1y', 'd2y', 'd2w']
        l1 = '3.4, 5.6'
        l2 = '4.5, 6.7, 0'
        l3 = '5.6, ,,0,2'
        l4 = '7.8, ,1.5,1.6,2.1'
        out = io._parse_line(io._initialize_output(cols), cols, l1)
        out = io._parse_line(out, cols, l2)
        out = io._parse_line(out, cols, l3)
        out = io._parse_line(out, cols, l4)
        nptest.assert_array_equal([3.4, 4.5], out['x'])
        nptest.assert_array_equal([5.6, 6.7], out['y'])
        nptest.assert_array_equal([(4.5, 0.0), (7.8, 1.5)], out['set_d1_x'])
        nptest.assert_array_equal([(5.6, 0.0), (7.8, 1.6)], out['set_d2_x'])
        nptest.assert_array_equal([2.0, 2.1], out['set_d2_w'])

    def test__parse_line_xyw_weight_missing(self):
        cols = ['x', 'y', 'w']
        l1 = '3.4, 5.6, 1.2'
        l2 = '4.5, 6.7'
        out = io._parse_line(io._initialize_output(cols), cols, l1)
        self.assertRaisesRegex(ValueError, 'y value .+ weight is also required', io._parse_line, out=out, cols=cols, line=l2)

    def test__parse_line_xy_d2w_weight_missing(self):
        cols = ['x', 'y', 'd2y', 'd2w']
        l1 = '3.4, 5.6, 2.4, 1.2'
        l2 = '4.5, 6.7, 3.2'
        out = io._parse_line(io._initialize_output(cols), cols, l1)
        self.assertRaisesRegex(ValueError, '2 derivative .+ weight is also required', io._parse_line, out=out, cols=cols, line=l2)

    def test__parse_line_xyw_ignore_wt_if_no_val(self):
        cols = ['x', 'y', 'w', 'd1y']
        l1 = '3.4, 5.6, 1.2'
        l2 = '4.5, ,1.1, 3.4'
        out = io._parse_line(io._initialize_output(cols), cols, l1)
        out = io._parse_line(out, cols, l2)
        self.assertEqual(4, len(out.keys()))
        nptest.assert_array_equal([3.4], out['x'])
        nptest.assert_array_equal([5.6], out['y'])
        nptest.assert_array_equal([1.2], out['w'])
        nptest.assert_array_equal([(4.5, 3.4)], out['set_d1_x'])


    def test__parse_line_xy_d1w_ignore_wt_if_no_val(self):
        cols = ['x', 'y', 'd1y', 'd1w']
        l1 = '3.4, 5.6, 6.7, 1.2'
        l2 = '4.5, 2.2, , 1.1'
        out = io._parse_line(io._initialize_output(cols), cols, l1)
        out = io._parse_line(out, cols, l2)
        self.assertEqual(4, len(out.keys()))
        nptest.assert_array_equal([3.4, 4.5], out['x'])
        nptest.assert_array_equal([5.6, 2.2], out['y'])
        nptest.assert_array_equal([(3.4, 6.7)], out['set_d1_x'])
        nptest.assert_array_equal([1.2], out['set_d1_w'])

    def test__parse_line_missing_x(self):
        cols = ['x', 'y', 'd1y', 'd1w']
        l1 = '3.4, 5.6, 6.7, 1.2'
        l2 = ', 3.3, 2.2, 1.1'
        out = io._parse_line(io._initialize_output(cols), cols, l1)
        self.assertRaisesRegex(ValueError, 'Every line must have an x', io._parse_line, out=out, cols=cols, line=l2)

    def test__parse_line_tab_delim(self):
        cols = ['x', 'y', 'd1y', 'd1w']
        l1 = '3.4\t5.6\t6.7\t1.2'
        out = io._parse_line(io._initialize_output(cols), cols, l1, '\t')
        nptest.assert_array_equal([3.4], out['x'])
        nptest.assert_array_equal([5.6], out['y'])
        nptest.assert_array_equal([(3.4,6.7)], out['set_d1_x'])
        nptest.assert_array_equal([1.2], out['set_d1_w'])

    def test__remove_empty_keys_none_empty(self):
        out = {'x': [3.4, 4.5], 'y': [5.6, 2.2]}
        nptest.assert_array_equal(out, io._remove_empty_keys(out))

    def test__remove_empty_keys_empty_deriv(self):
        out = {'x':[3.4, 4.5], 'y':[5.6, 2.2], 'set_d1_x':[], 'set_d2_x':[(3.4, 6.7), (4.5, 7.8)]}
        newout = io._remove_empty_keys(out)
        self.assertFalse('set_d1_x' in newout.keys())
        self.assertTrue('x' in newout.keys())
        self.assertTrue('y' in newout.keys())
        self.assertTrue('set_d2_x' in newout.keys())

    def test_load_data(self):
        out = io.load_data('bfit/test/importtest.csv', 3, with_weights=[2])
        self.assertEqual(5, len(out.keys()))
        nptest.assert_array_equal([3.4, 4.5, 6.7], out['x'])
        nptest.assert_array_equal([4.5, 5.6, 7.8], out['y'])
        nptest.assert_array_equal([(5.6, 2.3), (6.7, 2.0)], out['set_d1_x'])
        nptest.assert_array_equal([(4.5, 1.2), (5.6, 0.0), (6.7, 1.0)], out['set_d2_x'])
        nptest.assert_array_equal([1.1, 1.5, 1.1], out['set_d2_w'])




