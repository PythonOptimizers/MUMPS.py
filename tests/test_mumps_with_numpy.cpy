#!/usr/bin/env python

"""
This file tests basic umfpack operations on **all** types supported by MUMPS
and on symmetric and general matrices.
"""

from mumps.mumps_context import MUMPSContext
import numpy as np
from numpy.testing import *
import sys


{% for index_type in mumps_index_list %}
    {% for element_type in mumps_type_list %}
class NumpyMUMPSContextTestCase_@index_type@_@element_type@(TestCase):
    def setUp(self):
        self.n = 4
        self.A = np.array([[1, 2, 3, 4], [5, 0, 7, 8], [9, 10, 0, 12], [13, 14, 15, 0]], dtype=np.@element_type|lower@)
        self.arow = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=np.@index_type|lower@)
        self.acol = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.@index_type|lower@)
        self.aval = np.array([1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 0, 12, 13, 14, 15, 0], dtype=np.@element_type|lower@)
        self.sym = False

    def test_init(self):
        self.context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        assert_equal(self.sym, self.context.sym)

    def test_analyze(self):
        self.context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        self.context.analyze()
        assert(self.context.analyzed==True)

    def test_factorize(self):
        self.context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        self.context.factorize()
        assert(self.context.analyzed==True)
        assert(self.context.factorized==True)

    def test_dense_solve_single_rhs(self):
        self.context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        self.context.factorize()
        e = np.ones(self.n, dtype=np.@element_type|lower@)
        rhs = np.dot(self.A, e)
        x = self.context.solve(rhs=rhs)
        assert_almost_equal(x,e)

    def test_dense_solve_multiple_rhs(self):
        self.context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        self.context.factorize()
        B = np.ones([self.n, 3], dtype=np.@element_type|lower@)
        B[: ,1] = 2 * B[:,1]
        B[: ,2] = 3 * B[:,2]
        rhs = np.dot(self.A,B)
        x = self.context.solve(rhs=rhs)
        assert_almost_equal(x,B,6)

    def test_sparse_solve_multiple_rhs(self):
        self.context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        self.context.factorize()
        acol_csc = np.array([1,5,9,13,17],
        dtype=np.@index_type|lower@)-1
        arow_csc = np.array([1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4], dtype=np.@index_type|lower@)-1
        aval_csc = np.array([1,5,9,13,2,0,10,14,3,7,0,15,4,8,12,0], dtype=np.@element_type|lower@)
        x = self.context.solve(rhs_col_ptr=acol_csc, rhs_row_ind=arow_csc, rhs_val=aval_csc)
        assert_almost_equal(x,np.eye(4),6)
  {% endfor %}
{% endfor %}


