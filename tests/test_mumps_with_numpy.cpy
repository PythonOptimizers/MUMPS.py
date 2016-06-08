#!/usr/bin/env python
"""
This file tests basic umfpack operations on **all** types supported by MUMPS
and on symmetric and general matrices.
"""

from mumps.mumps_context import MUMPSContext
import numpy as np
import sys
from unittest import TestCase

{% for index_type in index_list %}
    {% for element_type in type_list %}
class NumpyMUMPSContextTestCase_@index_type@_@element_type@(TestCase):
    def setUp(self):
        self.n = 4
        self.A = np.array([[1, 2, 3, 4], [5, 0, 7, 8], [9, 10, 0, 12], [13, 14, 15, 0]], dtype=np.@element_type|lower@)
        self.arow = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=np.@index_type|lower@)
        self.acol = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.@index_type|lower@)
        self.aval = np.array([1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 0, 12, 13, 14, 15, 0], dtype=np.@element_type|lower@)
        self.sym = False

    def test_init(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        assert self.sym == context.sym

    def test_analyze(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.analyze()
        assert context.analyzed == True

    def test_factorize(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        assert context.analyzed == True
        assert context.factorized == True

    def test_dense_solve_single_rhs(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        e = np.ones(self.n, dtype=np.@element_type|lower@)
        rhs = np.dot(self.A, e)
        x = context.solve(rhs=rhs)
        assert np.allclose(x, e)

    def test_dense_solve_multiple_rhs(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        B = np.ones([self.n, 3], dtype=np.@element_type|lower@)
        B[:, 1] = 2 * B[:, 1]
        B[:, 2] = 3 * B[:, 2]
        rhs = np.dot(self.A, B)
        x = context.solve(rhs=rhs)
        assert np.allclose(x, B, 1e-6)

    def test_sparse_solve_multiple_rhs(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        acol_csc = np.array([1, 5, 9, 13, 17],
        dtype=np.@index_type|lower@)-1
        arow_csc = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], dtype=np.@index_type|lower@)-1
        aval_csc = np.array([1, 5, 9, 13, 2, 0, 10, 14, 3, 7, 0, 15, 4, 8, 12, 0], dtype=np.@element_type|lower@)
        x = context.solve(rhs_col_ptr=acol_csc, rhs_row_ind=arow_csc, rhs_val=aval_csc)
        assert np.allclose(x, np.eye(4, dtype=np.@element_type|lower@), 1e-4, 1e-4)

    def test_iterative_refinement_single_rhs(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        e = np.ones(self.n, dtype=np.@element_type|lower@)
        rhs = np.dot(self.A, e)
        x = context.solve(rhs=rhs)
        x = context.refine(rhs, 3)
        assert np.allclose(x, e)


  {% endfor %}
{% endfor %}

if __name__ == "__main__":
      run_module_suite()
