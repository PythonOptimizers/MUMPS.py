#!/usr/bin/env python
"""
This file tests basic umfpack operations on **all** types supported by MUMPS
and on symmetric and general matrices.
"""

from mumps.mumps_context import MUMPSContext
from unittest import TestCase
import numpy as np
import pytest

class NumpyMUMPSContextTestCase_INT32_COMPLEX64(TestCase):
    def setUp(self):
        self.n = 4
        self.A = np.array([[1, 2, 3, 4], [5, 0, 7, 8], [9, 10, 0, 12], [13, 14, 15, 0]], dtype=np.complex64)
        self.arow = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
        self.acol = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
        self.aval = np.array([1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 0, 12, 13, 14, 15, 0], dtype=np.complex64)
        self.sym = False

    def test_init(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        assert self.sym == context.sym

    def test_analyze(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.analyze()
        assert context.analyzed is True

    def test_factorize(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        assert context.analyzed is True
        assert context.factorized is True

    def test_dense_solve_single_rhs(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        e = np.ones(self.n, dtype=np.complex64)
        rhs = np.dot(self.A, e)
        x = context.solve(rhs=rhs)
        assert np.allclose(x, e)

    def test_dense_solve_multiple_rhs(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        B = np.ones([self.n, 3], dtype=np.complex64)
        B[:, 1] = 2 * B[:, 1]
        B[:, 2] = 3 * B[:, 2]
        rhs = np.dot(self.A, B)
        x = context.solve(rhs=rhs)
        assert np.allclose(x, B, 1e-6)

    def test_sparse_solve_multiple_rhs(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        acol_csc = np.array([1, 5, 9, 13, 17],
        dtype = np.int32)-1
        arow_csc = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                            dtype=np.int32) - 1
        aval_csc = np.array([1, 5, 9, 13, 2, 0, 10, 14, 3, 7, 0, 15, 4, 8, 12, 0],
                            dtype=np.complex64)
        x = context.solve(rhs_col_ptr=acol_csc, rhs_row_ind=arow_csc,
                          rhs_val=aval_csc)
        assert np.allclose(x, np.eye(4, dtype=np.complex64),
                           1e-4, 1e-4)

    def test_iterative_refinement_single_rhs(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        e = np.ones(self.n, dtype=np.complex64)
        rhs = np.dot(self.A, e)
        x = context.solve(rhs=rhs)
        x = context.refine(rhs, 3)
        assert np.allclose(x, e)


class NumpyMUMPSContextTestCase_INT32_COMPLEX128(TestCase):
    def setUp(self):
        self.n = 4
        self.A = np.array([[1, 2, 3, 4], [5, 0, 7, 8], [9, 10, 0, 12], [13, 14, 15, 0]], dtype=np.complex128)
        self.arow = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
        self.acol = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
        self.aval = np.array([1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 0, 12, 13, 14, 15, 0], dtype=np.complex128)
        self.sym = False

    def test_init(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        assert self.sym == context.sym

    def test_analyze(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.analyze()
        assert context.analyzed is True

    def test_factorize(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        assert context.analyzed is True
        assert context.factorized is True

    def test_dense_solve_single_rhs(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        e = np.ones(self.n, dtype=np.complex128)
        rhs = np.dot(self.A, e)
        x = context.solve(rhs=rhs)
        assert np.allclose(x, e)

    def test_dense_solve_multiple_rhs(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        B = np.ones([self.n, 3], dtype=np.complex128)
        B[:, 1] = 2 * B[:, 1]
        B[:, 2] = 3 * B[:, 2]
        rhs = np.dot(self.A, B)
        x = context.solve(rhs=rhs)
        assert np.allclose(x, B, 1e-6)

    def test_sparse_solve_multiple_rhs(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        acol_csc = np.array([1, 5, 9, 13, 17],
        dtype = np.int32)-1
        arow_csc = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                            dtype=np.int32) - 1
        aval_csc = np.array([1, 5, 9, 13, 2, 0, 10, 14, 3, 7, 0, 15, 4, 8, 12, 0],
                            dtype=np.complex128)
        x = context.solve(rhs_col_ptr=acol_csc, rhs_row_ind=arow_csc,
                          rhs_val=aval_csc)
        assert np.allclose(x, np.eye(4, dtype=np.complex128),
                           1e-4, 1e-4)

    def test_iterative_refinement_single_rhs(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        e = np.ones(self.n, dtype=np.complex128)
        rhs = np.dot(self.A, e)
        x = context.solve(rhs=rhs)
        x = context.refine(rhs, 3)
        assert np.allclose(x, e)


class NumpyMUMPSContextTestCase_INT32_FLOAT32(TestCase):
    def setUp(self):
        self.n = 4
        self.A = np.array([[1, 2, 3, 4], [5, 0, 7, 8], [9, 10, 0, 12], [13, 14, 15, 0]], dtype=np.float32)
        self.arow = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
        self.acol = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
        self.aval = np.array([1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 0, 12, 13, 14, 15, 0], dtype=np.float32)
        self.sym = False

    def test_init(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        assert self.sym == context.sym

    def test_analyze(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.analyze()
        assert context.analyzed is True

    def test_factorize(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        assert context.analyzed is True
        assert context.factorized is True

    def test_dense_solve_single_rhs(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        e = np.ones(self.n, dtype=np.float32)
        rhs = np.dot(self.A, e)
        x = context.solve(rhs=rhs)
        assert np.allclose(x, e)

    def test_dense_solve_multiple_rhs(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        B = np.ones([self.n, 3], dtype=np.float32)
        B[:, 1] = 2 * B[:, 1]
        B[:, 2] = 3 * B[:, 2]
        rhs = np.dot(self.A, B)
        x = context.solve(rhs=rhs)
        assert np.allclose(x, B, 1e-6)

    def test_sparse_solve_multiple_rhs(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        acol_csc = np.array([1, 5, 9, 13, 17],
        dtype = np.int32)-1
        arow_csc = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                            dtype=np.int32) - 1
        aval_csc = np.array([1, 5, 9, 13, 2, 0, 10, 14, 3, 7, 0, 15, 4, 8, 12, 0],
                            dtype=np.float32)
        x = context.solve(rhs_col_ptr=acol_csc, rhs_row_ind=arow_csc,
                          rhs_val=aval_csc)
        assert np.allclose(x, np.eye(4, dtype=np.float32),
                           1e-4, 1e-4)

    def test_iterative_refinement_single_rhs(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        e = np.ones(self.n, dtype=np.float32)
        rhs = np.dot(self.A, e)
        x = context.solve(rhs=rhs)
        x = context.refine(rhs, 3)
        assert np.allclose(x, e)


class NumpyMUMPSContextTestCase_INT32_FLOAT64(TestCase):
    def setUp(self):
        self.n = 4
        self.A = np.array([[1, 2, 3, 4], [5, 0, 7, 8], [9, 10, 0, 12], [13, 14, 15, 0]], dtype=np.float64)
        self.arow = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
        self.acol = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
        self.aval = np.array([1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 0, 12, 13, 14, 15, 0], dtype=np.float64)
        self.sym = False

    def test_init(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        assert self.sym == context.sym

    def test_analyze(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.analyze()
        assert context.analyzed is True

    def test_factorize(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        assert context.analyzed is True
        assert context.factorized is True

    def test_dense_solve_single_rhs(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        e = np.ones(self.n, dtype=np.float64)
        rhs = np.dot(self.A, e)
        x = context.solve(rhs=rhs)
        assert np.allclose(x, e)

    def test_dense_solve_multiple_rhs(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        B = np.ones([self.n, 3], dtype=np.float64)
        B[:, 1] = 2 * B[:, 1]
        B[:, 2] = 3 * B[:, 2]
        rhs = np.dot(self.A, B)
        x = context.solve(rhs=rhs)
        assert np.allclose(x, B, 1e-6)

    def test_sparse_solve_multiple_rhs(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        acol_csc = np.array([1, 5, 9, 13, 17],
        dtype = np.int32)-1
        arow_csc = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                            dtype=np.int32) - 1
        aval_csc = np.array([1, 5, 9, 13, 2, 0, 10, 14, 3, 7, 0, 15, 4, 8, 12, 0],
                            dtype=np.float64)
        x = context.solve(rhs_col_ptr=acol_csc, rhs_row_ind=arow_csc,
                          rhs_val=aval_csc)
        assert np.allclose(x, np.eye(4, dtype=np.float64),
                           1e-4, 1e-4)

    def test_iterative_refinement_single_rhs(self):
        context = MUMPSContext((self.n, self.arow, self.acol, self.aval, self.sym), verbose=False)
        context.factorize()
        e = np.ones(self.n, dtype=np.float64)
        rhs = np.dot(self.A, e)
        x = context.solve(rhs=rhs)
        x = context.refine(rhs, 3)
        assert np.allclose(x, e)


