#!/usr/bin/env python
"""
This file tests basic operations on **all** types supported by MUMPS
and on symmetric and general matrices.
"""
try:
    from cysparse.sparse.ll_mat import *
    import cysparse.common_types.cysparse_types as types
except ImportError:
    pass

from mumps.mumps_context import MUMPSContext
from unittest import TestCase
import numpy as np
import pytest

class CySparseMUMPSContextTestCase_INT32_COMPLEX64(TestCase):
    def setUp(self):
        pytest.importorskip("cysparse.common_types.cysparse_types")
        pytest.importorskip("mumps.src.cysparse_mumps_INT32_COMPLEX64")
        self.n = 4
        arow = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
        acol = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
        aval = np.array([1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 0, 12, 13, 14, 15, 0], dtype=np.complex64)
        self.sym = False
        self.A = LLSparseMatrix(size=self.n, itype=types.INT32_T, dtype=types.COMPLEX64_T)
        self.A.put_triplet(arow, acol, aval)

    def test_init(self):
        context = MUMPSContext(self.A, verbose=False)
        assert self.sym == context.sym

    def test_analyze(self):
        context = MUMPSContext(self.A, verbose=False)
        context.analyze()
        assert context.analyzed is True

    def test_factorize(self):
        context = MUMPSContext(self.A, verbose=False)
        context.factorize()
        assert context.analyzed is True
        assert context.factorized is True

    def test_dense_solve_single_rhs(self):
        context = MUMPSContext(self.A, verbose=False)
        context.factorize()
        e = np.ones(self.n, dtype=np.complex64)
        rhs = self.A * e
        x = context.solve(rhs=rhs)
        assert np.allclose(x, e)

    def test_dense_solve_multiple_rhs(self):
        context = MUMPSContext(self.A, verbose=False)
        context.factorize()
        B = np.ones([self.n, 3], dtype=np.complex64)
        B[:, 1] = 2 * B[:, 1]
        B[:, 2] = 3 * B[:, 2]
        rhs = np.ones([self.n, 3], dtype=np.complex64)
        rhs[:, 0] = self.A * B[:, 0]
        rhs[:, 1] = self.A * B[:, 1]
        rhs[:, 2] = self.A * B[:, 2]
        x = context.solve(rhs=rhs)
        assert np.allclose(x, B, 1e-6)

    def test_sparse_solve_multiple_rhs(self):
        context = MUMPSContext(self.A, verbose=False)
        context.factorize()
        acol_csc = np.array([1, 5, 9, 13, 17], dtype=np.int32)-1
        arow_csc = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                            dtype=np.int32)-1
        aval_csc = np.array([1, 5, 9, 13, 2, 0, 10, 14, 3, 7, 0, 15, 4, 8, 12, 0],
                            dtype=np.complex64)
        x = context.solve(rhs_col_ptr=acol_csc, rhs_row_ind=arow_csc,
                          rhs_val=aval_csc)
        assert np.allclose(x, np.eye(4), 1e-6, 1e-6)

    def test_iterative_refinement_single_rhs(self):
        context = MUMPSContext(self.A, verbose=False)
        context.factorize()
        e = np.ones(self.n, dtype=np.complex64)
        rhs = self.A * e
        x = context.solve(rhs=rhs)
        x = context.refine(rhs, 3)
        assert np.allclose(x, e)


class CySparseMUMPSContextTestCase_INT32_COMPLEX128(TestCase):
    def setUp(self):
        pytest.importorskip("cysparse.common_types.cysparse_types")
        pytest.importorskip("mumps.src.cysparse_mumps_INT32_COMPLEX128")
        self.n = 4
        arow = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
        acol = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
        aval = np.array([1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 0, 12, 13, 14, 15, 0], dtype=np.complex128)
        self.sym = False
        self.A = LLSparseMatrix(size=self.n, itype=types.INT32_T, dtype=types.COMPLEX128_T)
        self.A.put_triplet(arow, acol, aval)

    def test_init(self):
        context = MUMPSContext(self.A, verbose=False)
        assert self.sym == context.sym

    def test_analyze(self):
        context = MUMPSContext(self.A, verbose=False)
        context.analyze()
        assert context.analyzed is True

    def test_factorize(self):
        context = MUMPSContext(self.A, verbose=False)
        context.factorize()
        assert context.analyzed is True
        assert context.factorized is True

    def test_dense_solve_single_rhs(self):
        context = MUMPSContext(self.A, verbose=False)
        context.factorize()
        e = np.ones(self.n, dtype=np.complex128)
        rhs = self.A * e
        x = context.solve(rhs=rhs)
        assert np.allclose(x, e)

    def test_dense_solve_multiple_rhs(self):
        context = MUMPSContext(self.A, verbose=False)
        context.factorize()
        B = np.ones([self.n, 3], dtype=np.complex128)
        B[:, 1] = 2 * B[:, 1]
        B[:, 2] = 3 * B[:, 2]
        rhs = np.ones([self.n, 3], dtype=np.complex128)
        rhs[:, 0] = self.A * B[:, 0]
        rhs[:, 1] = self.A * B[:, 1]
        rhs[:, 2] = self.A * B[:, 2]
        x = context.solve(rhs=rhs)
        assert np.allclose(x, B, 1e-6)

    def test_sparse_solve_multiple_rhs(self):
        context = MUMPSContext(self.A, verbose=False)
        context.factorize()
        acol_csc = np.array([1, 5, 9, 13, 17], dtype=np.int32)-1
        arow_csc = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                            dtype=np.int32)-1
        aval_csc = np.array([1, 5, 9, 13, 2, 0, 10, 14, 3, 7, 0, 15, 4, 8, 12, 0],
                            dtype=np.complex128)
        x = context.solve(rhs_col_ptr=acol_csc, rhs_row_ind=arow_csc,
                          rhs_val=aval_csc)
        assert np.allclose(x, np.eye(4), 1e-6, 1e-6)

    def test_iterative_refinement_single_rhs(self):
        context = MUMPSContext(self.A, verbose=False)
        context.factorize()
        e = np.ones(self.n, dtype=np.complex128)
        rhs = self.A * e
        x = context.solve(rhs=rhs)
        x = context.refine(rhs, 3)
        assert np.allclose(x, e)


class CySparseMUMPSContextTestCase_INT32_FLOAT32(TestCase):
    def setUp(self):
        pytest.importorskip("cysparse.common_types.cysparse_types")
        pytest.importorskip("mumps.src.cysparse_mumps_INT32_FLOAT32")
        self.n = 4
        arow = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
        acol = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
        aval = np.array([1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 0, 12, 13, 14, 15, 0], dtype=np.float32)
        self.sym = False
        self.A = LLSparseMatrix(size=self.n, itype=types.INT32_T, dtype=types.FLOAT32_T)
        self.A.put_triplet(arow, acol, aval)

    def test_init(self):
        context = MUMPSContext(self.A, verbose=False)
        assert self.sym == context.sym

    def test_analyze(self):
        context = MUMPSContext(self.A, verbose=False)
        context.analyze()
        assert context.analyzed is True

    def test_factorize(self):
        context = MUMPSContext(self.A, verbose=False)
        context.factorize()
        assert context.analyzed is True
        assert context.factorized is True

    def test_dense_solve_single_rhs(self):
        context = MUMPSContext(self.A, verbose=False)
        context.factorize()
        e = np.ones(self.n, dtype=np.float32)
        rhs = self.A * e
        x = context.solve(rhs=rhs)
        assert np.allclose(x, e)

    def test_dense_solve_multiple_rhs(self):
        context = MUMPSContext(self.A, verbose=False)
        context.factorize()
        B = np.ones([self.n, 3], dtype=np.float32)
        B[:, 1] = 2 * B[:, 1]
        B[:, 2] = 3 * B[:, 2]
        rhs = np.ones([self.n, 3], dtype=np.float32)
        rhs[:, 0] = self.A * B[:, 0]
        rhs[:, 1] = self.A * B[:, 1]
        rhs[:, 2] = self.A * B[:, 2]
        x = context.solve(rhs=rhs)
        assert np.allclose(x, B, 1e-6)

    def test_sparse_solve_multiple_rhs(self):
        context = MUMPSContext(self.A, verbose=False)
        context.factorize()
        acol_csc = np.array([1, 5, 9, 13, 17], dtype=np.int32)-1
        arow_csc = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                            dtype=np.int32)-1
        aval_csc = np.array([1, 5, 9, 13, 2, 0, 10, 14, 3, 7, 0, 15, 4, 8, 12, 0],
                            dtype=np.float32)
        x = context.solve(rhs_col_ptr=acol_csc, rhs_row_ind=arow_csc,
                          rhs_val=aval_csc)
        assert np.allclose(x, np.eye(4), 1e-6, 1e-6)

    def test_iterative_refinement_single_rhs(self):
        context = MUMPSContext(self.A, verbose=False)
        context.factorize()
        e = np.ones(self.n, dtype=np.float32)
        rhs = self.A * e
        x = context.solve(rhs=rhs)
        x = context.refine(rhs, 3)
        assert np.allclose(x, e)


class CySparseMUMPSContextTestCase_INT32_FLOAT64(TestCase):
    def setUp(self):
        pytest.importorskip("cysparse.common_types.cysparse_types")
        pytest.importorskip("mumps.src.cysparse_mumps_INT32_FLOAT64")
        self.n = 4
        arow = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
        acol = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
        aval = np.array([1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 0, 12, 13, 14, 15, 0], dtype=np.float64)
        self.sym = False
        self.A = LLSparseMatrix(size=self.n, itype=types.INT32_T, dtype=types.FLOAT64_T)
        self.A.put_triplet(arow, acol, aval)

    def test_init(self):
        context = MUMPSContext(self.A, verbose=False)
        assert self.sym == context.sym

    def test_analyze(self):
        context = MUMPSContext(self.A, verbose=False)
        context.analyze()
        assert context.analyzed is True

    def test_factorize(self):
        context = MUMPSContext(self.A, verbose=False)
        context.factorize()
        assert context.analyzed is True
        assert context.factorized is True

    def test_dense_solve_single_rhs(self):
        context = MUMPSContext(self.A, verbose=False)
        context.factorize()
        e = np.ones(self.n, dtype=np.float64)
        rhs = self.A * e
        x = context.solve(rhs=rhs)
        assert np.allclose(x, e)

    def test_dense_solve_multiple_rhs(self):
        context = MUMPSContext(self.A, verbose=False)
        context.factorize()
        B = np.ones([self.n, 3], dtype=np.float64)
        B[:, 1] = 2 * B[:, 1]
        B[:, 2] = 3 * B[:, 2]
        rhs = np.ones([self.n, 3], dtype=np.float64)
        rhs[:, 0] = self.A * B[:, 0]
        rhs[:, 1] = self.A * B[:, 1]
        rhs[:, 2] = self.A * B[:, 2]
        x = context.solve(rhs=rhs)
        assert np.allclose(x, B, 1e-6)

    def test_sparse_solve_multiple_rhs(self):
        context = MUMPSContext(self.A, verbose=False)
        context.factorize()
        acol_csc = np.array([1, 5, 9, 13, 17], dtype=np.int32)-1
        arow_csc = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                            dtype=np.int32)-1
        aval_csc = np.array([1, 5, 9, 13, 2, 0, 10, 14, 3, 7, 0, 15, 4, 8, 12, 0],
                            dtype=np.float64)
        x = context.solve(rhs_col_ptr=acol_csc, rhs_row_ind=arow_csc,
                          rhs_val=aval_csc)
        assert np.allclose(x, np.eye(4), 1e-6, 1e-6)

    def test_iterative_refinement_single_rhs(self):
        context = MUMPSContext(self.A, verbose=False)
        context.factorize()
        e = np.ones(self.n, dtype=np.float64)
        rhs = self.A * e
        x = context.solve(rhs=rhs)
        x = context.refine(rhs, 3)
        assert np.allclose(x, e)


