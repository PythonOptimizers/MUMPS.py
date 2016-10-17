"""
Factory method to access MUMPS.
"""
import numpy as np

from mumps.src.numpy_mumps_INT32_COMPLEX64 import NumpyMUMPSSolver_INT32_COMPLEX64
from mumps.src.numpy_mumps_INT32_COMPLEX128 import NumpyMUMPSSolver_INT32_COMPLEX128
from mumps.src.numpy_mumps_INT32_FLOAT32 import NumpyMUMPSSolver_INT32_FLOAT32
from mumps.src.numpy_mumps_INT32_FLOAT64 import NumpyMUMPSSolver_INT32_FLOAT64

cysparse_installed = False
try:
    from mumps.src.cysparse_mumps_INT32_COMPLEX64 import CySparseMUMPSSolver_INT32_COMPLEX64
    from mumps.src.cysparse_mumps_INT32_COMPLEX128 import CySparseMUMPSSolver_INT32_COMPLEX128
    from mumps.src.cysparse_mumps_INT32_FLOAT32 import CySparseMUMPSSolver_INT32_FLOAT32
    from mumps.src.cysparse_mumps_INT32_FLOAT64 import CySparseMUMPSSolver_INT32_FLOAT64
    from cysparse.sparse.ll_mat import PyLLSparseMatrix_Check
    from cysparse.common_types.cysparse_types import *
    cysparse_installed = True
except:
    pass

allowed_types = '\titype:INT32,\n\tdtype:COMPLEX64,COMPLEX128,FLOAT32,FLOAT64,\n'
type_error_msg = 'Matrix has an index and/or element type that is incompatible with MUMPS\nAllowed types:\n%s' % allowed_types

def MUMPSContext(arg1, verbose=False):
    """
    Create and return the right MUMPS context based on the element type
    supplied as input.

    MUMPS ("MUltifrontal Massively Parallel Solver") is a package for solving systems
    of linear equations of the form Ax = b, where A is a square **sparse** matrix that can be
    either unsymmetric, symmetric positive definite, or general symmetric, on distributed
    memory computers.

    MUMPS performs a Gaussian factorization
      A = LU
    where L is a lower triangular matrix and U an upper triangular matrix.

    If the matrix is symmetric then MUMPS performs the factorization
      A = LDL^T
    where D is block diagonal matrix.

    Args:
        n: size of matrix A
        a_row: row indices of non zero elements of A
        a_col: column indices of non zero elements of A
        a_val: values of non zeros elements of A
        sym:   a boolean indicating if A is a symmetric matrix or not
        verbose: a boolean to turn on or off the verbosity of MUMPS
    """
    if isinstance(arg1, tuple):
        if len(arg1) != 5:
            raise ValueError("If a tuple is supplied, it must have 5"+
                             "items: n, a_row, a_col, a_val, sym")
        n = arg1[0]
        a_row = arg1[1]
        a_col = arg1[2]
        a_val = arg1[3]
        nnz = a_row.shape[0]
        assert a_col.shape[0] == nnz
        assert a_val.shape[0] == nnz

        sym = arg1[4]

        itype = a_row.dtype
        dtype = a_val.dtype

        if itype == np.int32:
            if dtype == np.complex64:
                solver = NumpyMUMPSSolver_INT32_COMPLEX64(n, nnz, sym=sym, verbose=verbose)
                solver.get_matrix_data(a_row, a_col, a_val)
                return solver
            elif dtype == np.complex128:
                solver = NumpyMUMPSSolver_INT32_COMPLEX128(n, nnz, sym=sym, verbose=verbose)
                solver.get_matrix_data(a_row, a_col, a_val)
                return solver
            elif dtype == np.float32:
                solver = NumpyMUMPSSolver_INT32_FLOAT32(n, nnz, sym=sym, verbose=verbose)
                solver.get_matrix_data(a_row, a_col, a_val)
                return solver
            elif dtype == np.float64:
                solver = NumpyMUMPSSolver_INT32_FLOAT64(n, nnz, sym=sym, verbose=verbose)
                solver.get_matrix_data(a_row, a_col, a_val)
                return solver
        else:
            raise TypeError(type_error_msg)

    elif cysparse_installed:
        if not PyLLSparseMatrix_Check(arg1):
            raise TypeError('arg1 should be a LLSparseMatrix')

        A = arg1
        nnz = A.nnz
        itype = A.itype
        dtype = A.dtype
        n = A.nrow
        sym = A.is_symmetric
        assert A.ncol == n

        if itype == INT32_T:
            if dtype == COMPLEX64_T:
                solver = CySparseMUMPSSolver_INT32_COMPLEX64(n, nnz, sym=sym, verbose=verbose)
                solver.get_matrix_data(A)
                return solver
            elif dtype == COMPLEX128_T:
                solver = CySparseMUMPSSolver_INT32_COMPLEX128(n, nnz, sym=sym, verbose=verbose)
                solver.get_matrix_data(A)
                return solver
            elif dtype == FLOAT32_T:
                solver = CySparseMUMPSSolver_INT32_FLOAT32(n, nnz, sym=sym, verbose=verbose)
                solver.get_matrix_data(A)
                return solver
            elif dtype == FLOAT64_T:
                solver = CySparseMUMPSSolver_INT32_FLOAT64(n, nnz, sym=sym, verbose=verbose)
                solver.get_matrix_data(A)
                return solver
        else:
            raise TypeError(type_error_msg)
    else:
        raise TypeError(type_error_msg)