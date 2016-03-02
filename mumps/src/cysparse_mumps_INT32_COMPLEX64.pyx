from cysparse.common_types.cysparse_types cimport *
from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_COMPLEX64_t cimport LLSparseMatrix_INT32_t_COMPLEX64_t

from mumps.src.mumps_INT32_COMPLEX64 cimport BaseMUMPSSolver_INT32_COMPLEX64, c_to_fortran_index_array, MUMPS_INT
 
from mumps.src.mumps_INT32_COMPLEX64 cimport CMUMPS_COMPLEX

from cpython.mem cimport PyMem_Malloc, PyMem_Free

from libc.stdint cimport int64_t
from libc.string cimport strncpy

import numpy as np
cimport numpy as cnp

cnp.import_array()

import time

cdef class CySparseMUMPSSolver_INT32_COMPLEX64(BaseMUMPSSolver_INT32_COMPLEX64):
    """
    MUMPS Context.

    This version **only** deals with ``LLSparseMatrix_INT32_t_COMPLEX64_t`` objects.

    We follow the common use of MUMPS. In particular, we use the same names for the methods of this
    class as their corresponding counter-parts in MUMPS.

    """
    def __cinit__(self, MUMPS_INT n, MUMPS_INT nnz,
                  comm_fortran=-987654, sym=False, verbose=False):
        """
        Args:
            n: size of matrix A
            nnz: number of non zero elements of matrix A
            comm_fortran: MPI communicator
            sym:   a boolean indicating if A is a symmetric matrix or not
            verbose: a boolean to turn on or off the verbosity of MUMPS
        Warning:
            The solver takes a "snapshot" of the matrix ``A``, i.e. the results given by the solver are only
            valid for the matrix given. If the matrix ``A`` changes aferwards, the results given by the solver won't
            reflect this change.
        """
        pass

    def __dealloc__(self):
        PyMem_Free(self.params.irn)
        PyMem_Free(self.params.jcn)
        PyMem_Free(self.params.a)

    cpdef get_matrix_data(self, LLSparseMatrix_INT32_t_COMPLEX64_t A):
        """
        Args:
            A: :class:`LLSparseMatrix_INT32_t_COMPLEX64_t` object.

        Note: we keep the same name for this method in all derived classes.
        """
        assert A.ncol == A.nrow

        n = A.nrow
        nnz = A.nnz
        assert self.sym == A.is_symmetric

        # create i, j, val
        arow = <MUMPS_INT *> PyMem_Malloc(nnz * sizeof(MUMPS_INT))
        acol = <MUMPS_INT *> PyMem_Malloc(nnz * sizeof(MUMPS_INT))
 
 
        a_val = <COMPLEX64_t *> PyMem_Malloc(nnz * sizeof(COMPLEX64_t))
        A.fill_triplet(arow, acol, a_val)
        aval = <CMUMPS_COMPLEX *> a_val

        self.get_data_pointers(arow, acol, aval)