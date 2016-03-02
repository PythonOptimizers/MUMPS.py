from mumps.src.mumps_INT32_COMPLEX64 cimport BaseMUMPSSolver_INT32_COMPLEX64, c_to_fortran_index_array, MUMPS_INT
from mumps.src.mumps_INT32_COMPLEX64 cimport CMUMPS_COMPLEX

import numpy as np
cimport numpy as cnp

cnp.import_array()


cdef class NumpyMUMPSSolver_INT32_COMPLEX64(BaseMUMPSSolver_INT32_COMPLEX64):
    """
    MUMPS Context.

    Only deals with Numpy arrays.

    We follow the common use of MUMPS. In particular, we use the same names for the methods of this
    class as their corresponding counter-parts in MUMPS.

    Warning: if the numpy arrays are modified externally by the user between 2 calls to solve,
    the changes in arrays won't be passed to MUMPS.
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

        Warning: if the numpy arrays are modified externally by the user between 2 calls to solve,
        the changes in arrays won't be passed to MUMPS.
        """
        pass

    cpdef get_matrix_data(self, cnp.ndarray[MUMPS_INT, ndim=1] arow,
                                cnp.ndarray[MUMPS_INT, ndim=1] acol,
                                cnp.ndarray[CMUMPS_COMPLEX, ndim=1] aval):
        """

        Args:
            arow: row indices of non zero elements of A
            acol: column indices of non zero elements of A
            aval: values of non zeros elements of A

        Note: we keep the same name for this method in all derived classes.
        """

        assert arow.shape[0] == self.nnz
        assert acol.shape[0] == self.nnz
        assert aval.shape[0] == self.nnz

        self.get_data_pointers(<MUMPS_INT *> cnp.PyArray_DATA(arow),
                               <MUMPS_INT *> cnp.PyArray_DATA(acol),
                               <CMUMPS_COMPLEX *> cnp.PyArray_DATA(aval))