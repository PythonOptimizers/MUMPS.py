from mumps.src.mumps_INT32_FLOAT64 cimport BaseMUMPSSolver_INT32_FLOAT64, c_to_fortran_index_array, MUMPS_INT
from mumps.src.mumps_INT32_FLOAT64 cimport DMUMPS_COMPLEX

import numpy as np
cimport numpy as cnp

cnp.import_array()


cdef class NumpyMUMPSSolver_INT32_FLOAT64(BaseMUMPSSolver_INT32_FLOAT64):
    cpdef get_matrix_data(self, cnp.ndarray[MUMPS_INT, ndim=1] arow,
                                cnp.ndarray[MUMPS_INT, ndim=1] acol,
                                cnp.ndarray[DMUMPS_COMPLEX, ndim=1] aval)