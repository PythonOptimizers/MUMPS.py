from mumps.src.mumps_INT32_FLOAT32 cimport BaseMUMPSSolver_INT32_FLOAT32, c_to_fortran_index_array, MUMPS_INT
from mumps.src.mumps_INT32_FLOAT32 cimport SMUMPS_COMPLEX

import numpy as np
cimport numpy as cnp

cnp.import_array()


cdef class NumpyMUMPSSolver_INT32_FLOAT32(BaseMUMPSSolver_INT32_FLOAT32):
    cpdef get_matrix_data(self, cnp.ndarray[MUMPS_INT, ndim=1] arow,
                                cnp.ndarray[MUMPS_INT, ndim=1] acol,
                                cnp.ndarray[SMUMPS_COMPLEX, ndim=1] aval)