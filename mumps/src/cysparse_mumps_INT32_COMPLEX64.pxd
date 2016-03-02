from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_COMPLEX64_t cimport LLSparseMatrix_INT32_t_COMPLEX64_t

from mumps.src.mumps_INT32_COMPLEX64 cimport BaseMUMPSSolver_INT32_COMPLEX64

cimport numpy as cnp
cdef class CySparseMUMPSSolver_INT32_COMPLEX64(BaseMUMPSSolver_INT32_COMPLEX64):
    cpdef get_matrix_data(self, LLSparseMatrix_INT32_t_COMPLEX64_t A)
 