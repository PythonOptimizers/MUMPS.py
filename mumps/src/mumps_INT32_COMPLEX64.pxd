cimport numpy as cnp

cdef extern from "mumps_c_types.h":

    ctypedef int        MUMPS_INT
    ctypedef cnp.int8_t  MUMPS_INT8

    ctypedef float      SMUMPS_COMPLEX
    ctypedef float      SMUMPS_REAL

    ctypedef double     DMUMPS_COMPLEX
    ctypedef double     DMUMPS_REAL

    ctypedef struct mumps_complex:
        float r,i

    ctypedef mumps_complex  CMUMPS_COMPLEX
    ctypedef float          CMUMPS_REAL

    ctypedef struct mumps_double_complex:
        double r, i

    ctypedef mumps_double_complex  ZMUMPS_COMPLEX
    ctypedef double                ZMUMPS_REAL

cdef extern from "cmumps_c.h":
    ctypedef struct CMUMPS_STRUC_C:
        MUMPS_INT      sym, par, job
        MUMPS_INT      comm_fortran    # Fortran communicator
        MUMPS_INT      icntl[40]
        MUMPS_INT      keep[500]
        CMUMPS_REAL    cntl[15]
        CMUMPS_REAL    dkeep[130];
        MUMPS_INT8     keep8[150];
        MUMPS_INT      n

        # used in matlab interface to decide if we
        # free + malloc when we have large variation
        MUMPS_INT      nz_alloc

        # Assembled entry
        MUMPS_INT      nz
        MUMPS_INT      *irn
        MUMPS_INT      *jcn
        CMUMPS_COMPLEX *a

        # Distributed entry
        MUMPS_INT      nz_loc
        MUMPS_INT      *irn_loc
        MUMPS_INT      *jcn_loc
        CMUMPS_COMPLEX *a_loc

        # Element entry
        MUMPS_INT      nelt
        MUMPS_INT      *eltptr
        MUMPS_INT      *eltvar
        CMUMPS_COMPLEX *a_elt

        # Ordering, if given by user
        MUMPS_INT      *perm_in

        # Orderings returned to user
        MUMPS_INT      *sym_perm    # symmetric permutation
        MUMPS_INT      *uns_perm    # column permutation

        # Scaling (input only in this version)
        CMUMPS_REAL    *colsca
        CMUMPS_REAL    *rowsca
        MUMPS_INT colsca_from_mumps;
        MUMPS_INT rowsca_from_mumps;


        # RHS, solution, ouptput data and statistics
        CMUMPS_COMPLEX *rhs
        CMUMPS_COMPLEX *redrhs
        CMUMPS_COMPLEX *rhs_sparse
        CMUMPS_COMPLEX *sol_loc
        MUMPS_INT      *irhs_sparse
        MUMPS_INT      *irhs_ptr
        MUMPS_INT      *isol_loc
        MUMPS_INT      nrhs, lrhs, lredrhs, nz_rhs, lsol_loc
        MUMPS_INT      schur_mloc, schur_nloc, schur_lld
        MUMPS_INT      mblock, nblock, nprow, npcol
        MUMPS_INT      info[40]
        MUMPS_INT      infog[40]
        CMUMPS_REAL    rinfo[40]
        CMUMPS_REAL    rinfog[40]

        # Null space
        MUMPS_INT      deficiency
        MUMPS_INT      *pivnul_list
        MUMPS_INT      *mapping

        # Schur
        MUMPS_INT      size_schur
        MUMPS_INT      *listvar_schur
        CMUMPS_COMPLEX *schur

        # Internal parameters
        MUMPS_INT      instance_number
        CMUMPS_COMPLEX *wk_user

        char *version_number
        # For out-of-core
        char *ooc_tmpdir
        char *ooc_prefix
        # To save the matrix in matrix market format
        char *write_problem
        MUMPS_INT      lwk_user

    cdef void cmumps_c(CMUMPS_STRUC_C *)

cdef class mumps_int_array:
    """
    Internal classes to use x[i] = value and x[i] setters and getters
    int version.

    """
    cdef:
        MUMPS_INT * array
        int ub

    cdef get_array(self, MUMPS_INT * array, int ub = ?)

cdef class cmumps_real_array:
    """
    Internal classes to use x[i] = value and x[i] setters and getters
    Real version.

    """
    cdef:
        CMUMPS_REAL * array
        int ub

    cdef get_array(self, CMUMPS_REAL * array, int ub = ?)

cdef c_to_fortran_index_array(MUMPS_INT * a, MUMPS_INT a_size)

cdef class BaseMUMPSSolver_INT32_COMPLEX64:
    cdef:

        MUMPS_INT nrow
        MUMPS_INT ncol
        MUMPS_INT nnz

        # MUMPS
        CMUMPS_STRUC_C params

        # internal classes for getters and setters
        mumps_int_array icntl
        mumps_int_array info
        mumps_int_array infog

        cmumps_real_array cntl
        cmumps_real_array rinfo
        cmumps_real_array rinfog

        MUMPS_INT * a_row
        MUMPS_INT * a_col
        CMUMPS_COMPLEX *  a_val

        bint analyzed
        bint factorized
        bint out_of_core

        object analysis_stats
        object factorize_stats
        object solve_stats

    cdef get_data_pointers(self,
                           MUMPS_INT * a_row,
                           MUMPS_INT * a_col,
                           CMUMPS_COMPLEX * a_val)

    cdef initialize_mumps_struct(self, comm_fortran, sym)

    cdef mumps_call(self)

    cdef set_centralized_assembled_matrix(self)

    cdef solve_dense(self, CMUMPS_COMPLEX * rhs,
                     MUMPS_INT rhs_length, MUMPS_INT nrhs)
    cdef solve_sparse(self, MUMPS_INT * rhs_col_ptr, MUMPS_INT * rhs_row_ind,
                      CMUMPS_COMPLEX * rhs_val,
                      MUMPS_INT rhs_nnz, MUMPS_INT nrhs,
                      CMUMPS_COMPLEX * x, MUMPS_INT x_length)
