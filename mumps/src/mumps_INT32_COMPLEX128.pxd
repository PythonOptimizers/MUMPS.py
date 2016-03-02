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

cdef extern from "zmumps_c.h":
    ctypedef struct ZMUMPS_STRUC_C:
        MUMPS_INT      sym, par, job
        MUMPS_INT      comm_fortran    # Fortran communicator
        MUMPS_INT      icntl[40]
        MUMPS_INT      keep[500]
        ZMUMPS_REAL    cntl[15]
        ZMUMPS_REAL    dkeep[130];
        MUMPS_INT8     keep8[150];
        MUMPS_INT      n

        # used in matlab interface to decide if we
        # free + malloc when we have large variation
        MUMPS_INT      nz_alloc

        # Assembled entry
        MUMPS_INT      nz
        MUMPS_INT      *irn
        MUMPS_INT      *jcn
        ZMUMPS_COMPLEX *a

        # Distributed entry
        MUMPS_INT      nz_loc
        MUMPS_INT      *irn_loc
        MUMPS_INT      *jcn_loc
        ZMUMPS_COMPLEX *a_loc

        # Element entry
        MUMPS_INT      nelt
        MUMPS_INT      *eltptr
        MUMPS_INT      *eltvar
        ZMUMPS_COMPLEX *a_elt

        # Ordering, if given by user
        MUMPS_INT      *perm_in

        # Orderings returned to user
        MUMPS_INT      *sym_perm    # symmetric permutation
        MUMPS_INT      *uns_perm    # column permutation

        # Scaling (input only in this version)
        ZMUMPS_REAL    *colsca
        ZMUMPS_REAL    *rowsca
        MUMPS_INT colsca_from_mumps;
        MUMPS_INT rowsca_from_mumps;


        # RHS, solution, ouptput data and statistics
        ZMUMPS_COMPLEX *rhs
        ZMUMPS_COMPLEX *redrhs
        ZMUMPS_COMPLEX *rhs_sparse
        ZMUMPS_COMPLEX *sol_loc
        MUMPS_INT      *irhs_sparse
        MUMPS_INT      *irhs_ptr
        MUMPS_INT      *isol_loc
        MUMPS_INT      nrhs, lrhs, lredrhs, nz_rhs, lsol_loc
        MUMPS_INT      schur_mloc, schur_nloc, schur_lld
        MUMPS_INT      mblock, nblock, nprow, npcol
        MUMPS_INT      info[40]
        MUMPS_INT      infog[40]
        ZMUMPS_REAL    rinfo[40]
        ZMUMPS_REAL    rinfog[40]

        # Null space
        MUMPS_INT      deficiency
        MUMPS_INT      *pivnul_list
        MUMPS_INT      *mapping

        # Schur
        MUMPS_INT      size_schur
        MUMPS_INT      *listvar_schur
        ZMUMPS_COMPLEX *schur

        # Internal parameters
        MUMPS_INT      instance_number
        ZMUMPS_COMPLEX *wk_user

        char *version_number
        # For out-of-core
        char *ooc_tmpdir
        char *ooc_prefix
        # To save the matrix in matrix market format
        char *write_problem
        MUMPS_INT      lwk_user

    cdef void zmumps_c(ZMUMPS_STRUC_C *)

cdef class mumps_int_array:
    """
    Internal classes to use x[i] = value and x[i] setters and getters
    int version.

    """
    cdef:
        MUMPS_INT * array
        int ub

    cdef get_array(self, MUMPS_INT * array, int ub = ?)

cdef class zmumps_real_array:
    """
    Internal classes to use x[i] = value and x[i] setters and getters
    Real version.

    """
    cdef:
        ZMUMPS_REAL * array
        int ub

    cdef get_array(self, ZMUMPS_REAL * array, int ub = ?)

cdef c_to_fortran_index_array(MUMPS_INT * a, MUMPS_INT a_size)

cdef class BaseMUMPSSolver_INT32_COMPLEX128:
    cdef:

        MUMPS_INT nrow
        MUMPS_INT ncol
        MUMPS_INT nnz

        # MUMPS
        ZMUMPS_STRUC_C params

        # internal classes for getters and setters
        mumps_int_array icntl
        mumps_int_array info
        mumps_int_array infog

        zmumps_real_array cntl
        zmumps_real_array rinfo
        zmumps_real_array rinfog

        MUMPS_INT * a_row
        MUMPS_INT * a_col
        ZMUMPS_COMPLEX *  a_val

        bint analyzed
        bint factorized
        bint out_of_core

        object analysis_stats
        object factorize_stats
        object solve_stats

    cdef get_data_pointers(self,
                           MUMPS_INT * a_row,
                           MUMPS_INT * a_col,
                           ZMUMPS_COMPLEX * a_val)

    cdef initialize_mumps_struct(self, comm_fortran, sym)

    cdef mumps_call(self)

    cdef set_centralized_assembled_matrix(self)

    cdef solve_dense(self, ZMUMPS_COMPLEX * rhs,
                     MUMPS_INT rhs_length, MUMPS_INT nrhs)
    cdef solve_sparse(self, MUMPS_INT * rhs_col_ptr, MUMPS_INT * rhs_row_ind,
                      ZMUMPS_COMPLEX * rhs_val,
                      MUMPS_INT rhs_nnz, MUMPS_INT nrhs,
                      ZMUMPS_COMPLEX * x, MUMPS_INT x_length)
