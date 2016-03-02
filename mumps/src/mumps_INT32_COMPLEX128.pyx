"""
This is the base class for the interface to MUMPS (http://mumps.enseeiht.fr/index.php?page=home)

"""

"""
Some notes for the maintainers of this library.

Basic working:
--------------

MUMPS working is simple:

- initialize/modify the C struct MUMPS_STRUC_C;
- call mumps_c(MUMPS_STRUC_C *).

For example:

# analyze
MUMPS_STRUC_C.job = 1
mumps_c(&MUMPS_STRUC_C)

# factorize
MUMPS_STRUC_C.job = 2
mumps_c(&MUMPS_STRUC_C)

# solve
MUMPS_STRUC_C.job = 3
mumps_c(&MUMPS_STRUC_C)

etc.

Access to:
 - icntl
 - info
 - infog
 - cntl
 - rinfo
 - rinfog

**must** be done through properties!!!!


Typed C struct:
---------------

Each MUMPS_STRUC_C is specialized and prefixed by a letter:

- SMUMPS_STRUC_C: simple precision;
- DMUMPS_STRUC_C: double precision;
- CMUMPS_STRUC_C: simple complex;
- ZMUMPS_STRUC_C: double complex.

In MUMPSContext_INT32_COMPLEX128, (S,D,C,Z)mumps_c() is called by self.mumps_call().

<ZMUMPS_COMPLEX *> can be used for **ALL** four types.

Solve:
------

MUMPS **overwrites** the rhs member and replaces it by the solution(s) it finds.
If sparse solve is used, the solution is placed in a dummy dense rhs member.

The rhs member can be a matrix or a vector.

1-based index arrays:
---------------------

MUMPS uses exclusively FORTRAN routines and by consequence **all** array indices start with index **1** (not 0).

Default 32 bit integers compilation:
------------------------------------

By default, MUMPS is compiled in 32 bit integers **unless** it is compiled with the option -DINTSIZE64.
32 and 64 bit versions are **not** compatible.

Lib creation:
-------------

The :file:`libmpiseq.so` file is *missing* by default in lib and must be added by hand. It is compiled in directory
libseq. :file:`libmpiseq.so` is essentially a dummy file to deal with sequential code.

The order in which the library (.so) files are given to construct the MUMPS part
of this interface **is** important... and not standard.

"""

from mumps.mumps_statistics import AnalysisStatistics, FactorizationStatistics, SolveStatistics

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython cimport Py_INCREF, Py_DECREF

from libc.stdint cimport int64_t
from libc.string cimport strncpy

import numpy as np
cimport numpy as cnp

cnp.import_array()

import time

cdef extern from "mumps_c_types.h":

    ctypedef int        MUMPS_INT
    ctypedef int64_t    MUMPS_INT8 # warning: mumps uses "stdint.h" which might define int64_t as long long...

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

# MUMPS possible ordering methods
orderings = { 'amd' : 0, 'amf' : 2, 'scotch' : 3, 'pord' : 4, 'metis' : 5,
              'qamd' : 6, 'auto' : 7 }

ordering_name = [ 'amd', 'user-defined', 'amf',
                  'scotch', 'pord', 'metis', 'qamd']

# MUMPS ERRORS
# TODO: decouple
error_messages = {
    -5 : "Not enough memory during analysis phase",
    -6 : "Matrix is singular in structure",
    -7 : "Not enough memory during analysis phase",
    -10 : "Matrix is numerically singular",
    -11 : "The authors of MUMPS would like to hear about this",
    -12 : "The authors of MUMPS would like to hear about this",
    -13 : "Not enough memory"
}


class MUMPSError(RuntimeError):
    def __init__(self, infog):
        self.error = infog[1]
        if self.error in error_messages:
            msg = "{}. (MUMPS error {})".format(
                error_messages[self.error], self.error)
        else:
            msg = "MUMPS failed with error {}.".format(self.error)

        RuntimeError.__init__(self, msg)

# MUMPS HELPERS
cdef class mumps_int_array:
    """
    Internal classes to use x[i] = value and x[i] setters and getters

    Integer version.

    """
    def __cinit__(self):
        pass

    cdef get_array(self, MUMPS_INT * array, int ub = 40):
        """
        Args:
            ub: upper bound.
        """
        self.ub = ub
        self.array = array

    def __getitem__(self, key):
        if key < 1:
            raise IndexError('MUMPS index must be >= 1 (Fortran style)')
        if key > self.ub:
            raise IndexError('MUMPS index must be <= %d' % self.ub)

        return self.array[key - 1]

    def __setitem__(self, key, value):
        self.array[key - 1] = value

cdef class zmumps_real_array:
    """
    Internal classes to use x[i] = value and x[i] setters and getters

    Real version.

    """
    def __cinit__(self):
        pass

    cdef get_array(self, ZMUMPS_REAL * array, int ub = 40):
        """
        Args:
            ub: upper bound.
        """
        self.ub = ub
        self.array = array

    def __getitem__(self, key):
        if key < 1:
            raise IndexError('MUMPS index must be >= 1 (Fortran style)')
        if key > self.ub:
            raise IndexError('MUMPS index must be <= %d' % self.ub)

        return self.array[key - 1]

    def __setitem__(self, key, value):
        self.array[key - 1] = value



cdef c_to_fortran_index_array(MUMPS_INT * a, MUMPS_INT a_size):
    cdef:
        MUMPS_INT i

    for i from 0 <= i < a_size:
        a[i] += 1

# MUMPS CONTEXT
cdef class BaseMUMPSSolver_INT32_COMPLEX128:
    """
    Base MUMPS Context.

    This version **only** deals with array pointers.

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
        """
        self.nrow = n
        self.ncol = n
        self.nnz = nnz

        assert self.ncol == self.nrow

        self.initialize_mumps_struct(comm_fortran, sym)

        # `initialize_mumps_struct` **must** be called before assigning
        # a value to n (size of the matrix)
        self.params.n = <MUMPS_INT> self.nrow

        self.analyzed = False
        self.factorized = False
        self.out_of_core = False

        if not verbose:
            self.set_silent()


    cdef initialize_mumps_struct(self, comm_fortran, sym):
        """
        Initialize MUMPS structure and make control parameters and information
        avalaible to user.

        Args:
            comm_fortran: MPI communicator
            sym: a boolean indicating if A is a symmetric matrix or not
        """
        self.params.job = -1
        self.params.sym = sym
        self.params.par = 1

        self.params.comm_fortran = comm_fortran

        self.mumps_call()

        # integer control parameters
        self.icntl = mumps_int_array()
        self.icntl.get_array(self.params.icntl)

        # integer information parameters
        self.info = mumps_int_array()
        self.info.get_array(self.params.info)

        # integer information parameters
        self.infog = mumps_int_array()
        self.infog.get_array(self.params.infog)

        # real/complex control parameters
        self.cntl = zmumps_real_array()
        self.cntl.get_array(self.params.cntl)

        # real/complex information parameters
        self.rinfo = zmumps_real_array()
        self.rinfo.get_array(self.params.rinfo)

        # real/complex information parameters
        self.rinfog = zmumps_real_array()
        self.rinfog.get_array(self.params.rinfog)


    cdef get_data_pointers(self,
                           MUMPS_INT * a_row,
                           MUMPS_INT * a_col,
                           ZMUMPS_COMPLEX * a_val):
        """
        Get elements of A and their positions and transfer them to MUMPS internal structure.
        
        Args:
            a_row: pointer to an MUMPS_INT array containing row indices of non zero elements of A.
            a_col: pointer to an MUMPS_INT array containing column indices of non zero elements of A.
            a_val: pointer to an MUMPS_INT array containing values of non zeros elements of A.

        Note: row and column indices are adjusted for Fortran indexing.
        """

        self.a_row = <MUMPS_INT *> a_row
        self.a_col = <MUMPS_INT *> a_col
        self.a_val = <ZMUMPS_COMPLEX *> a_val

        # transform c index arrays to fortran arrays
        c_to_fortran_index_array(self.a_row, self.nnz)
        c_to_fortran_index_array(self.a_col, self.nnz)

        self.set_centralized_assembled_matrix()


    cdef set_centralized_assembled_matrix(self):
        """
        Set the centralized assembled matrix
        The rank 0 process supplies the entire matrix.
        """

        self.params.nz = <MUMPS_INT> self.nnz

        self.params.irn = <MUMPS_INT *> self.a_row
        self.params.jcn = <MUMPS_INT *> self.a_col
        self.params.a = <ZMUMPS_COMPLEX *> self.a_val


    def __dealloc__(self):
        # autodestruct mumps internal
        self.params.job = -2
        self.mumps_call()
        self.params.job = -1
        self.mumps_call()

    # Properties
    # COMMON Properties
    property analyzed:
        def __get__(self): return self.analyzed
    property factorized:
        def __get__(self): return self.factorized
    property sym:
        def __get__(self): return self.params.sym
        def __set__(self, value): self.params.sym = value
    property par:
        def __get__(self): return self.params.par
        def __set__(self, value): self.params.par = value
    property job:
        def __get__(self): return self.params.job
        def __set__(self, value): self.params.job = value

    property comm_fortran:
        def __get__(self): return self.params.comm_fortran
        def __set__(self, value): self.params.comm_fortran = value

    property icntl:
        def __get__(self):
            return self.icntl

    property n:
        def __get__(self): return self.params.n
        def __set__(self, value): self.params.n = value
    property nz_alloc:
        def __get__(self): return self.params.nz_alloc
        def __set__(self, value): self.params.nz_alloc = value

    property nz:
        def __get__(self): return self.params.nz
        def __set__(self, value): self.params.nz = value
    property irn:
        def __get__(self): return <long> self.params.irn
        def __set__(self, long value): self.params.irn = <MUMPS_INT*> value
    property jcn:
        def __get__(self): return <long> self.params.jcn
        def __set__(self, long value): self.params.jcn = <MUMPS_INT*> value

    property nz_loc:
        def __get__(self): return self.params.nz_loc
        def __set__(self, value): self.params.nz_loc = value
    property irn_loc:
        def __get__(self): return <long> self.params.irn_loc
        def __set__(self, long value): self.params.irn_loc = <MUMPS_INT*> value
    property jcn_loc:
        def __get__(self): return <long> self.params.jcn_loc
        def __set__(self, long value): self.params.jcn_loc = <MUMPS_INT*> value

    property nelt:
        def __get__(self): return self.params.nelt
        def __set__(self, value): self.params.nelt = value
    property eltptr:
        def __get__(self): return <long> self.params.eltptr
        def __set__(self, long value): self.params.eltptr = <MUMPS_INT*> value
    property eltvar:
        def __get__(self): return <long> self.params.eltvar
        def __set__(self, long value): self.params.eltvar = <MUMPS_INT*> value

    property perm_in:
        def __get__(self): return <long> self.params.perm_in
        def __set__(self, long value): self.params.perm_in = <MUMPS_INT*> value

    property sym_perm:
        def __get__(self): return <long> self.params.sym_perm
        def __set__(self, long value): self.params.sym_perm = <MUMPS_INT*> value
    property uns_perm:
        def __get__(self): return <long> self.params.uns_perm
        def __set__(self, long value): self.params.uns_perm = <MUMPS_INT*> value

    property irhs_sparse:
        def __get__(self): return <long> self.params.irhs_sparse
        def __set__(self, long value): self.params.irhs_sparse = <MUMPS_INT*> value
    property irhs_ptr:
        def __get__(self): return <long> self.params.irhs_ptr
        def __set__(self, long value): self.params.irhs_ptr = <MUMPS_INT*> value
    property isol_loc:
        def __get__(self): return <long> self.params.isol_loc
        def __set__(self, long value): self.params.isol_loc = <MUMPS_INT*> value

    property nrhs:
        def __get__(self): return self.params.nrhs
        def __set__(self, value): self.params.nrhs = value
    property lrhs:
        def __get__(self): return self.params.lrhs
        def __set__(self, value): self.params.lrhs = value
    property lredrhs:
        def __get__(self): return self.params.lredrhs
        def __set__(self, value): self.params.lredrhs = value
    property nz_rhs:
        def __get__(self): return self.params.nz_rhs
        def __set__(self, value): self.params.nz_rhs = value
    property lsol_loc:
        def __get__(self): return self.params.lsol_loc
        def __set__(self, value): self.params.lsol_loc = value

    property schur_mloc:
        def __get__(self): return self.params.schur_mloc
        def __set__(self, value): self.params.schur_mloc = value
    property schur_nloc:
        def __get__(self): return self.params.schur_nloc
        def __set__(self, value): self.params.schur_nloc = value
    property schur_lld:
        def __get__(self): return self.params.schur_lld
        def __set__(self, value): self.params.schur_lld = value

    property mblock:
        def __get__(self): return self.params.mblock
        def __set__(self, value): self.params.mblock = value
    property nblock:
        def __get__(self): return self.params.nblock
        def __set__(self, value): self.params.nblock = value
    property nprow:
        def __get__(self): return self.params.nprow
        def __set__(self, value): self.params.nprow = value
    property npcol:
        def __get__(self): return self.params.npcol
        def __set__(self, value): self.params.npcol = value

    property info:
        def __get__(self):
            return self.info

    property infog:
        def __get__(self):
            return self.infog

    property deficiency:
        def __get__(self): return self.params.deficiency
        def __set__(self, value): self.params.deficiency = value
    property pivnul_list:
        def __get__(self): return <long> self.params.pivnul_list
        def __set__(self, long value): self.params.pivnul_list = <MUMPS_INT*> value
    property mapping:
        def __get__(self): return <long> self.params.mapping
        def __set__(self, long value): self.params.mapping = <MUMPS_INT*> value

    property size_schur:
        def __get__(self): return self.params.size_schur
        def __set__(self, value): self.params.size_schur = value
    property listvar_schur:
        def __get__(self): return <long> self.params.listvar_schur
        def __set__(self, long value): self.params.listvar_schur = <MUMPS_INT*> value

    property instance_number:
        def __get__(self): return self.params.instance_number
        def __set__(self, value): self.params.instance_number = value

    property version_number:
        def __get__(self):
            return (<bytes> self.params.version_number).decode('ascii')

    property ooc_tmpdir:
        def __get__(self):
            return (<bytes> self.params.ooc_tmpdir).decode('ascii')
        def __set__(self, char *value):
            strncpy(self.params.ooc_tmpdir, value, sizeof(self.params.ooc_tmpdir))
    property ooc_prefix:
        def __get__(self):
            return (<bytes> self.params.ooc_prefix).decode('ascii')
        def __set__(self, char *value):
            strncpy(self.params.ooc_prefix, value, sizeof(self.params.ooc_prefix))

    property write_problem:
        def __get__(self):
            return (<bytes> self.params.write_problem).decode('ascii')
        def __set__(self, char *value):
            strncpy(self.params.write_problem, value, sizeof(self.params.write_problem))

    property lwk_user:
        def __get__(self): return self.params.lwk_user
        def __set__(self, value): self.params.lwk_user = value

    # TYPED Properties
    property cntl:
        def __get__(self):
            return self.cntl

    property a:
        def __get__(self): return <long> self.params.a
        def __set__(self, long value): self.params.a = <ZMUMPS_COMPLEX*> value

    property a_loc:
        def __get__(self): return <long> self.params.a_loc
        def __set__(self, long value): self.params.a_loc = <ZMUMPS_COMPLEX*> value

    property a_elt:
        def __get__(self): return <long> self.params.a_elt
        def __set__(self, long value): self.params.a_elt = <ZMUMPS_COMPLEX*> value

    property colsca:
        def __get__(self): return <long> self.params.colsca
        def __set__(self, long value): self.params.colsca = <ZMUMPS_REAL*> value
    property rowsca:
        def __get__(self): return <long> self.params.rowsca
        def __set__(self, long value): self.params.rowsca = <ZMUMPS_REAL*> value

    property rhs:
        def __get__(self): return <long> self.params.rhs
        def __set__(self, long value): self.params.rhs = <ZMUMPS_COMPLEX*> value
    property redrhs:
        def __get__(self): return <long> self.params.redrhs
        def __set__(self, long value): self.params.redrhs = <ZMUMPS_COMPLEX*> value
    property rhs_sparse:
        def __get__(self): return <long> self.params.rhs_sparse
        def __set__(self, long value): self.params.rhs_sparse = <ZMUMPS_COMPLEX*> value
    property sol_loc:
        def __get__(self): return <long> self.params.sol_loc
        def __set__(self, long value): self.params.sol_loc = <ZMUMPS_COMPLEX*> value

    property rinfo:
        def __get__(self):
            return self.rinfo

    property rinfog:
        def __get__(self):
            return self.rinfog

    property schur:
        def __get__(self): return <long> self.params.schur
        def __set__(self, long value): self.params.schur = <ZMUMPS_COMPLEX*> value

    property wk_user:
        def __get__(self): return <long> self.params.wk_user
        def __set__(self, long value): self.params.wk_user = <ZMUMPS_COMPLEX*> value

    # MUMPS CALL
    cdef mumps_call(self):
        """
        Call to Xmumps_c(XMUMPS_STRUC_C).
        """
        zmumps_c(&self.params)


    def set_silent(self):
        """
        Silence **all* MUMPS output.

        See MUMPS documentation.
        """
        self.icntl[1] = 0
        self.icntl[2] = 0
        self.icntl[3] = 0
        self.icntl[4] = 0


    def analyze(self, ordering='auto'):
        """
        Performs analysis step of MUMPS.

        In the analyis step, MUMPS is able to figure out a reordering for the 
        given matrix. It does so if `ordering` is set to 'auto'.
        If not MUMPS will use the provided ordering.
        MUMPS statistics for the analysis are available in `analysis_stats`.

        Args:
            ordering : { 'auto', 'amd', 'amf', 'scotch', 'pord', 'metis', 'qamd' }
                ordering to use in the factorization. The availability of a
                particular ordering depends on the MUMPS installation.  Default is
                'auto'.
        """
        if self.analyzed:
            return

        self.icntl[7] = orderings[ordering]
        t1 = time.clock()
        self.params.job = 1   # analyse
        self.mumps_call()
        t2 = time.clock()

        if self.infog[1] < 0:
            raise MUMPSError(self.infog)

        self.analyzed = True

        # self.analysis_stats = AnalysisStatistics(self.params,
        #                                          t2 - t1)

    def factorize(self, ordering='auto', pivot_tol=0.01):
        """
        Perform the LU factorization of the matrix (or LDL' if the matrix
        is symmetric).

        This factorization can then later be used to solve a linear system
        with `solve`. Statistical data of the factorization is stored in
        `factor_stats`.

        Args:
            ordering : { 'auto', 'amd', 'amf', 'scotch', 'pord', 'metis', 'qamd' }
                ordering to use in the factorization. The availability of a
                particular ordering depends on the MUMPS installation.  Default is
                'auto'.
            pivot_tol: number in the range [0, 1]
                pivoting threshold. Pivoting is typically limited in sparse
                solvers, as too much pivoting destroys sparsity. 1.0 means full
                pivoting, whereas 0.0 means no pivoting. Default is 0.01.
        """
        # TODO: ordering

        # Analysis phase must be done before factorization
        if not self.analyzed :
            self.analyze(ordering=ordering)

        self.icntl[22] = 1 if self.out_of_core else 1
        self.cntl[1] = pivot_tol
        self.params.job = 2

        done = False
        while not done:
            t1 = time.clock()
            self.mumps_call()
            t2 = time.clock()

            # error -8, -9 (not enough allocated memory) is treated
            # specially, by increasing the memory relaxation parameter
            if self.infog[1] < 0:
                if self.infog[1] in (-8, -9):
                    # double the additional memory
                    self.icntl[14] = self.icntl[14]*2
                else:
                    raise MUMPSError(self.infog)
            else:
                done = True

        self.factorized = True
        # self.factorize_stats = FactorizationStatistics(self.params, t2 - t1)

    cdef solve_dense(self, ZMUMPS_COMPLEX * rhs, MUMPS_INT rhs_length, MUMPS_INT nrhs):
        """
        Solve a linear system after the LU (or LDL^T) factorization has previously been performed by `factorize`

        Args:
            rhs: the right hand side (dense matrix or vector)
            rhs_length: Length of each column of the ``rhs``.
            nrhs: Number of columns in the matrix ``rhs``.

        Warning:
            MUMPS overwrites ``rhs`` and replaces it by the solution of the linear system.
        """
        self.params.nrhs = <MUMPS_INT> nrhs
        self.params.lrhs = <MUMPS_INT> rhs_length
        self.params.rhs = <ZMUMPS_COMPLEX *>rhs
        self.params.job = 3  # solve
        self.mumps_call()

    cdef solve_sparse(self, MUMPS_INT * rhs_col_ptr, MUMPS_INT * rhs_row_ind,
                      ZMUMPS_COMPLEX * rhs_val, MUMPS_INT rhs_nnz, MUMPS_INT nrhs, 
                      ZMUMPS_COMPLEX * x, MUMPS_INT x_length):
        """
        Solve a linear system after the LU (or LDL^t) factorization has previously been performed by `factorize`

        Args:
            rhs_length: Length of each column of the ``rhs``.
            nrhs: Number of columns in the matrix ``rhs``.
            overwrite_rhs : ``True`` or ``False``
                whether the data in ``rhs`` may be overwritten, which can lead to a small
                performance gain. Default is ``False``.
            x : the solution to the linear system as a dense matrix or vector.
            x_length: ``self.nrow`` (sequential version).

        Warning:
            MUMPS overwrites ``rhs`` and replaces it by the solution of the linear system.

        """
        self.params.nz_rhs = rhs_nnz
        self.params.nrhs = nrhs # nrhs -1 ?

        self.params.rhs_sparse = <ZMUMPS_COMPLEX *> rhs_val
        self.params.irhs_sparse = <MUMPS_INT *> rhs_row_ind
        self.params.irhs_ptr = <MUMPS_INT *> rhs_col_ptr

        # MUMPS places the solution(s) of the linear system in its dense rhs...
        self.params.lrhs = <MUMPS_INT> x_length
        self.params.rhs = <ZMUMPS_COMPLEX *> x

        self.params.job = 3        # solve
        self.icntl[20] = 1  # tell solver rhs is sparse
        self.mumps_call()

    def solve(self, **kwargs):
        """

        Args:
            rhs: dense NumPy array (matrix or vector).
            rhs_col_ptr, rhs_row_ind, rhs_val: sparse NumPy CSC arrays (matrix or vector).
            transpose_solve : ``True`` or ``False`` whether to solve A * x = rhs or A^T * x = rhs. Default is ``False``

        Returns:
            Dense NumPy array ``x`` (matrix or vector) with the solution(s) of the linear system.
        """
        if not self.factorized:
            self.factorize()

        transpose_solve = kwargs.get('transpose_solve', False)
        self.icntl[9] = 2 if transpose_solve else 1

        cdef:
            MUMPS_INT nrhs

        # rhs can be dense or sparse
        if 'rhs' in kwargs:
            rhs = kwargs['rhs']

            if not cnp.PyArray_Check(rhs):
                raise TypeError('rhs dense arrays must be an NumPy array')

            # check is dimensions are OK
            rhs_shape = rhs.shape

            if (rhs_shape[0] != self.nrow):
                raise ValueError("Right hand side has wrong size"
                                 "Attempting to solve the linear system, where A is of size (%d, %d) "
                                 "and rhs is of size (%g)"%(self.nrow, self.nrow, rhs_shape))

            # create x
            x = np.asfortranarray(rhs.copy())

            # test number of columns in rhs
            if rhs.ndim == 1:
                nrhs = 1
            else:
                nrhs = <MUMPS_INT> rhs_shape[1]

            self.solve_dense(<ZMUMPS_COMPLEX *> cnp.PyArray_DATA(x), rhs_shape[0], nrhs)

        elif all(arg in kwargs for arg in ['rhs_col_ptr', 'rhs_row_ind', 'rhs_val']) :

            rhs_col_ptr = kwargs['rhs_col_ptr']
            rhs_row_ind = kwargs['rhs_row_ind']
            rhs_val = kwargs['rhs_val']

            # fortran indices, done internally in C: no efficiency lost
            rhs_col_ptr += 1
            rhs_row_ind += 1

            nrhs = rhs_col_ptr.size - 1
            x_length = self.nrow
            rhs_nnz = rhs_val.size

            x = np.zeros([self.nrow, nrhs], dtype=np.complex128)

            self.solve_sparse(<MUMPS_INT *>cnp.PyArray_DATA(rhs_col_ptr), <MUMPS_INT *>cnp.PyArray_DATA(rhs_row_ind),
                              <ZMUMPS_COMPLEX *> cnp.PyArray_DATA(rhs_val),
                              rhs_nnz, nrhs, <ZMUMPS_COMPLEX *> cnp.PyArray_DATA(x), x_length)
        else:
            raise TypeError('rhs not given in the right format (dense: rhs=..., sparse: rhs_col_ptr=..., rhs_row_ind=..., rhs_val=...)')

        return x

    def refine(self, rhs, nitref = 3, tol=None):
        """
        refine(rhs, nitref) performs iterative refinement if necessary
        until the scaled residual norm ||b-Ax||/(1+||b||) falls below the
        threshold 'tol' or until nitref steps are taken.
        Parameters:
        nitref :  < 0 : Fixed number of steps of iterative refinement. No stopping criterion is used. 
                  0 : No iterative refinement.
                  > 0 : Maximum number of steps of iterative refinement. A stopping criterion is used,
                         therefore a test for convergence is done at each step of the iterative refinement algorithm.
                 Default: 3
        tol :  is the stopping criterion for iterative refinement

        Make sure you have called `solve()` with the same right-hand
        side rhs before calling `refine()`.
        The residual vector self.residual will be updated to reflect
        the updated approximate solution.
        """
        cdef:
            cnp.npy_intp dim[1]

        if not cnp.PyArray_Check(rhs):
            raise TypeError('rhs dense arrays must be an NumPy array')

        # check is dimensions are OK
        rhs_shape = rhs.shape

        if (rhs_shape[0] != self.nrow):
            raise ValueError("Right hand side has wrong size"
                             "Attempting to solve the linear system, where A is of size (%d, %d) "
                             "and rhs is of size (%g)"%(self.nrow, self.nrow, rhs_shape))

        # test number of columns in rhs.
        # Only one rhs is allowed when dense rhs is provided
        if rhs.ndim == 1:
            nrhs = 1
        else:
            raise TypeError("Only one dense rhs is allowed for performing an iterative"+
                            "refinement.")

        # create x
        x = np.asfortranarray(rhs.copy())

        self.icntl[10] = nitref
        if tol is None:
            self.cntl[2] = 0
        else:
            self.cntl[2] = tol

        self.solve_dense(<ZMUMPS_COMPLEX *> cnp.PyArray_DATA(x), rhs_shape[0], nrhs)

        # reset to default values
        self.icntl[10] = 0
        self.cntl[2] = -1

        return x