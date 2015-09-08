from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types

from mumps.mumps_context import NewMumpsContext

import numpy as np

import sys


A = NewLLSparseMatrix(mm_filename=sys.argv[1], itype=types.INT32_T, dtype=types.FLOAT64_T)

print A


(n, m) = A.shape
e = np.ones(n, 'd')
#rhs = np.zeros(n, 'd')
rhs = A*e

arow, acol, aval = A.find()

context = NewMumpsContext(n, arow, acol, aval, sym=True, verbose=True)

print 'MUMPS version: ', context.version_number

context.analyze()
context.factorize()

x = context.solve(rhs=rhs)
print x

print "= " * 80

B = np.ones([n, 3], "d")
B[: ,1] = 2 * B[:,1]
B[: ,2] = 3 * B[:,2]

rhs = A * B

x = context.solve(rhs=rhs)
print x

print "x" * 80
print "sparse example"


CSC = A.to_csc()

rhs_col_ptr, rhs_row_ind, rhs_val = CSC.get_numpy_arrays()
print rhs_col_ptr, rhs_row_ind, rhs_val
print type(rhs_col_ptr), type(rhs_row_ind), type(rhs_val)

x = context.solve(rhs_col_ptr=rhs_col_ptr, rhs_row_ind=rhs_row_ind, rhs_val=rhs_val)

print x
