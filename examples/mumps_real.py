from mumps.mumps_context import MUMPSContext
import numpy as np
import sys

n = 4
A = np.array([[1, 2, 3, 4], [5, 0, 7, 8], [9, 10, 0, 12], [13, 14, 15, 0]], dtype=np.float64)
arow = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
acol = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
aval = np.array([1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 0, 12, 13, 14, 15, 0], dtype=np.float64)

context = MUMPSContext((n, arow, acol, aval, False), verbose=False)

print 'MUMPS version: ', context.version_number

context.analyze()
context.factorize()

e = np.ones(n, dtype=np.float64)
rhs = np.dot(A, e)

x = context.solve(rhs=rhs)
np.testing.assert_almost_equal(x,e)

print "= " * 80

B = np.ones([n, 3], dtype=np.float64)
B[: ,1] = 2 * B[:,1]
B[: ,2] = 3 * B[:,2]
rhs = np.dot(A,B)

x = context.solve(rhs=rhs)
np.testing.assert_almost_equal(x,B)

print "x" * 80
print "Sparse CSC Mutilple RHS example"

acol_csc = np.array([1,5,9,13,17], dtype=np.int32)-1
arow_csc = np.array([1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4], dtype=np.int32)-1
aval_csc = np.array([1,5,9,13,2,0,10,14,3,7,0,15,4,8,12,0], dtype=np.float64)
x = context.solve(rhs_col_ptr=acol_csc, rhs_row_ind=arow_csc, rhs_val=aval_csc)
np.testing.assert_almost_equal(x,np.eye(4))
