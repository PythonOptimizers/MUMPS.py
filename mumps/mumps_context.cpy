"""
Factory method to access MUMPS.
"""
import numpy as np

{% for index_type in index_list %}
    {% for element_type in type_list %}
from mumps.src.numpy_mumps_@index_type@_@element_type@ import NumpyMUMPSSolver_@index_type@_@element_type@
    {% endfor %}
{% endfor %}

cysparse_installed = False
try:
{% for index_type in index_list %}
    {% for element_type in type_list %}
    from mumps.src.cysparse_mumps_@index_type@_@element_type@ import CySparseMUMPSSolver_@index_type@_@element_type@
    {% endfor %}
    from cysparse.sparse.ll_mat import PyLLSparseMatrix_Check
    from cysparse.common_types.cysparse_types import *
    cysparse_installed = True
{% endfor %}
except:
    pass

allowed_types = '\titype:
{%- for index_name in index_list -%}
    @index_name@
    {%- if index_name != mumps_index_list|last -%}
    ,
    {%- endif -%}
{%- endfor -%}
\n\tdtype:
{%- for element_name in type_list -%}
    @element_name@
    {%- if element_name != mumps_type_list|last -%}
    ,
    {%- endif -%}
{%- endfor -%}
\n'
type_error_msg = 'Matrix has an index and/or element type that is incompatible with MUMPS\nAllowed types:\n%s' % allowed_types

def MUMPSContext(arg1, verbose=False):
    """
    Create and return the right MUMPS context based on the element type
    supplied as input.

    MUMPS ("MUltifrontal Massively Parallel Solver") is a package for solving systems
    of linear equations of the form Ax = b, where A is a square **sparse** matrix that can be
    either unsymmetric, symmetric positive definite, or general symmetric, on distributed
    memory computers. 
    
    MUMPS performs a Gaussian factorization
      A = LU
    where L is a lower triangular matrix and U an upper triangular matrix.

    If the matrix is symmetric then MUMPS performs the factorization
      A = LDL^T 
    where D is block diagonal matrix.
    
    Args:
        n: size of matrix A
        a_row: row indices of non zero elements of A
        a_col: column indices of non zero elements of A
        a_val: values of non zeros elements of A
        sym:   a boolean indicating if A is a symmetric matrix or not
        verbose: a boolean to turn on or off the verbosity of MUMPS
    """
    if isinstance(arg1, tuple):
        if len(arg1) != 5:
            raise ValueError("If a tuple is supplied, it must have 5"+
                             "items: n, a_row, a_col, a_val, sym")
        n = arg1[0]
        a_row = arg1[1]
        a_col = arg1[2]
        a_val = arg1[3]
        nnz = a_row.shape[0]
        assert a_col.shape[0] == nnz
        assert a_val.shape[0] == nnz

        sym = arg1[4]

        itype = a_row.dtype
        dtype = a_val.dtype

{% for index_type in index_list %}
  {% if index_type == index_list |first %}
        if itype == np.@index_type|lower@:
      {% for element_type in type_list %}
        {% if element_type == type_list |first %}
            if dtype == np.@element_type|lower@:
        {% else %}
            elif dtype == np.@element_type|lower@:
        {% endif %}
                solver = NumpyMUMPSSolver_@index_type@_@element_type@(n, nnz, sym=sym, verbose=verbose)
                solver.get_matrix_data(a_row, a_col, a_val)
                return solver
      {% endfor %}
  {% else %}
        elif itype == np.@index_type|lower@:
      {% for element_type in type_list %}
        {% if element_type == type_list |first %}
            if dtype == np.@element_type|lower@:
        {% else %}
            elif dtype == np.@element_type|lower@:
        {% endif %}
                solver = NumpyMUMPSSolver_@index_type@_@element_type@(n, nnz, sym=sym, verbose=verbose)
                solver.get_matrix_data(a_row, a_col, a_val)
                return solver
      {% endfor %}
  {% endif %}
{% endfor %}
        else:
            raise TypeError(type_error_msg)

    elif cysparse_installed:
        if not PyLLSparseMatrix_Check(arg1):
            raise TypeError('arg1 should be a LLSparseMatrix')

        A = arg1
        nnz = A.nnz
        itype = A.itype
        dtype = A.dtype
        n = A.nrow
        sym = A.is_symmetric
        assert A.ncol == n

{% for index_type in index_list %}
    {% if index_type == index_list |first %}
        if itype == @index_type@_T:
    {% for element_type in type_list %}
        {% if element_type == type_list |first %}
            if dtype == @element_type@_T:
        {% else %}
            elif dtype == @element_type@_T:
        {% endif %}
                solver = CySparseMUMPSSolver_@index_type@_@element_type@(n, nnz, sym=sym, verbose=verbose)
                solver.get_matrix_data(A)
                return solver
    {% endfor %}
    {% else %}
        elif itype == @index_type@_T:
    {% for element_type in type_list %}
        {% if element_type == type_list |first %}
            if dtype == @element_type@_T:
        {% else %}
            elif dtype == @element_type@_T:
        {% endif %}
                solver = CySparseMUMPSSolver_@index_type@_@element_type@(n, nnz, sym=sym, verbose=verbose)
                solver.get_matrix_data(A)
                return solver
    {% endfor %}
    {% endif %}
{% endfor %}
        else:
            raise TypeError(type_error_msg)

