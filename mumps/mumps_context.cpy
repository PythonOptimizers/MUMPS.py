"""
Factory method to access MUMPS.
           
Note: the python code (this module) is automatically generated because the code depends
on the compile/architecture configuration.
"""
import numpy as np

{% for index_type in mumps_index_list %}
    {% for element_type in mumps_type_list %}
from mumps.src.mumps_@index_type@_@element_type@ import NumpyMUMPSContext_@index_type@_@element_type@
    {% endfor %}
{% endfor %}

def MUMPSContext(n, a_row, a_col, a_val, sym=False, verbose=False):
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
    itype = a_row.dtype
    dtype = a_val.dtype

{% for index_type in mumps_index_list %}
  {% if index_type == mumps_index_list |first %}
    if itype == np.@index_type|lower@:
      {% for element_type in mumps_type_list %}
        {% if element_type == mumps_type_list |first %}
        if dtype == np.@element_type|lower@:
        {% else %}
        elif dtype == np.@element_type|lower@:
        {% endif %}
           return NumpyMUMPSContext_@index_type@_@element_type@(n, a_row, a_col, a_val, sym=sym, verbose=verbose)
      {% endfor %}
  {% else %}
    elif itype == np.@index_type|lower@:
      {% for element_type in mumps_type_list %}
        {% if element_type == mumps_type_list |first %}
        if dtype == np.@element_type|lower@:
        {% else %}
        elif dtype == np.@element_type|lower@:
        {% endif %}
           return NumpyMUMPSContext_@index_type@_@element_type@(n, a_row, a_col, a_val, sym=sym, verbose=verbose)
      {% endfor %}
  {% endif %}
{% endfor %}

    allowed_types = '\titype:
    {%- for index_name in mumps_index_list -%}
       @index_name@
       {%- if index_name != mumps_index_list|last -%}
       ,
       {%- endif -%}
     {%- endfor -%}
     \n\tdtype:
     {%- for element_name in mumps_type_list -%}
       @element_name@
       {%- if element_name != mumps_type_list|last -%}
       ,
       {%- endif -%}
     {%- endfor -%}
     \n'

    type_error_msg = 'Matrix has an index and/or element type that is incompatible with MUMPS\nAllowed types:\n%s' % allowed_types
    raise TypeError(type_error_msg)
