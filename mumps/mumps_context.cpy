"""
Factory method to access Mumps.

The python code (this module) is autmotically generated because the code depends
on the compile/architecture configuration.

"""
import numpy as np

{% for index_type in mumps_index_list %}
    {% for element_type in mumps_type_list %}
from mumps.src.mumps_@index_type@_@element_type@ import MumpsContext_@index_type@_@element_type@
    {% endfor %}
{% endfor %}

def NewMumpsContext(n, a_row, a_col, a_val, sym=False, verbose=False):
    """
    Create and return the right Mumps context object.

    Args:
        
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
           return MumpsContext_@index_type@_@element_type@(n, a_row, a_col, a_val, sym=sym, verbose=verbose)
      {% endfor %}
  {% else %}
    elif itype == np.@index_type|lower@:
      {% for element_type in mumps_type_list %}
        {% if element_type == mumps_type_list |first %}
        if dtype == np.@element_type|lower@:
        {% else %}
        elif dtype == np.@element_type|lower@:
        {% endif %}
           return MumpsContext_@index_type@_@element_type@(n, a_row, a_col, a_val, sym=sym, verbose=verbose)
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

    type_error_msg = 'Matrix has an index and/or element type that is incompatible with Mumps\nAllowed types:\n%s' % allowed_types
    raise TypeError(type_error_msg)
