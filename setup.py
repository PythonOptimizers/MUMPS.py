#!/usr/bin/env python

# The file setup.py is automatically generated
# Generate it with
# python generate_code -s

from distutils.core import setup
from setuptools import find_packages
from distutils.extension import Extension

import numpy as np

import ConfigParser
import os
import copy

from codecs import open
from os import path

# HELPERS
#--------
def prepare_Cython_extensions_as_C_extensions(extensions):
    """
    Modify the list of sources to transform `Cython` extensions into `C` extensions.
    Args:
        extensions: A list of (`Cython`) `distutils` extensions.
    Warning:
        The extensions are changed in place. This function is not compatible with `C++` code.
    Note:
        Only `Cython` source files are modified into their `C` equivalent source files. Other file types are unchanged.
    """
    for extension in extensions:
        c_sources = list()
        for source_path in extension.sources:
            path, source = os.path.split(source_path)
            filename, ext = os.path.splitext(source)

            if ext == '.pyx':
                c_sources.append(os.path.join(path, filename + '.c'))
            elif ext in ['.pxd', '.pxi']:
                pass
            else:
                # copy source as is
                c_sources.append(source_path)

        # modify extension in place
        extension.sources = c_sources

mumps_config = ConfigParser.SafeConfigParser()
mumps_config.read('site.cfg')

version = {}
with open("mumps/version.py") as fp:
      exec(fp.read(), version)
# later on we use: version['version']

numpy_include = np.get_include()

# Use Cython?
use_cython = mumps_config.getboolean('CODE_GENERATION', 'use_cython')
if use_cython:
    try:
        from Cython.Distutils import build_ext
        from Cython.Build import cythonize
    except ImportError:
        raise ImportError("Check '%s': Cython is not properly installed." % mumps_config_file)

# DEFAULT
default_include_dir = mumps_config.get('DEFAULT', 'include_dirs').split(os.pathsep)
default_library_dir = mumps_config.get('DEFAULT', 'library_dirs').split(os.pathsep)

# mumps
mumps_compiled_in_64bits = mumps_config.getboolean('MUMPS', 'mumps_compiled_in_64bits')

# Debug mode
use_debug_symbols = mumps_config.getboolean('CODE_GENERATION', 'use_debug_symbols')

# find user defined directories
mumps_include_dirs = mumps_config.get('MUMPS', 'include_dirs').split(os.pathsep)
if mumps_include_dirs == '':
    mumps_include_dirs = default_include_dir
mumps_library_dirs = mumps_config.get('MUMPS', 'library_dirs').split(os.pathsep)
if mumps_library_dirs == '':
    mumps_library_dirs = default_library_dir
           
# OPTIONAL
build_cysparse_ext = False           
if mumps_config.has_section('CYSPARSE'):
    build_cysparse_ext = True
    cysparse_rootdir = mumps_config.get('CYSPARSE', 'cysparse_rootdir').split(os.pathsep)
    if cysparse_rootdir == '':
        raise ValueError("You must specify where CySparse source code is" +
                         "located. Use `cysparse_rootdir` to specify its path.")


########################################################################################################################
# EXTENSIONS
########################################################################################################################
include_dirs = [numpy_include, '.']

ext_params = {}
ext_params['include_dirs'] = include_dirs
if not use_debug_symbols:
    ext_params['extra_compile_args'] = ["-O2", '-std=c99', '-Wno-unused-function']
    ext_params['extra_link_args'] = []
else:
    ext_params['extra_compile_args'] = ["-g", '-std=c99', '-Wno-unused-function']
    ext_params['extra_link_args'] = ["-g"]

context_ext_params = copy.deepcopy(ext_params)
mumps_ext = []
base_ext_params_INT32_COMPLEX64 = copy.deepcopy(ext_params)
base_ext_params_INT32_COMPLEX64['include_dirs'].extend(mumps_include_dirs)
base_ext_params_INT32_COMPLEX64['library_dirs'] = mumps_library_dirs
base_ext_params_INT32_COMPLEX64['libraries'] = [] # 'scalapack', 'pord']
base_ext_params_INT32_COMPLEX64['libraries'].append('cmumps')
base_ext_params_INT32_COMPLEX64['libraries'].append('mumps_common')
base_ext_params_INT32_COMPLEX64['libraries'].append('pord')
base_ext_params_INT32_COMPLEX64['libraries'].append('mpiseq')
base_ext_params_INT32_COMPLEX64['libraries'].append('blas')
base_ext_params_INT32_COMPLEX64['libraries'].append('pthread')

mumps_ext.append(Extension(name="mumps.src.mumps_INT32_COMPLEX64",
                sources=['mumps/src/mumps_INT32_COMPLEX64.pxd',
                'mumps/src/mumps_INT32_COMPLEX64.pyx'],
                **base_ext_params_INT32_COMPLEX64))

numpy_ext_params_INT32_COMPLEX64 = copy.deepcopy(ext_params)
numpy_ext_params_INT32_COMPLEX64['include_dirs'].extend(mumps_include_dirs)
mumps_ext.append(Extension(name="mumps.src.numpy_mumps_INT32_COMPLEX64",
                 sources=['mumps/src/numpy_mumps_INT32_COMPLEX64.pxd',
                 'mumps/src/numpy_mumps_INT32_COMPLEX64.pyx'],
                 **numpy_ext_params_INT32_COMPLEX64))

base_ext_params_INT32_COMPLEX128 = copy.deepcopy(ext_params)
base_ext_params_INT32_COMPLEX128['include_dirs'].extend(mumps_include_dirs)
base_ext_params_INT32_COMPLEX128['library_dirs'] = mumps_library_dirs
base_ext_params_INT32_COMPLEX128['libraries'] = [] # 'scalapack', 'pord']
base_ext_params_INT32_COMPLEX128['libraries'].append('zmumps')
base_ext_params_INT32_COMPLEX128['libraries'].append('mumps_common')
base_ext_params_INT32_COMPLEX128['libraries'].append('pord')
base_ext_params_INT32_COMPLEX128['libraries'].append('mpiseq')
base_ext_params_INT32_COMPLEX128['libraries'].append('blas')
base_ext_params_INT32_COMPLEX128['libraries'].append('pthread')

mumps_ext.append(Extension(name="mumps.src.mumps_INT32_COMPLEX128",
                sources=['mumps/src/mumps_INT32_COMPLEX128.pxd',
                'mumps/src/mumps_INT32_COMPLEX128.pyx'],
                **base_ext_params_INT32_COMPLEX128))

numpy_ext_params_INT32_COMPLEX128 = copy.deepcopy(ext_params)
numpy_ext_params_INT32_COMPLEX128['include_dirs'].extend(mumps_include_dirs)
mumps_ext.append(Extension(name="mumps.src.numpy_mumps_INT32_COMPLEX128",
                 sources=['mumps/src/numpy_mumps_INT32_COMPLEX128.pxd',
                 'mumps/src/numpy_mumps_INT32_COMPLEX128.pyx'],
                 **numpy_ext_params_INT32_COMPLEX128))

base_ext_params_INT32_FLOAT32 = copy.deepcopy(ext_params)
base_ext_params_INT32_FLOAT32['include_dirs'].extend(mumps_include_dirs)
base_ext_params_INT32_FLOAT32['library_dirs'] = mumps_library_dirs
base_ext_params_INT32_FLOAT32['libraries'] = [] # 'scalapack', 'pord']
base_ext_params_INT32_FLOAT32['libraries'].append('smumps')
base_ext_params_INT32_FLOAT32['libraries'].append('mumps_common')
base_ext_params_INT32_FLOAT32['libraries'].append('pord')
base_ext_params_INT32_FLOAT32['libraries'].append('mpiseq')
base_ext_params_INT32_FLOAT32['libraries'].append('blas')
base_ext_params_INT32_FLOAT32['libraries'].append('pthread')

mumps_ext.append(Extension(name="mumps.src.mumps_INT32_FLOAT32",
                sources=['mumps/src/mumps_INT32_FLOAT32.pxd',
                'mumps/src/mumps_INT32_FLOAT32.pyx'],
                **base_ext_params_INT32_FLOAT32))

numpy_ext_params_INT32_FLOAT32 = copy.deepcopy(ext_params)
numpy_ext_params_INT32_FLOAT32['include_dirs'].extend(mumps_include_dirs)
mumps_ext.append(Extension(name="mumps.src.numpy_mumps_INT32_FLOAT32",
                 sources=['mumps/src/numpy_mumps_INT32_FLOAT32.pxd',
                 'mumps/src/numpy_mumps_INT32_FLOAT32.pyx'],
                 **numpy_ext_params_INT32_FLOAT32))

base_ext_params_INT32_FLOAT64 = copy.deepcopy(ext_params)
base_ext_params_INT32_FLOAT64['include_dirs'].extend(mumps_include_dirs)
base_ext_params_INT32_FLOAT64['library_dirs'] = mumps_library_dirs
base_ext_params_INT32_FLOAT64['libraries'] = [] # 'scalapack', 'pord']
base_ext_params_INT32_FLOAT64['libraries'].append('dmumps')
base_ext_params_INT32_FLOAT64['libraries'].append('mumps_common')
base_ext_params_INT32_FLOAT64['libraries'].append('pord')
base_ext_params_INT32_FLOAT64['libraries'].append('mpiseq')
base_ext_params_INT32_FLOAT64['libraries'].append('blas')
base_ext_params_INT32_FLOAT64['libraries'].append('pthread')

mumps_ext.append(Extension(name="mumps.src.mumps_INT32_FLOAT64",
                sources=['mumps/src/mumps_INT32_FLOAT64.pxd',
                'mumps/src/mumps_INT32_FLOAT64.pyx'],
                **base_ext_params_INT32_FLOAT64))

numpy_ext_params_INT32_FLOAT64 = copy.deepcopy(ext_params)
numpy_ext_params_INT32_FLOAT64['include_dirs'].extend(mumps_include_dirs)
mumps_ext.append(Extension(name="mumps.src.numpy_mumps_INT32_FLOAT64",
                 sources=['mumps/src/numpy_mumps_INT32_FLOAT64.pxd',
                 'mumps/src/numpy_mumps_INT32_FLOAT64.pyx'],
                 **numpy_ext_params_INT32_FLOAT64))


if build_cysparse_ext:
    cysparse_ext_params_INT32_COMPLEX64 = copy.deepcopy(ext_params)
    cysparse_ext_params_INT32_COMPLEX64['include_dirs'].extend(cysparse_rootdir)
    cysparse_ext_params_INT32_COMPLEX64['include_dirs'].extend(mumps_include_dirs)
    mumps_ext.append(Extension(name="mumps.src.cysparse_mumps_INT32_COMPLEX64",
                 sources=['mumps/src/cysparse_mumps_INT32_COMPLEX64.pxd',
                 'mumps/src/cysparse_mumps_INT32_COMPLEX64.pyx'],
                 **cysparse_ext_params_INT32_COMPLEX64))

    cysparse_ext_params_INT32_COMPLEX128 = copy.deepcopy(ext_params)
    cysparse_ext_params_INT32_COMPLEX128['include_dirs'].extend(cysparse_rootdir)
    cysparse_ext_params_INT32_COMPLEX128['include_dirs'].extend(mumps_include_dirs)
    mumps_ext.append(Extension(name="mumps.src.cysparse_mumps_INT32_COMPLEX128",
                 sources=['mumps/src/cysparse_mumps_INT32_COMPLEX128.pxd',
                 'mumps/src/cysparse_mumps_INT32_COMPLEX128.pyx'],
                 **cysparse_ext_params_INT32_COMPLEX128))

    cysparse_ext_params_INT32_FLOAT32 = copy.deepcopy(ext_params)
    cysparse_ext_params_INT32_FLOAT32['include_dirs'].extend(cysparse_rootdir)
    cysparse_ext_params_INT32_FLOAT32['include_dirs'].extend(mumps_include_dirs)
    mumps_ext.append(Extension(name="mumps.src.cysparse_mumps_INT32_FLOAT32",
                 sources=['mumps/src/cysparse_mumps_INT32_FLOAT32.pxd',
                 'mumps/src/cysparse_mumps_INT32_FLOAT32.pyx'],
                 **cysparse_ext_params_INT32_FLOAT32))

    cysparse_ext_params_INT32_FLOAT64 = copy.deepcopy(ext_params)
    cysparse_ext_params_INT32_FLOAT64['include_dirs'].extend(cysparse_rootdir)
    cysparse_ext_params_INT32_FLOAT64['include_dirs'].extend(mumps_include_dirs)
    mumps_ext.append(Extension(name="mumps.src.cysparse_mumps_INT32_FLOAT64",
                 sources=['mumps/src/cysparse_mumps_INT32_FLOAT64.pxd',
                 'mumps/src/cysparse_mumps_INT32_FLOAT64.pyx'],
                 **cysparse_ext_params_INT32_FLOAT64))



packages_list = ['mumps', 'mumps.src', 'tests']


# PACKAGE PREPARATION FOR EXCLUSIVE C EXTENSIONS
########################################################################################################################
# We only use the C files **without** Cython. In fact, Cython doesn't need to be installed.
if not use_cython:
    prepare_Cython_extensions_as_C_extensions(mumps_ext)

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: Python
Programming Language :: Cython
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS :: MacOS X
Natural Language :: English
"""

here = path.abspath(path.dirname(__file__))
# Get the long description from the relevant file
with open(path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup_args = {
      'name' : 'MUMPS.py',
      'version' : version['version'],
      'description' : 'A python interface to MUMPS.',
      'long_description' : long_description,
      #Author details

      'author' : 'Sylvain Arreckx, Dominique Orban and Nikolaj van Omme',
      'maintainer' : "Sylvain Arreckx",

      'maintainer_email' : "sylvain.arreckx@gmail.com",
      'summary' : "A Cython/Python interface to the MUMPS solver.",
      'url' : "https://github.com/PythonOptimizers/MUMPS.py.git",
      'download_url' : "https://github.com/PythonOptimizers/MUMPS.py.git",
      'license' : 'LGPL',
      'classifiers' : filter(None, CLASSIFIERS.split('\n')),
      'install_requires' : ['numpy'],
      'ext_modules' : mumps_ext,
      'package_dir' : {"mumps": "mumps"},
      'packages' : packages_list,
      'zip_safe' : False}

if use_cython:
    setup_args['cmdclass'] = {'build_ext': build_ext}

setup(**setup_args)