#!/usr/bin/env python
#################################################################################################
# This script generates all templated code for CySparse
# It this the single one script to use before Cythonizing the CySparse library.
# This script is NOT automatically called by setup.py
#
# The order of code generation is from the "inside towards the outside":
#
# - first generate the most inner code, i.e. the code that is used inside other code;
# - layer by layer, generate the code that only depends on already action code.
#
# We use this homemade script with the Jinja2 template engine:
# http://jinja.pocoo.org/docs/dev/
#
#################################################################################################

import os
import sys
import glob
import fnmatch

import argparse
import logging
import ConfigParser

from subprocess import call

from jinja2 import Environment, FileSystemLoader

from solid.generate import *
import numpy as np

#################################################################################################
# INIT
#################################################################################################
PATH = os.path.dirname(os.path.abspath(__file__))

LOG_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }


def make_parser():
    """
    Create a comment line argument parser.

    Returns:
        The command line parser.
    """
    parser = argparse.ArgumentParser(description='%s: a code generator' % os.path.basename(sys.argv[0]))
    parser.add_argument("-a", "--all", help="Create all action files", action='store_true', required=False)
    parser.add_argument("-s", "--setup", help="Create setup file", action='store_true', required=False)
    parser.add_argument("-t", "--tests", help="Create generic tests", action='store_true', required=False)
    parser.add_argument("-c", "--clean", help="Clean action files", action='store_true', required=False)
    parser.add_argument("-u", "--untrack", help="Untrack files from git", action='store_true', required=False)
    return parser

# JINJA2 FILTERS
def numpy_to_mumps_type(numpy_type):
    if numpy_type in ['FLOAT32']:
        return 's'
    elif numpy_type in ['FLOAT64']:
        return 'd'
    elif numpy_type in ['COMPLEX64']:
        return 'c'
    elif numpy_type in ['COMPLEX128']:
        return 'z'
    else:
        raise TypeError("Not a recognized Numpy type")

# SETUP FILE
SETUP_FILE = os.path.join(PATH, 'setup.cpy')
SETUP_PY_FILE = os.path.join(PATH, 'setup.py')

MUMPS_DIR = os.path.join(PATH, 'mumps')
MUMPS_TEMPLATE_DIR = os.path.join(MUMPS_DIR, 'src')
MUMPS_FACTORY_METHOD_FILE = os.path.join(MUMPS_DIR, 'mumps_context.cpy')
MUMPS_DECLARATION_FILES = glob.glob(os.path.join(MUMPS_TEMPLATE_DIR, '*.cpd'))
MUMPS_DEFINITION_FILES = glob.glob(os.path.join(MUMPS_TEMPLATE_DIR, '*.cpx'))

# TESTS
TESTS_TEMPLATE_DIR = os.path.join(PATH, 'tests')


if __name__ == "__main__":

    # line arguments
    parser = make_parser()
    arg_options = parser.parse_args()

    # type of platform? 32bits or 64bits?
    is_64bits = sys.maxsize > 2**32
    default_index_type_str = '32bits'
    if is_64bits:
        default_index_type_str = '64bits'

    mumps_config = ConfigParser.SafeConfigParser()
    mumps_config.read('site.cfg')

    # test if compiled lib has been compiled in 64 or 32 bits
    MUMPS_INT = None
    if mumps_config.getboolean('MUMPS', 'mumps_compiled_in_64bits'):
        MUMPS_INT = 'INT64'
    else:
        MUMPS_INT = 'INT32'

    INDEX_TYPES = [MUMPS_INT]
    ELEMENT_TYPES = ['FLOAT32', 'FLOAT64', 'COMPLEX64', 'COMPLEX128']

    GENERAL_CONTEXT = {'mumps_index_list': INDEX_TYPES,
                       'mumps_type_list': ELEMENT_TYPES}

    GENERAL_ENVIRONMENT = Environment(
        autoescape=False,
        loader=FileSystemLoader('/'), # we use absolute filenames
        trim_blocks=True,
        lstrip_blocks=True,
        variable_start_string='@',
        variable_end_string='@')

    GENERAL_ENVIRONMENT.filters['numpy_to_mumps_type'] = numpy_to_mumps_type

    # create logger
    logger_name = mumps_config.get('CODE_GENERATION', 'log_name')
    if logger_name == '':
        logger_name = 'mumps_generator'

    logger = logging.getLogger(logger_name)

    # levels
    log_level = LOG_LEVELS[mumps_config.get('CODE_GENERATION', 'log_level')]
    console_log_level = LOG_LEVELS[mumps_config.get('CODE_GENERATION', 'console_log_level')]
    file_log_level = LOG_LEVELS[mumps_config.get('CODE_GENERATION', 'file_log_level')]

    logger.setLevel(log_level)

    # create console handler and set logging level
    ch = logging.StreamHandler()
    ch.setLevel(console_log_level)

    # create file handler and set logging level
    log_file_name = logger_name + '.log'
    fh = logging.FileHandler(log_file_name)
    fh.setLevel(file_log_level)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # add formatter to ch and fh
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add ch and fh to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info('*' * 100)
    logger.info('*' * 100)
    logger.info("Start some action(s)")

    action = False

    if arg_options.setup or arg_options.all:
        action = True
        logger.info("Act for setup file")

        if arg_options.clean:
            clean_cython_files(logger, PATH, [SETUP_PY_FILE], untrack=arg_options.untrack)
        else:
            generate_template_files(logger, [SETUP_FILE], GENERAL_ENVIRONMENT, GENERAL_CONTEXT, '.py')


    if arg_options.all:
        action = True
        logger.info("Act for generic contexts")

        if arg_options.clean:
            clean_cython_files(logger, MUMPS_DIR, [MUMPS_FACTORY_METHOD_FILE[:-4] + '.py'], untrack=arg_options.untrack)
            clean_cython_files(logger, MUMPS_TEMPLATE_DIR, untrack=arg_options.untrack)
        else:
            generate_template_files(logger, [MUMPS_FACTORY_METHOD_FILE], GENERAL_ENVIRONMENT, GENERAL_CONTEXT, '.py')
            generate_following_type_and_index(logger, MUMPS_DECLARATION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pxd')
            generate_following_type_and_index(logger, MUMPS_DEFINITION_FILES, GENERAL_ENVIRONMENT, GENERAL_CONTEXT, ELEMENT_TYPES, INDEX_TYPES, '.pyx')

    if not action:
        logger.warning("Nothing has been done...")

    logger.info("Stop some action(s)")
