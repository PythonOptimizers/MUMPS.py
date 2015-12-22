#!/usr/bin/env python
################################################################################
# This script generates all templated code for MUMPS.py
# It is the single one script to use before Cythonizing the MUMPS.py library.
# This script is NOT automatically called by setup.py
#
# The order of code generation is from the "inside towards the outside":
#
# - first generate the most inner code, i.e. the code that is used inside other
#   code;
# - layer by layer, generate the code that only depends on already action code.
#
# We use this homemade script with the Jinja2 template engine:
# http://jinja.pocoo.org/docs/dev/
#
################################################################################
from cygenja.generator import Generator
from jinja2 import Environment, FileSystemLoader

import os
import sys
import shutil
import argparse
import logging
import ConfigParser

################################################################################
# INIT
################################################################################
PATH = os.path.dirname(os.path.abspath(__file__))


def make_parser():
    """
    Create a comment line argument parser.

    Returns:
        The command line parser.
    """
    basename = os.path.basename(sys.argv[0])
    parser = argparse.ArgumentParser(description='%s: a code generator' % basename)

    parser.add_argument("-c", "--clean", help="Clean action files",
                        action='store_true', required=False)
    parser.add_argument("-d", "--dry_run", help="Dry run: no action is taken",
                        action='store_true', required=False)
    parser.add_argument("-f", "--force", help="Force generation no matter what",
                        action='store_true', required=False)
    parser.add_argument("-u", "--untrack", help="Untrack files from git",
                        action='store_true', required=False)
    parser.add_argument('dir_pattern', nargs='?', default='.',
                        help='Glob pattern')
    parser.add_argument('file_pattern', nargs='?', default='*.*',
                        help='Fnmatch pattern')
    return parser

# Conditionnal code generation
# ----------------------------
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

COMPLEX_ELEMENT_TYPES = ['COMPLEX64', 'COMPLEX128']
REAL_ELEMENT_TYPES = ['FLOAT32', 'FLOAT64']
ELEMENT_TYPES = COMPLEX_ELEMENT_TYPES + REAL_ELEMENT_TYPES

GENERAL_CONTEXT = {'index_list': INDEX_TYPES,
                   'type_list': ELEMENT_TYPES,
                   'complex_list': COMPLEX_ELEMENT_TYPES}


# ACTION FUNCTION
def single_generation():
    """
    Only generate one file without any suffix.
    """
    yield '', GENERAL_CONTEXT


def generate_following_index_and_element():
    """
    Generate files following the index and element types.
    """
    for index in INDEX_TYPES:
        GENERAL_CONTEXT['index'] = index
        for type in ELEMENT_TYPES:
            GENERAL_CONTEXT['type'] = type
            yield '_%s_%s' % (index, type), GENERAL_CONTEXT


# JINJA2 FILTERS
def generic_to_mumps_type(generic_type):
    if generic_type in ['FLOAT32']:
        return 's'
    elif generic_type in ['FLOAT64']:
        return 'd'
    elif generic_type in ['COMPLEX64']:
        return 'c'
    elif generic_type in ['COMPLEX128']:
        return 'z'
    else:
        raise TypeError("Not a recognized generic type")


def generic_to_c_type(generic_type):
    if generic_type in ['INT32']:
        return 'int'
    elif generic_type in ['INT64']:
        return 'long'
    elif generic_type in ['FLOAT32']:
        return 'float'
    elif generic_type in ['FLOAT64']:
        return 'double'
    elif generic_type in ['COMPLEX64']:
        return 'float complex'
    elif generic_type in ['COMPLEX128']:
        return 'double complex'
    else:
        raise TypeError("Not a recognized generic type")


def generic_to_single_double_type(generic_type):
    if generic_type in ['FLOAT32', 'COMPLEX64']:
        return 'float32'
    elif generic_type in ['FLOAT64', 'COMPLEX128']:
        return 'float64'
    else:
        raise TypeError("Not a recognized generic type")


def generic_to_c_single_double_type(generic_type):
    if generic_type in ['FLOAT32', 'COMPLEX64']:
        return 'float'
    elif generic_type in ['FLOAT64', 'COMPLEX128']:
        return 'double'
    else:
        raise TypeError("Not a recognized generic type")


# JINJA 2 environment
GENERAL_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader('/'),  # we use absolute filenames
    trim_blocks=True,
    lstrip_blocks=True,
    variable_start_string='@',
    variable_end_string='@')


# LOGGER
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL}


def create_logger(mumps_config):
    logger_name = mumps_config.get('CODE_GENERATION', 'log_name')
    if logger_name == '':
        logger_name = 'mumps_generator'

    logger = logging.getLogger(logger_name)

    # levels
    log_level = LOG_LEVELS[mumps_config.get('CODE_GENERATION',
                                            'log_level')]
    console_log_level = LOG_LEVELS[mumps_config.get('CODE_GENERATION',
                                                    'console_log_level')]
    file_log_level = LOG_LEVELS[mumps_config.get('CODE_GENERATION',
                                                 'file_log_level')]

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

    return logger


if __name__ == "__main__":

    # command line arguments
    parser = make_parser()
    arg_options = parser.parse_args()

    # create logger
    logger = create_logger(mumps_config)

    # cygenja engine
    current_directory = os.path.dirname(os.path.abspath(__file__))
    cygenja_engine = Generator(current_directory,
                               GENERAL_ENVIRONMENT,
                               logger=logger)

    # register filters
    cygenja_engine.register_filter('generic_to_mumps_type',
                                   generic_to_mumps_type)
    cygenja_engine.register_filter('generic_to_c_type',
                                   generic_to_c_type)
    cygenja_engine.register_filter('generic_to_c_single_double_type',
                                   generic_to_c_single_double_type)
    cygenja_engine.register_filter('generic_to_single_double_type',
                                   generic_to_single_double_type)

    # register extensions
    cygenja_engine.register_extension('.cpy', '.py')
    cygenja_engine.register_extension('.cpx', '.pyx')
    cygenja_engine.register_extension('.cpd', '.pxd')
    cygenja_engine.register_extension('.cpi', '.pxi')

    # register actions
    cygenja_engine.register_action('config', '*.*', single_generation)
    cygenja_engine.register_action('mumps', '*.*',
                                   single_generation)
    cygenja_engine.register_action('mumps/src', '*.*',
                                   generate_following_index_and_element)

    # Generation
    if arg_options.dry_run:
        cygenja_engine.generate(arg_options.dir_pattern,
                                arg_options.file_pattern,
                                action_ch='d',
                                recursively=True,
                                force=arg_options.force)
    elif arg_options.clean:
        cygenja_engine.generate(arg_options.dir_pattern,
                                arg_options.file_pattern,
                                action_ch='c',
                                recursively=True,
                                force=arg_options.force)
    else:
        cygenja_engine.generate(arg_options.dir_pattern,
                                arg_options.file_pattern,
                                action_ch='g',
                                recursively=True,
                                force=arg_options.force)
        # special case for the setup.py file
        shutil.copy2(os.path.join('config', 'setup.py'), '.')
