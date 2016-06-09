# MUMPS.py
Python inferface to MUMPS ([MUltifrontal Massively Parallel sparse direct Solver](http://mumps.enseeiht.fr/)).

[![Build Status](https://travis-ci.com/PythonOptimizers/MUMPS.py.svg?token=33z5zptBt5SzXC4ZvLpF&branch=master)](https://travis-ci.com/PythonOptimizers/MUMPS.py)

It supports all four types (single real, double real, single complex and double complex).


## Installation

### branch `master`

1. You need to install MUMPS. Follow instructions on [their website](http://mumps.enseeiht.fr/).
       If you are under OS X, a [Homebrew](http://brew.sh) formula is available. Follow the instructions to install Homebrew.
       Then, MUMPS and its dependencies can be installed automatically in `/usr/local` by typing

    	brew install gcc  # contains gfortran

    	brew tap homebrew/science

    	brew install mumps

2. Clone repository

        git clone git@github.com:optimizers/MUMPS.py.git

3. Install Python dependencies

        pip install numpy

4. Copy `site.template.cfg` to `site.cfg` and adjust it to your needs
    
5. Install `MUMPS.py`

    	python setup.py build
    	python setup.py install


### branch `develop`

1. You need to install MUMPS. Follow instructions on [their website](http://mumps.enseeiht.fr/).
       If you are under OS X, a [Homebrew](http://brew.sh) formula is available. Follow the instructions to install Homebrew.
       Then, MUMPS and its dependencies can be installed automatically in `/usr/local` by typing

    	brew install gcc  # contains gfortran

    	brew tap homebrew/science

    	brew install mumps

2. Clone repository

        git clone git@github.com:optimizers/MUMPS.py.git

3. Install Python dependencies

        pip install numpy
        pip install cygenja

4. Copy `site.template.cfg` to `site.cfg` and adjust it to your needs

5. Generate Cython files

        python generate_code.py

6. Install `MUMPS.py`

    	python setup.py build
    	python setup.py install



## Running tests

    pip install pytest
    py.test tests

## TODO:

  - [ ] Add a `refine` method 
  - [ ] Add tests for everything
  - [ ] Make statistics work
  - [ ] ensure all code is PEP8 and PEP257 compliant
