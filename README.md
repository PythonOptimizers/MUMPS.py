# MUMPS.py
Python inferface to MUMPS ([MUltifrontal Massively Parallel sparse direct Solver](http://mumps.enseeiht.fr/)).

It supports all four types (single real, double real, single complex and double complex).


## Installation
    
1. You need to install MUMPS. Follow instructions on [their website](http://mumps.enseeiht.fr/).
       If you are under OS X, a [Homebrew](http://brew.sh) formula is available. Follow the instructions to install Homebrew.
       Then, MUMPS and its dependencies can be installed automatically in `/usr/local` by typing

    	brew install gcc  # contains gfortran

    	brew tap homebrew/science

    	brew install mumps

2. Clone this repo and modify the `site.cfg` to match your configuration
    
3. Install `MUMPS.py`

    	python generate_code.py -a
    	python setup.py build
    	python setup.py install

## Running examples

## TODO:

  - [ ] Add a `refine` method 
  - [ ] Add tests for everything
  - [ ] Make statistics work
  - [ ] ensure all code is PEP8 and PEP257 compliant
