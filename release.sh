#!/bin/sh
#
# Runs before git flow release finish

git branch -D without-cython
git push origin --delete without-cython
git checkout -b without-cython

git rm --cached \*.cpy
git rm --cached \*.cpx
git rm --cached \*.cpd
git rm --cached generate_code.py
git rm --cached release.sh

mv .gitignore /tmp/.gitignore-qr_mumps

cp config/site.template.cython.cfg site.cfg

python generate_code.py -c
python generate_code.py
python setup.py install

git add \*.c
git rm --cached -r build
git rm --cached -r config

cp config/site.template.cfg .
cp config/.gitignore .
cp config/.travis.yml

git add tests/\*.py
git add setup.py
git add --all

git commit -m "c files from last commit in develop"
git push --set-upstream origin without-cython
git checkout develop

