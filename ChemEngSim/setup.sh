#!/bin/bash

START=$PWD
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $DIR/Fortran-Simson
make clean
make mpi=yes
cp $DIR/Fortran-Simson/bla $DIR/init_sim/
make clean
cd $START
