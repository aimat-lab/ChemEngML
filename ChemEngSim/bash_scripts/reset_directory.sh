#!/bin/bash

# BE CAREFUL!
# This script will remove all SIMSON files except the ibm.bin from the subdirectories.
# It can be used when calculations failed and you want a clean start.
# Syntax: ./ChemEngSim/bash_scripts/reset_directory.sh /absolute/path/to/structure/folder

echo "Cleaning "$1

find $1 -name \*.u -type f -delete
find $1 -name \*.out -type f -delete
find $1 -name \stop.now -type f -delete
find $1 -name \xy.stat -type f -delete
find $1 -name \bla* -type f -delete
