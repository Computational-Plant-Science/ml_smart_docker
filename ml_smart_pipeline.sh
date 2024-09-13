#!/bin/bash

python3 /opt/code/ml_smart.py -p $INPUT -o $OUTPUT 

# copy nested output files to working directory
find . -type f -name "*.xlsx" -exec cp {} $WORKDIR \;
