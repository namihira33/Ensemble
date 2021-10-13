#!/bin/bash

for mn in 1e-7 2e-7 3e-7 4e-7 5e-7 6e-7 7e-7 8e-7 9e-7
do
    python ./src/ensemble.py cv=1 evaluate=0 lr=$mn
done