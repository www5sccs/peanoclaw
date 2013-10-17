#!/bin/bash

mkdir -p vtkOutput
rm vtkOutput/*

mpirun -np 10 python shallow2DParallel.py amr_type=peano

mpirun -np 1 python shallow2DParallel.py amr_type=peano

python ../tools/compareResult.py vtkOutput/adaptive-rank-__RANK__-__ITERATION__.vtk SameAsPath1 1 0 1 1:10 1e-8
