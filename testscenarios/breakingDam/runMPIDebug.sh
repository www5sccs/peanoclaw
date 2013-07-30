#!/bin/bash

if [ -z $1 ]
then
  echo "Usage: $0 <numberOfMPIProcesses>"
fi

mpirun -np $1 xterm -e gdb -ex run -args python shallow2D.py amr_type=peano
