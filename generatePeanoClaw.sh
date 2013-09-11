#!/bin/bash
./generateHeapData.sh

java -jar /home/hpc/pr63so/lu26hij3/workspace/peano/pdt/pdt.jar peanoclaw.peano-specification src/peanoclaw src/peanoclaw
