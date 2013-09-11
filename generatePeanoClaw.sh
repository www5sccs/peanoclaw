#!/bin/bash
./generateHeapData.sh

java -jar /home/atanasoa/workspace/peano/pdt/pdt.jar peanoclaw.peano-specification src/peanoclaw src/peanoclaw
