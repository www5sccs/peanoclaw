#!/bin/bash
./generateHeapData.sh

java -jar ../p3/pdt/pdt.jar peanoclaw.peano-specification src/peanoclaw src/peanoclaw
