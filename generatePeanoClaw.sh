#!/bin/bash
./generateHeapData.sh
PDT_PATH=../peano3/pdt

java -jar $PDT_PATH/pdt.jar peanoclaw.peano-specification src/peanoclaw src/peanoclaw
