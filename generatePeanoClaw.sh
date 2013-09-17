#!/bin/bash
./generateHeapData.sh
PDT_PATH=../pdt

java -jar $PDT_PATH/pdt.jar peanoclaw.peano-specification src/peanoclaw src/peanoclaw
