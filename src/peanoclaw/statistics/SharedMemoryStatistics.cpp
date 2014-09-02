/*
 * SharedMemoryStatistics.cpp
 *
 *  Created on: Sep 2, 2014
 *      Author: kristof
 */
#include "peanoclaw/statistics/SharedMemoryStatistics.h"

#include "peanoclaw/Patch.h"

#include <pthread.h>

tarch::logging::Log peanoclaw::statistics::SharedMemoryStatistics::_log("peanoclaw::statistics::SharedMemoryStatistics");

peanoclaw::statistics::SharedMemoryStatistics::SharedMemoryStatistics() {
}

peanoclaw::statistics::SharedMemoryStatistics::~SharedMemoryStatistics() {
  logStatistics();
}

void peanoclaw::statistics::SharedMemoryStatistics::addCellUpdatesForThread(
  const peanoclaw::Patch& subgrid
) {
  #ifdef SharedTBB
  unsigned long int threadID = pthread_self();
  std::map<unsigned long int, double>::iterator entry = _cellUpdatesPerThread.find(threadID);
  if(entry == _cellUpdatesPerThread.end()) {
    _cellUpdatesPerThread[threadID] = 0;
    entry = _cellUpdatesPerThread.find(threadID);
  }
  _cellUpdatesPerThread[threadID] += tarch::la::volume(subgrid.getSubdivisionFactor());
  #endif
}

void peanoclaw::statistics::SharedMemoryStatistics::logStatistics() const {
  for(std::map<unsigned long int, double>::const_iterator entry = _cellUpdatesPerThread.begin();
      entry != _cellUpdatesPerThread.end();
      entry++) {
    logInfo("logStatistics()", "CellUpdates on thread " << entry->first << ": " << entry->second);
  }
}


