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

void peanoclaw::statistics::SharedMemoryStatistics::addCellUpdatesForThread(
  double cellUpdates,
  unsigned long int threadID
) {
  std::map<unsigned long int, double>::iterator entry = _cellUpdatesPerThread.find(threadID);
  if(entry == _cellUpdatesPerThread.end()) {
    _cellUpdatesPerThread[threadID] = 0;
    entry = _cellUpdatesPerThread.find(threadID);
  }
  entry->second += cellUpdates;
}

peanoclaw::statistics::SharedMemoryStatistics::SharedMemoryStatistics() {
}

peanoclaw::statistics::SharedMemoryStatistics::~SharedMemoryStatistics() {
}

void peanoclaw::statistics::SharedMemoryStatistics::addCellUpdatesForThread(
  const peanoclaw::Patch& subgrid
) {
  #ifdef SharedTBB
  unsigned long int threadID = pthread_self();
  addCellUpdatesForThread(
    tarch::la::volume(subgrid.getSubdivisionFactor()),
    threadID
  );
  #endif
}

void peanoclaw::statistics::SharedMemoryStatistics::logStatistics() const {
  logInfo("logStatistics()", "#Threads=" << _cellUpdatesPerThread.size());
  int threadIndex = 0;
  for(std::map<unsigned long int, double>::const_iterator entry = _cellUpdatesPerThread.begin();
      entry != _cellUpdatesPerThread.end();
      entry++) {
    logInfo("logStatistics()", "CellUpdates on thread " << threadIndex << ": " << entry->second);
    threadIndex++;
  }
}

void peanoclaw::statistics::SharedMemoryStatistics::merge(const SharedMemoryStatistics& statistics) {
  for(std::map<unsigned long int, double>::const_iterator entry = statistics._cellUpdatesPerThread.begin();
        entry != statistics._cellUpdatesPerThread.end();
        entry++) {
    addCellUpdatesForThread(entry->second, entry->first);
  }
}
