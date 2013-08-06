/*
 * ParallelStatistics.cpp
 *
 *  Created on: Aug 5, 2013
 *      Author: kristof
 */
#include "peanoclaw/statistics/ParallelStatistics.h"

tarch::logging::Log peanoclaw::statistics::ParallelStatistics::_log("peanoclaw::statistics::ParallelStatistics");

peanoclaw::statistics::ParallelStatistics::ParallelStatistics(std::string name)
 : _name(name), _sentNeighborData(0), _sentPaddingNeighborData(0), _receivedNeighborData(0), _receivedPaddingNeighborData(0) {
}

void peanoclaw::statistics::ParallelStatistics::sentNeighborData( int numberOfSentSubgrids ) {
  _sentNeighborData += numberOfSentSubgrids;
}

void peanoclaw::statistics::ParallelStatistics::sentPaddingNeighborData( int numberOfSentSubgrids ) {
  _sentPaddingNeighborData += numberOfSentSubgrids;
}

void peanoclaw::statistics::ParallelStatistics::receivedNeighborData( int numberOfReceivedSubgrids ) {
  _receivedNeighborData += numberOfReceivedSubgrids;
}

void peanoclaw::statistics::ParallelStatistics::receivedPaddingNeighborData( int numberOfReceivedSubgrids ) {
  _receivedPaddingNeighborData += numberOfReceivedSubgrids;
}

void peanoclaw::statistics::ParallelStatistics::logStatistics() const {
  logInfo("logStatistics()", "Parallel statistics for " << _name << ":");
  logInfo("logStatistics()", "Subgrids sent to neighbors: " << _sentNeighborData);
  logInfo("logStatistics()", "Padding subgrids sent to neighbors: " << _sentPaddingNeighborData);
  logInfo("logStatistics()", "Subgrids received from neighbors: " << _receivedNeighborData);
  logInfo("logStatistics()", "Padding subgrids received from neighbors: " << _receivedPaddingNeighborData);
}

void peanoclaw::statistics::ParallelStatistics::merge(const ParallelStatistics& otherStatistics) {
  _sentNeighborData            += otherStatistics._sentNeighborData;
  _sentPaddingNeighborData     += otherStatistics._sentPaddingNeighborData;
  _receivedNeighborData        += otherStatistics._receivedNeighborData;
  _receivedPaddingNeighborData += otherStatistics._receivedPaddingNeighborData;
}
