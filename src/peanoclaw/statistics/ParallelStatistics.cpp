/*
 * ParallelStatistics.cpp
 *
 *  Created on: Aug 5, 2013
 *      Author: kristof
 */
#include "peanoclaw/statistics/ParallelStatistics.h"

tarch::logging::Log peanoclaw::statistics::ParallelStatistics::_log("peanoclaw::statistics::ParallelStatistics");

peanoclaw::statistics::ParallelStatistics::ParallelStatistics(std::string name)
 : _name(name),
   _sentNeighborData(0),
   _sentPaddingNeighborData(0),
   _receivedNeighborData(0),
   _receivedPaddingNeighborData(0),
   _waitingTimeMasterWorkerSpacetreeCommunication(0.0),
   _samplesMasterWorkerSpacetreeCommunication(0),
   _waitingTimeMasterWorkerSubgridCommunication(0.0),
   _samplesMasterWorkerSubgridCommunication(0),
   _waitingTimeNeighborSubgridCommunication(0.0),
   _samplesNeighborSubgridCommunication(0) {
}

void peanoclaw::statistics::ParallelStatistics::sentNeighborData( int numberOfSentSubgrids ) {
  _sentNeighborData += 1; //numberOfSentSubgrids;
}

void peanoclaw::statistics::ParallelStatistics::sentPaddingNeighborData( int numberOfSentSubgrids ) {
  _sentPaddingNeighborData += 1; //numberOfSentSubgrids;
}

void peanoclaw::statistics::ParallelStatistics::receivedNeighborData( int numberOfReceivedSubgrids ) {
  _receivedNeighborData += 1; //numberOfReceivedSubgrids;
}

void peanoclaw::statistics::ParallelStatistics::receivedPaddingNeighborData( int numberOfReceivedSubgrids ) {
  _receivedPaddingNeighborData += 1; //numberOfReceivedSubgrids;
}

void peanoclaw::statistics::ParallelStatistics::addWaitingTimeForMasterWorkerSpacetreeCommunication(
  double time
) {
  _waitingTimeMasterWorkerSpacetreeCommunication += time;
  _samplesMasterWorkerSpacetreeCommunication++;
}

void peanoclaw::statistics::ParallelStatistics::addWaitingTimeForMasterWorkerSubgridCommunication(
  double time
) {
  _waitingTimeMasterWorkerSubgridCommunication += time;
  _samplesMasterWorkerSubgridCommunication++;
}

void peanoclaw::statistics::ParallelStatistics::addWaitingTimeForNeighborSubgridCommunication(
  double time
) {
  _waitingTimeNeighborSubgridCommunication += time;
  _samplesNeighborSubgridCommunication++;
}

void peanoclaw::statistics::ParallelStatistics::logIterationStatistics() const {
  logInfo("logStatistics()", "Parallel statistics for " << _name << ":");
  logInfo("logStatistics()", "Subgrids sent to neighbors: " << _sentNeighborData);
  logInfo("logStatistics()", "Padding subgrids sent to neighbors: " << _sentPaddingNeighborData);
  logInfo("logStatistics()", "Subgrids received from neighbors: " << _receivedNeighborData);
  logInfo("logStatistics()", "Padding subgrids received from neighbors: " << _receivedPaddingNeighborData);
}

void peanoclaw::statistics::ParallelStatistics::logTotalStatistics() const {
//  logInfo("logStatistics()", "Waiting time for master-worker spacetree communication: "
//      << _waitingTimeMasterWorkerSpacetreeCommunication << " (total), "
//      << (_waitingTimeMasterWorkerSpacetreeCommunication / _samplesMasterWorkerSpacetreeCommunication) << " (average) "
//      << _samplesMasterWorkerSpacetreeCommunication << " samples");
//  logInfo("logStatistics()", "Waiting time for master-worker subgrid communication: "
//      << _waitingTimeMasterWorkerSubgridCommunication << " (total), "
//      << (_waitingTimeMasterWorkerSubgridCommunication / _samplesMasterWorkerSubgridCommunication) << " (average) "
//      << _samplesMasterWorkerSubgridCommunication << " samples");
//  logInfo("logStatistics()", "Waiting time for neighbor subgrid communication: "
//      << _waitingTimeNeighborSubgridCommunication << " (total), "
//      << (_waitingTimeNeighborSubgridCommunication / _samplesNeighborSubgridCommunication) << " (average) "
//      << _samplesNeighborSubgridCommunication << " samples");
}

void peanoclaw::statistics::ParallelStatistics::merge(const ParallelStatistics& otherStatistics) {
  _sentNeighborData            += otherStatistics._sentNeighborData;
  _sentPaddingNeighborData     += otherStatistics._sentPaddingNeighborData;
  _receivedNeighborData        += otherStatistics._receivedNeighborData;
  _receivedPaddingNeighborData += otherStatistics._receivedPaddingNeighborData;
  _waitingTimeMasterWorkerSpacetreeCommunication += otherStatistics._waitingTimeMasterWorkerSpacetreeCommunication;
  _samplesMasterWorkerSpacetreeCommunication     += otherStatistics._samplesMasterWorkerSpacetreeCommunication;
  _waitingTimeMasterWorkerSubgridCommunication   += otherStatistics._waitingTimeMasterWorkerSubgridCommunication;
  _samplesMasterWorkerSubgridCommunication       += otherStatistics._samplesMasterWorkerSubgridCommunication;
  _waitingTimeNeighborSubgridCommunication       += otherStatistics._waitingTimeNeighborSubgridCommunication;
  _samplesNeighborSubgridCommunication           += otherStatistics._samplesNeighborSubgridCommunication;
}
