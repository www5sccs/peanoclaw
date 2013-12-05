/*
 * SubgridCommunicator.cpp
 *
 *  Created on: Dec 4, 2013
 *      Author: kristof
 */
#include "peanoclaw/parallel/SubgridCommunicator.h"

#include "tarch/Assertions.h"

tarch::logging::Log peanoclaw::parallel::SubgridCommunicator::_log("peanoclaw::parallel::SubgridCommunicator");

peanoclaw::parallel::SubgridCommunicator::SubgridCommunicator(
  int                                         remoteRank,
  const tarch::la::Vector<DIMENSIONS,double>& position,
  int                                         level,
  peano::heap::MessageType                    messageType
) : _remoteRank(remoteRank), _position(position), _level(level), _messageType(messageType) {

}

void peanoclaw::parallel::SubgridCommunicator::sendCellDescription(int cellDescriptionIndex)
{
  logTraceInWith1Argument("sendCellDescription", cellDescriptionIndex);
  #if defined(Asserts) && defined(Parallel)
  CellDescription& cellDescription = CellDescriptionHeap::getInstance().getData(cellDescriptionIndex).at(0);
  assertion1(!cellDescription.getIsPaddingSubgrid(), cellDescription.toString());
  assertion1(!cellDescription.getIsRemote(), cellDescription.toString());
  #endif

  CellDescriptionHeap::getInstance().sendData(cellDescriptionIndex, _remoteRank, _position, _level, _messageType);
  logTraceOut("sendCellDescription");
}

void peanoclaw::parallel::SubgridCommunicator::sendPaddingCellDescription(
  const tarch::la::Vector<DIMENSIONS, double>& position,
  int                                          level,
  const tarch::la::Vector<DIMENSIONS, double>& subgridSize
) {
  logTraceIn("sendPaddingCellDescription");
  int cellDescriptionIndex = CellDescriptionHeap::getInstance().createData();

//  CellDescription paddingCellDescription;
//  #ifdef Parallel
//  paddingCellDescription.setIsPaddingSubgrid(true);
//  paddingCellDescription.setIsRemote(true);
//  #endif
//  paddingCellDescription.setPosition(position);
//  paddingCellDescription.setLevel(level);
//  paddingCellDescription.setSize(subgridSize);
//
//  CellDescriptionHeap::getInstance().getData(cellDescriptionIndex).push_back(paddingCellDescription);

  CellDescriptionHeap::getInstance().sendData(cellDescriptionIndex, _remoteRank, _position, _level, _messageType);

  CellDescriptionHeap::getInstance().deleteData(cellDescriptionIndex);
  logTraceOut("sendPaddingCellDescription");
}

void peanoclaw::parallel::SubgridCommunicator::sendDataArray(int index) {
  logTraceInWith3Arguments("sendDataArray", index, _position, _level);
  DataHeap::getInstance().sendData(index, _remoteRank, _position, _level, _messageType);
  logTraceOut("sendDataArray");
}

void peanoclaw::parallel::SubgridCommunicator::sendPaddingDataArray() {
  logTraceInWith2Arguments("sendPaddingDataArray", _position, _level);
  int index = DataHeap::getInstance().createData();
  sendDataArray(index);
  DataHeap::getInstance().deleteData(index);
  logTraceOut("sendPaddingDataArray");
}

int peanoclaw::parallel::SubgridCommunicator::receiveDataArray() {
  logTraceIn("receiveDataArray");

  std::vector<Data> remoteArray = DataHeap::getInstance().receiveData(_remoteRank, _position, _level, _messageType);

  int localIndex = DataHeap::getInstance().createData();
  std::vector<Data>& localArray = DataHeap::getInstance().getData(localIndex);

  // Copy array
  std::vector<Data>::iterator it = remoteArray.begin();
  localArray.assign(it, remoteArray.end());

  logTraceOut("receiveDataArray");
  return localIndex;
}

void peanoclaw::parallel::SubgridCommunicator::deleteArraysFromSubgrid(Patch& subgrid) {
  logTraceInWith1Argument("deleteArraysFromSubgrid", cellDescriptionIndex);
  if(subgrid.getUNewIndex() != -1) {
    DataHeap::getInstance().deleteData(subgrid.getUNewIndex());
  }
  logTraceOut("deleteArraysFromSubgrid");
}
