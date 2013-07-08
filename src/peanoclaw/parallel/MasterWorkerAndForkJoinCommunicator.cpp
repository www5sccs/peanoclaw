/*
 * MasterWorkerAndForkJoinCommunicator.cpp
 *
 *  Created on: Mar 19, 2013
 *      Author: kristof
 */
#include "MasterWorkerAndForkJoinCommunicator.h"

#include "peanoclaw/records/CellDescription.h"
#include "peanoclaw/records/Data.h"


tarch::logging::Log peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator::_log("peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator");

void peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator::sendCellDescription(int cellDescriptionIndex) {
  logTraceInWith1Argument("sendCellDescription", cellDescriptionIndex);
  _cellDescriptionHeap.sendData(cellDescriptionIndex, _remoteRank, _position, _level, _messageType);
  logTraceOut("sendCellDescription");
}

void peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator::sendDataArray(int index) {
  logTraceInWith3Arguments("sendDataArray", index, _position, _level);
  _dataHeap.sendData(index, _remoteRank, _position, _level, _messageType);
  logTraceOut("sendDataArray");
}

void peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator::deleteArraysFromPatch(int cellDescriptionIndex) {
  logTraceInWith1Argument("deleteArraysFromPatch", cellDescriptionIndex);
  if(cellDescriptionIndex != -1) {
    assertion2(_cellDescriptionHeap.isValidIndex(cellDescriptionIndex), _position, _level);
    CellDescription cellDescription = _cellDescriptionHeap.getData(cellDescriptionIndex).at(0);

    if(cellDescription.getUNewIndex() != -1) {
      _dataHeap.deleteData(cellDescription.getUNewIndex());
    }
    if(cellDescription.getUOldIndex() != -1) {
      _dataHeap.deleteData(cellDescription.getUOldIndex());
    }
    if(cellDescription.getAuxIndex() != -1) {
      _dataHeap.deleteData(cellDescription.getAuxIndex());
    }
  }
  logTraceOut("deleteArraysFromPatch");
}

int peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator::receiveDataArray() {
  logTraceIn("receiveDataArray");
  std::vector<Data> remoteArray = _dataHeap.receiveData(_remoteRank, _position, _level, _messageType);

  int localIndex = _dataHeap.createData();
  std::vector<Data>& localArray = _dataHeap.getData(localIndex);

  // Copy array
  std::vector<Data>::iterator it = remoteArray.begin();
  localArray.assign(it, remoteArray.end());
  assertionEquals2(remoteArray.size(), localArray.size(), _position, _level);

  logTraceOut("receiveDataArray");
  return localIndex;
}

peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator::MasterWorkerAndForkJoinCommunicator(
  int remoteRank,
  const tarch::la::Vector<DIMENSIONS,double> position,
  int level,
  bool forkOrJoin
) : _remoteRank(remoteRank),
    _position(position),
    _level(level),
    _cellDescriptionHeap(peano::heap::Heap<CellDescription>::getInstance()),
    _dataHeap(peano::heap::Heap<Data>::getInstance()),
    _messageType(forkOrJoin ? peano::heap::ForkOrJoinCommunication : peano::heap::MasterWorkerCommunication) {
  logTraceInWith3Arguments("MasterWorkerAndForkJoinCommunicator", remoteRank, position, level);

  logTraceOut("MasterWorkerAndForkJoinCommunicator");
}

void peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator::sendPatch(
  int cellDescriptionIndex
) {
  logTraceInWith3Arguments("sendPatch", cellDescriptionIndex, _position, _level);
  if(cellDescriptionIndex != -1) {
    sendCellDescription(cellDescriptionIndex);

    CellDescription cellDescription = _cellDescriptionHeap.getData(cellDescriptionIndex).at(0);

    if(cellDescription.getUNewIndex() != -1) {
      std::vector<Data>& localMatrix = peano::heap::Heap<peanoclaw::records::Data>::getInstance().getData(cellDescription.getUNewIndex());
//      std::cout << "sending new data with elements: " << localMatrix.size() << std::endl;
      sendDataArray(cellDescription.getUNewIndex());
    }

    if(cellDescription.getUOldIndex() != -1) {
      std::vector<Data>& localMatrix = peano::heap::Heap<peanoclaw::records::Data>::getInstance().getData(cellDescription.getUOldIndex());
//      std::cout << "sending new data with elements: " << localMatrix.size() << std::endl;
      sendDataArray(cellDescription.getUOldIndex());
    }

    if(cellDescription.getAuxIndex() != -1) {
      std::vector<Data>& localMatrix = peano::heap::Heap<peanoclaw::records::Data>::getInstance().getData(cellDescription.getAuxIndex());
//      std::cout << "sending old data with elements: " << localMatrix.size() << std::endl;
      sendDataArray(cellDescription.getAuxIndex());
    }
  }
  logTraceOut("sendPatch");
}

void peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator::receivePatch(int localCellDescriptionIndex) {
  logTraceInWith3Arguments("receivePatch", localCellDescriptionIndex, _position, _level);

  std::vector<CellDescription> remoteCellDescriptionVector = _cellDescriptionHeap.receiveData(_remoteRank, _position, _level, _messageType);
  assertionEquals2(remoteCellDescriptionVector.size(), 1, _position, _level);
  CellDescription remoteCellDescription = remoteCellDescriptionVector[0];

  assertion3(localCellDescriptionIndex >= 0, localCellDescriptionIndex, _position, _level);
  CellDescription localCellDescription = _cellDescriptionHeap.getData(localCellDescriptionIndex).at(0);
  #ifdef Asserts
  assertionNumericalEquals2(remoteCellDescription.getPosition(), localCellDescription.getPosition(), localCellDescription.toString(), remoteCellDescription.toString());
  assertionNumericalEquals2(remoteCellDescription.getSize(), localCellDescription.getSize(), localCellDescription.toString(), remoteCellDescription.toString());
  assertionEquals2(remoteCellDescription.getLevel(), localCellDescription.getLevel(), localCellDescription.toString(), remoteCellDescription.toString());
  #endif

  //TODO unterweg debug
  std::cout << "Received cell description: " << remoteCellDescription.toString() << std::endl;

  //Load arrays and stores according indices in cell description
  if(remoteCellDescription.getUNewIndex() != -1) {
    remoteCellDescription.setUNewIndex(receiveDataArray());
  }

  if(remoteCellDescription.getUOldIndex() != -1) {
    remoteCellDescription.setUOldIndex(receiveDataArray());
  }

  if(remoteCellDescription.getAuxIndex() != -1) {
    remoteCellDescription.setAuxIndex(receiveDataArray());
  }

  //Copy remote cell description to local cell description
  deleteArraysFromPatch(localCellDescriptionIndex);
  remoteCellDescription.setCellDescriptionIndex(localCellDescriptionIndex);
  _cellDescriptionHeap.getData(localCellDescriptionIndex).at(0) = remoteCellDescription;
  assertionEquals(_cellDescriptionHeap.getData(localCellDescriptionIndex).size(), 1);

  assertionEquals(_cellDescriptionHeap.getData(localCellDescriptionIndex).at(0).getCellDescriptionIndex(), localCellDescriptionIndex);
  logTraceOut("receivePatch");
}
