/*
 * NeighbourCommunicator.cpp
 *
 *  Created on: Mar 15, 2013
 *      Author: unterweg
 */
#include "NeighbourCommunicator.h"

#include "peanoclaw/records/CellDescription.h"
#include "peanoclaw/records/Data.h"

tarch::logging::Log peanoclaw::parallel::NeighbourCommunicator::_log("peanoclaw::parallel::NeighbourCommunicator");

void peanoclaw::parallel::NeighbourCommunicator::sendCellDescription(int cellDescriptionIndex) {
  logTraceInWith1Argument("sendCellDescription", cellDescriptionIndex);
  _cellDescriptionHeap.sendData(cellDescriptionIndex, _remoteRank, _position, _level, peano::heap::NeighbourCommunication);
  logTraceOut("sendCellDescription");
}

void peanoclaw::parallel::NeighbourCommunicator::sendPaddingCellDescription() {
  logTraceIn("sendPaddingCellDescription");
  int cellDescriptionIndex = _cellDescriptionHeap.createData();

  CellDescription paddingCellDescription;
  #ifdef Parallel
  paddingCellDescription.setIsPaddingSubgrid(true);
  #endif

  _cellDescriptionHeap.getData(cellDescriptionIndex).push_back(paddingCellDescription);

  sendCellDescription(cellDescriptionIndex);
  _cellDescriptionHeap.deleteData(cellDescriptionIndex);
  logTraceOut("sendPaddingCellDescription");
}

void peanoclaw::parallel::NeighbourCommunicator::sendDataArray(int index) {
  logTraceInWith3Arguments("sendDataArray", index, _position, _level);
  _dataHeap.sendData(index, _remoteRank, _position, _level, peano::heap::NeighbourCommunication);
  logTraceOut("sendDataArray");
}

void peanoclaw::parallel::NeighbourCommunicator::sendPaddingDataArray() {
  logTraceInWith2Arguments("sendPaddingDataArray", _position, _level);
  int index = _dataHeap.createData();
  sendDataArray(index);
  _dataHeap.deleteData(index);
  logTraceOut("sendPaddingDataArray");
}

void peanoclaw::parallel::NeighbourCommunicator::deleteArraysFromPatch(int cellDescriptionIndex) {
  logTraceInWith1Argument("deleteArraysFromPatch", cellDescriptionIndex);
  if(cellDescriptionIndex != -1) {
    assertion2(_cellDescriptionHeap.isValidIndex(cellDescriptionIndex), _position, _level);
    CellDescription cellDescription = _cellDescriptionHeap.getData(cellDescriptionIndex).at(0);

    if(cellDescription.getUNewIndex() != -1) {
      _dataHeap.deleteData(cellDescription.getUNewIndex());
    }
//    if(cellDescription.getUOldIndex() != -1) {
//      _dataHeap.deleteData(cellDescription.getUOldIndex());
//    }
//    if(cellDescription.getAuxIndex() != -1) {
//      _dataHeap.deleteData(cellDescription.getAuxIndex());
//    }
  }
  logTraceOut("deleteArraysFromPatch");
}

int peanoclaw::parallel::NeighbourCommunicator::receiveDataArray() {
  logTraceIn("receiveDataArray");
  std::vector<Data> remoteArray = _dataHeap.receiveData(_remoteRank, _position, _level, peano::heap::NeighbourCommunication);

  int localIndex = _dataHeap.createData();
  std::vector<Data>& localArray = _dataHeap.getData(localIndex);

  // Copy array
  std::vector<Data>::iterator it = remoteArray.begin();
  localArray.assign(it, remoteArray.end());

  logTraceOut("receiveDataArray");
  return localIndex;
}

void peanoclaw::parallel::NeighbourCommunicator::sendPatch(
  int cellDescriptionIndex
) {
  logTraceInWith3Arguments("sendPatch", cellDescriptionIndex, _position, _level);
  #ifdef Parallel

  //TODO unterweg debug
//  logInfo("", "Sending from " << tarch::parallel::Node::getInstance().getRank()
//      << " to " << _remoteRank
//      << ", position:" << _position);

  bool sendActualPatch = true;
  if(cellDescriptionIndex != -1) {
    CellDescription cellDescription = _cellDescriptionHeap.getData(cellDescriptionIndex).at(0);
    sendActualPatch = !cellDescription.getIsRemote();
//    sendActualPatch &= !cellDescription.getCurrentStateWasSend();
  } else {
    sendActualPatch = false;
  }

  if(sendActualPatch) {
    _statistics.sentNeighborData();
    sendCellDescription(cellDescriptionIndex);

    CellDescription cellDescription = _cellDescriptionHeap.getData(cellDescriptionIndex).at(0);

    //TODO unterweg debug
//    logInfo("", "Sending actual patch from " << tarch::parallel::Node::getInstance().getRank()
//      << " to " << _remoteRank
//      << ", position:" << _position
//      << " : " << cellDescription.toString()
//    );

    if(cellDescription.getUNewIndex() != -1) {
      sendDataArray(cellDescription.getUNewIndex());
    } else {
      sendPaddingDataArray();
    }

//    if(cellDescription.getUOldIndex() != -1) {
//      sendDataArray(cellDescription.getUOldIndex());
//    } else {
//      sendPaddingDataArray();
//    }
//
//    if(cellDescription.getAuxIndex() != -1) {
//      sendDataArray(cellDescription.getAuxIndex());
//    } else {
//      sendPaddingDataArray();
//    }
  } else {

    //TODO unterweg debug
//    logInfo("", "Sending padding patch from " << tarch::parallel::Node::getInstance().getRank()
//      << " to " << _remoteRank
//      << ", position:" << _position
//    );

    sendPaddingPatch();
  }
  #endif
  logTraceOut("sendPatch");
}

void peanoclaw::parallel::NeighbourCommunicator::sendPaddingPatch() {
  logTraceInWith2Arguments("sendPaddingPatch", _position, _level);
  _statistics.sentPaddingNeighborData();
  sendPaddingCellDescription();
  sendPaddingDataArray(); //UNew
//  sendPaddingDataArray(); //UOld
//  sendPaddingDataArray(); //Aux
  logTraceOut("sendPaddingPatch");
}

void peanoclaw::parallel::NeighbourCommunicator::receivePatch(int localCellDescriptionIndex) {
  #ifdef Parallel
  logTraceInWith3Arguments("receivePatch", localCellDescriptionIndex, _position, _level);

  assertion(localCellDescriptionIndex > -1);
  CellDescription localCellDescription = _cellDescriptionHeap.getData(localCellDescriptionIndex).at(0);

  std::vector<CellDescription> remoteCellDescriptionVector = _cellDescriptionHeap.receiveData(_remoteRank, _position, _level, peano::heap::NeighbourCommunication);
  assertionEquals2(remoteCellDescriptionVector.size(), 1, _position, _level);
  CellDescription remoteCellDescription = remoteCellDescriptionVector[0];

  assertionEquals2(localCellDescription.getLevel(), _level, localCellDescription.toString(), tarch::parallel::Node::getInstance().getRank());
  assertionNumericalEquals3(localCellDescription.getPosition(), remoteCellDescription.getPosition(),
      localCellDescription.toString(), remoteCellDescription.toString(), tarch::parallel::Node::getInstance().getRank());
  assertionNumericalEquals3(localCellDescription.getSize(), remoteCellDescription.getSize(),
      localCellDescription.toString(), remoteCellDescription.toString(), tarch::parallel::Node::getInstance().getRank());
  assertionEquals3(localCellDescription.getLevel(), remoteCellDescription.getLevel(),
      localCellDescription.toString(), remoteCellDescription.toString(), tarch::parallel::Node::getInstance().getRank());

  if(!remoteCellDescription.getIsPaddingSubgrid()) {
    _statistics.receivedNeighborData();
    //Load arrays and stores according indices in cell description
//    if(remoteCellDescription.getAuxIndex() != -1) {
//      remoteCellDescription.setAuxIndex(receiveDataArray());
//    } else {
//      receiveDataArray();
//    }
//    if(remoteCellDescription.getUOldIndex() != -1) {
//      remoteCellDescription.setUOldIndex(receiveDataArray());
//    } else {
//      receiveDataArray();
//    }
    if(remoteCellDescription.getUNewIndex() != -1) {
      remoteCellDescription.setUNewIndex(receiveDataArray());
    } else {
      receiveDataArray();
    }

    //Copy remote cell description to local cell description
    assertion2(localCellDescriptionIndex >= 0, _position, _level);
    deleteArraysFromPatch(localCellDescriptionIndex);
    remoteCellDescription.setCellDescriptionIndex(localCellDescriptionIndex);
    remoteCellDescription.setIsRemote(true); //TODO unterweg: Remote patches are currently never destroyed.
    _cellDescriptionHeap.getData(localCellDescriptionIndex).at(0) = remoteCellDescription;
    assertionEquals(_cellDescriptionHeap.getData(localCellDescriptionIndex).size(), 1);
  } else {
    _statistics.receivedPaddingNeighborData();
//    receiveDataArray(); //Aux
//    receiveDataArray(); //UOld
    receiveDataArray(); //UNew
  }

  assertionEquals(_cellDescriptionHeap.getData(localCellDescriptionIndex).at(0).getCellDescriptionIndex(), localCellDescriptionIndex);
  logTraceOut("receivePatch");
  #endif
}

void peanoclaw::parallel::NeighbourCommunicator::receivePaddingPatch() {
  logTraceInWith2Arguments("receivePaddingPatch", _position, _level);
  _statistics.receivedPaddingNeighborData();

  //Receive padding patch
  _cellDescriptionHeap.receiveData(_remoteRank, _position, _level, peano::heap::NeighbourCommunication);

  //Aux
//  _dataHeap.receiveData(_remoteRank, _position, _level, peano::heap::NeighbourCommunication);
//
//  //UOld
//  _dataHeap.receiveData(_remoteRank, _position, _level, peano::heap::NeighbourCommunication);

  //UNew
  _dataHeap.receiveData(_remoteRank, _position, _level, peano::heap::NeighbourCommunication);
  logTraceOut("receivePaddingPatch");
}

peanoclaw::parallel::NeighbourCommunicator::NeighbourCommunicator(
  int remoteRank,
  const tarch::la::Vector<DIMENSIONS,double> position,
  int level,
  peanoclaw::statistics::ParallelStatistics& statistics
) : _remoteRank(remoteRank),
    _position(position),
    _level(level),
    _cellDescriptionHeap(peano::heap::Heap<CellDescription>::getInstance()),
    _dataHeap(peano::heap::Heap<Data>::getInstance()),
    _statistics(statistics) {
  logTraceInWith3Arguments("NeighbourCommunicator", remoteRank, position, level);

  logTraceOut("NeighbourCommunicator");
}

void peanoclaw::parallel::NeighbourCommunicator::sendSubgridsForVertex(
  peanoclaw::Vertex&                           vertex,
  const tarch::la::Vector<DIMENSIONS, double>& vertexPosition,
  const tarch::la::Vector<DIMENSIONS, double>& adjacentSubgridSize,
  int                                          level
) {
  #ifdef Parallel
  if(!tarch::parallel::Node::getInstance().isGlobalMaster() && _remoteRank != 0) {
    //TODO unterweg debug
//    logInfo("", "Sending to neighbor " << tarch::parallel::Node::getInstance().getRank()
//      << " to " << toRank
//      << ", position:" << x
//      << ", level:" << level);

    int localRank = tarch::parallel::Node::getInstance().getRank();
    int neighborPatches = 0;
    int localPatches = 0;

    for(int i = 0; i < TWO_POWER_D; i++) {
      if(vertex.getAdjacentRanks()(i) == localRank) {
        int adjacentCellDescriptionIndex = vertex.getAdjacentCellDescriptionIndexInPeanoOrder(i);
        sendPatch(adjacentCellDescriptionIndex);
        localPatches++;
      } else if (vertex.getAdjacentRanks()(i) == _remoteRank) {
        neighborPatches++;
      }
    }

    //Send padding patches if remote rank has more patches than local rank
    for( ; localPatches < neighborPatches; localPatches++) {
      sendPaddingPatch();
    }
  }
  #endif
}

void peanoclaw::parallel::NeighbourCommunicator::receiveSubgridsForVertex(
  peanoclaw::Vertex&                           localVertex,
  const peanoclaw::Vertex&                     remoteVertex,
  const tarch::la::Vector<DIMENSIONS, double>& vertexPosition,
  const tarch::la::Vector<DIMENSIONS, double>& adjacentSubgridSize,
  int                                          level
) {
  #ifdef Parallel
  if(!tarch::parallel::Node::getInstance().isGlobalMaster() && _remoteRank != 0) {
    assertionEquals(localVertex.isInside(), remoteVertex.isInside());
    assertionEquals(localVertex.isBoundary(), remoteVertex.isBoundary());

    tarch::la::Vector<TWO_POWER_D, int> neighbourVertexRanks = localVertex.getAdjacentRanksDuringLastIteration();

    //Count local and neighbor patches
    int localRank = tarch::parallel::Node::getInstance().getRank();
    int localPatches = 0;
    int neighborPatches = 0;
    for(int i = TWO_POWER_D-1; i >= 0; i--) {
      if(neighbourVertexRanks(i) == _remoteRank) {
        neighborPatches++;
      } else if(neighbourVertexRanks(i) == localRank) {
        localPatches++;
      }
    }

    //Receive padding patches
    for( ; neighborPatches < localPatches; neighborPatches++) {
      receivePaddingPatch();
    }

    //Receive actual patches
    for(int i = TWO_POWER_D-1; i >= 0; i--) {
      tarch::la::Vector<DIMENSIONS,double> patchPosition = vertexPosition + tarch::la::multiplyComponents(adjacentSubgridSize, peano::utils::dDelinearised(i, 2).convertScalar<double>() - 1.0);
      int localAdjacentCellDescriptionIndex = localVertex.getAdjacentCellDescriptionIndexInPeanoOrder(i);
      int remoteAdjacentCellDescriptionIndex = remoteVertex.getAdjacentCellDescriptionIndexInPeanoOrder(i);

      assertion4(
        localAdjacentCellDescriptionIndex != -1
        || localVertex.isAdjacentToRemoteRank(),
        localAdjacentCellDescriptionIndex,
        remoteAdjacentCellDescriptionIndex,
        patchPosition,
        level
      );

      if(neighbourVertexRanks(i) == _remoteRank) {
        if(remoteAdjacentCellDescriptionIndex != -1) {
          assertion(localAdjacentCellDescriptionIndex != -1);
          receivePatch(localAdjacentCellDescriptionIndex);
        } else {
          //Receive dummy message
          receivePaddingPatch();
        }
      }
    }
  }
  #endif
}
