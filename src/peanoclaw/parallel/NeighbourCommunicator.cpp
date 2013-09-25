/*
 * NeighbourCommunicator.cpp
 *
 *  Created on: Mar 15, 2013
 *      Author: unterweg
 */
#include "NeighbourCommunicator.h"

#include "peanoclaw/Heap.h"
#include "peanoclaw/records/CellDescription.h"
#include "peanoclaw/records/Data.h"

tarch::logging::Log peanoclaw::parallel::NeighbourCommunicator::_log("peanoclaw::parallel::NeighbourCommunicator");

void peanoclaw::parallel::NeighbourCommunicator::sendCellDescription(int cellDescriptionIndex)
{
  logTraceInWith1Argument("sendCellDescription", cellDescriptionIndex);
  #if defined(Asserts) && defined(Parallel)
  CellDescription& cellDescription = CellDescriptionHeap::getInstance().getData(cellDescriptionIndex).at(0);
  assertion1(!cellDescription.getIsPaddingSubgrid(), cellDescription.toString());
  assertion1(!cellDescription.getIsRemote(), cellDescription.toString());
  #endif

  CellDescriptionHeap::getInstance().sendData(cellDescriptionIndex, _remoteRank, _position, _level, peano::heap::NeighbourCommunication);
  logTraceOut("sendCellDescription");
}

void peanoclaw::parallel::NeighbourCommunicator::sendPaddingCellDescription(
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

  CellDescriptionHeap::getInstance().sendData(cellDescriptionIndex, _remoteRank, _position, _level, peano::heap::NeighbourCommunication);

  CellDescriptionHeap::getInstance().deleteData(cellDescriptionIndex);
  logTraceOut("sendPaddingCellDescription");
}

void peanoclaw::parallel::NeighbourCommunicator::sendDataArray(int index) {
  logTraceInWith3Arguments("sendDataArray", index, _position, _level);
  DataHeap::getInstance().sendData(index, _remoteRank, _position, _level, peano::heap::NeighbourCommunication);
  logTraceOut("sendDataArray");
}

void peanoclaw::parallel::NeighbourCommunicator::sendPaddingDataArray() {
  logTraceInWith2Arguments("sendPaddingDataArray", _position, _level);
  int index = DataHeap::getInstance().createData();
  sendDataArray(index);
  DataHeap::getInstance().deleteData(index);
  logTraceOut("sendPaddingDataArray");
}

void peanoclaw::parallel::NeighbourCommunicator::deleteArraysFromPatch(int cellDescriptionIndex) {
  logTraceInWith1Argument("deleteArraysFromPatch", cellDescriptionIndex);
  if(cellDescriptionIndex != -1) {
    assertion2(CellDescriptionHeap::getInstance().isValidIndex(cellDescriptionIndex), _position, _level);
    CellDescription cellDescription = CellDescriptionHeap::getInstance().getData(cellDescriptionIndex).at(0);

    if(cellDescription.getUNewIndex() != -1) {
      DataHeap::getInstance().deleteData(cellDescription.getUNewIndex());
    }
//    if(cellDescription.getUOldIndex() != -1) {
//      DataHeap::getInstance().deleteData(cellDescription.getUOldIndex());
//    }
//    if(cellDescription.getAuxIndex() != -1) {
//      DataHeap::getInstance().deleteData(cellDescription.getAuxIndex());
//    }
  }
  logTraceOut("deleteArraysFromPatch");
}

int peanoclaw::parallel::NeighbourCommunicator::receiveDataArray() {
  logTraceIn("receiveDataArray");

  std::vector<Data> remoteArray = DataHeap::getInstance().receiveData(_remoteRank, _position, _level, peano::heap::NeighbourCommunication);

  int localIndex = DataHeap::getInstance().createData();
  std::vector<Data>& localArray = DataHeap::getInstance().getData(localIndex);

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

  #ifdef Asserts
  if(cellDescriptionIndex != -1) {
    CellDescription cellDescription = CellDescriptionHeap::getInstance().getData(cellDescriptionIndex).at(0);
    assertion3(tarch::la::allGreater(cellDescription.getSubdivisionFactor(), 0), cellDescription.toString(), _position, _level);
  }
  #endif

  tarch::la::Vector<DIMENSIONS, double> position;
  int                                   level;
  tarch::la::Vector<DIMENSIONS, double> subgridSize;
  bool sendActualPatch = true;
  if(cellDescriptionIndex != -1) {
    CellDescription cellDescription = CellDescriptionHeap::getInstance().getData(cellDescriptionIndex).at(0);
    position    = cellDescription.getPosition();
    level       = cellDescription.getLevel();
    subgridSize = cellDescription.getSize();

    sendActualPatch = !cellDescription.getIsRemote();
    if(_avoidSendingSubgridsThatAlreadyHaveBeenSent) {
      sendActualPatch &= !cellDescription.getCurrentStateWasSend();
    }
    if(_reduceMultipleSends) {
      if(cellDescription.getAdjacentRank() != -1) {
        if(cellDescription.getNumberOfSharedAdjacentVertices() > 1) {
          assertion1(cellDescription.getNumberOfSharedAdjacentVertices() <= TWO_POWER_D, cellDescription.toString());
          cellDescription.setNumberOfSharedAdjacentVertices(cellDescription.getNumberOfSharedAdjacentVertices() - 1);
          sendActualPatch = false;
        } else {
          assertion1(cellDescription.getNumberOfSharedAdjacentVertices() > 0, cellDescription.toString());
        }
      }
    }

    //Write changes back to heap
    CellDescriptionHeap::getInstance().getData(cellDescriptionIndex).at(0) = cellDescription;
  } else {
    sendActualPatch = false;
  }

  if(sendActualPatch) {
    _statistics.sentNeighborData();
    sendCellDescription(cellDescriptionIndex);

    CellDescription cellDescription = CellDescriptionHeap::getInstance().getData(cellDescriptionIndex).at(0);

    //TODO unterweg debug
    logDebug("", "Sending actual patch to " << _remoteRank
      << " at " << cellDescription.getPosition() << " on level " << cellDescription.getLevel()
    );

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
//      << ", position:" << position
//    );

    sendPaddingPatch(
      position,
      level,
      subgridSize
    );
  }
  #endif
  logTraceOut("sendPatch");
}

void peanoclaw::parallel::NeighbourCommunicator::sendPaddingPatch(
  const tarch::la::Vector<DIMENSIONS, double>& position,
  int                                          level,
  const tarch::la::Vector<DIMENSIONS, double>& subgridSize
) {
  logTraceInWith2Arguments("sendPaddingPatch", _position, _level);

  //TODO unterweg debug
  logDebug("", "Sending padding patch to " << _remoteRank << " at " << position << " on level " << level);

  _statistics.sentPaddingNeighborData();
  sendPaddingCellDescription(
    position,
    level,
    subgridSize
  );
  sendPaddingDataArray(); //UNew
//  sendPaddingDataArray(); //UOld
//  sendPaddingDataArray(); //Aux
  logTraceOut("sendPaddingPatch");
}

void peanoclaw::parallel::NeighbourCommunicator::receivePatch(int localCellDescriptionIndex) {
  #ifdef Parallel
  logTraceInWith3Arguments("receivePatch", localCellDescriptionIndex, _position, _level);

  assertion(localCellDescriptionIndex > -1);
  CellDescription localCellDescription = CellDescriptionHeap::getInstance().getData(localCellDescriptionIndex).at(0);

  std::vector<CellDescription> remoteCellDescriptionVector = CellDescriptionHeap::getInstance().receiveData(_remoteRank, _position, _level, peano::heap::NeighbourCommunication);

  //TODO unterweg debug
  logDebug("", "Receiving patch from " << _remoteRank << " at " << localCellDescription.getPosition() << " on level " << localCellDescription.getLevel());

  assertionEquals2(remoteCellDescriptionVector.size(), 1, _position, _level);
  CellDescription remoteCellDescription = remoteCellDescriptionVector[0];
  assertionEquals2(localCellDescription.getLevel(), _level, localCellDescription.toString(), tarch::parallel::Node::getInstance().getRank());

  //TODO unterweg debug
  logInfo("", "Receiving patch: " << remoteCellDescription.toString());

  assertion6(!remoteCellDescription.getIsRemote(), localCellDescription.toString(), remoteCellDescription.toString(), _position, _level, _remoteRank, tarch::parallel::Node::getInstance().getRank());
  assertionNumericalEquals6(localCellDescription.getPosition(), remoteCellDescription.getPosition(),
      localCellDescription.toString(), remoteCellDescription.toString(), tarch::parallel::Node::getInstance().getRank(), _remoteRank, _position, _level);
  assertionNumericalEquals4(localCellDescription.getSize(), remoteCellDescription.getSize(),
      localCellDescription.toString(), remoteCellDescription.toString(), tarch::parallel::Node::getInstance().getRank(), _remoteRank);
  assertionEquals4(localCellDescription.getLevel(), remoteCellDescription.getLevel(),
      localCellDescription.toString(), remoteCellDescription.toString(), tarch::parallel::Node::getInstance().getRank(), _remoteRank);

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
  CellDescriptionHeap::getInstance().getData(localCellDescriptionIndex).at(0) = remoteCellDescription;
  assertionEquals(CellDescriptionHeap::getInstance().getData(localCellDescriptionIndex).size(), 1);

  //TODO unterweg debug
//    logInfo("", "Merged: " << CellDescriptionHeap::getInstance().getData(localCellDescriptionIndex).at(0).toString());

  assertionEquals(CellDescriptionHeap::getInstance().getData(localCellDescriptionIndex).at(0).getCellDescriptionIndex(), localCellDescriptionIndex);
  logTraceOut("receivePatch");
  #endif
}

void peanoclaw::parallel::NeighbourCommunicator::receivePaddingPatch() {
  logTraceInWith2Arguments("receivePaddingPatch", _position, _level);
  #ifdef Parallel
  _statistics.receivedPaddingNeighborData();

  //TODO unterweg debug
  logDebug("", "Receiving padding patch from " << _remoteRank << " at " << _position << " on level " << _level);

  //Receive padding patch
  std::vector<CellDescription> remoteCellDescriptionVector = CellDescriptionHeap::getInstance().receiveData(_remoteRank, _position, _level, peano::heap::NeighbourCommunication);
  assertionEquals2(remoteCellDescriptionVector.size(), 0, _position, _level);
//  assertionNumericalEquals4(remoteCellDescriptionVector[0].getPosition(), _position, remoteCellDescriptionVector[0].toString(), _position, _level, _subgridSize);
//  assertionNumericalEquals4(remoteCellDescriptionVector[0].getSize(), _subgridSize, remoteCellDescriptionVector[0].toString(), _position, _level, _subgridSize);
//  assertionEquals4(remoteCellDescriptionVector[0].getLevel(), _level, remoteCellDescriptionVector[0].toString(), _position, _level, _subgridSize);

  //Aux
//  DataHeap::getInstance().receiveData(_remoteRank, _position, _level, peano::heap::NeighbourCommunication);
//
//  //UOld
//  DataHeap::getInstance().receiveData(_remoteRank, _position, _level, peano::heap::NeighbourCommunication);

  //UNew
  DataHeap::getInstance().receiveData(_remoteRank, _position, _level, peano::heap::NeighbourCommunication);
  #endif
  logTraceOut("receivePaddingPatch");
}

tarch::la::Vector<DIMENSIONS_PLUS_ONE, double> peanoclaw::parallel::NeighbourCommunicator::createRemoteSubgridKey() const {
  tarch::la::Vector<DIMENSIONS_PLUS_ONE, double> key;
  for(int d = 0; d < DIMENSIONS; d++) {
    key(d) = _position(d);
  }
  key(DIMENSIONS) = _level;
  return key;
}

peanoclaw::parallel::NeighbourCommunicator::NeighbourCommunicator(
  int                                         remoteRank,
  const tarch::la::Vector<DIMENSIONS,double>& position,
  int                                         level,
  const tarch::la::Vector<DIMENSIONS,double>& subgridSize,
  RemoteSubgridMap&                           remoteSubgridMap,
  peanoclaw::statistics::ParallelStatistics&  statistics
) : _remoteRank(remoteRank),
    _position(position),
    _level(level),
    _subgridSize(subgridSize),
//    _cellDescriptionHeap(CellDescriptionHeap::getInstance()),
//    _dataHeap(DataHeap::getInstance()),
    _remoteSubgridMap(remoteSubgridMap),
    _statistics(statistics),
    //En-/Disable optimizations
    _avoidSendingSubgridsThatAlreadyHaveBeenSent(false),
    _reduceNumberOfPaddingSubgrids(false),
    _reduceMultipleSends(false) {
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
    int localRank = tarch::parallel::Node::getInstance().getRank();
  /*  //TODO unterweg debug
//    logInfo("", "Sending to neighbor " << tarch::parallel::Node::getInstance().getRank()
//      << " to " << toRank
//      << ", position:" << x
//      << ", level:" << level);

    int neighborPatches = 0;
    int localPatches = 0;

    for(int i = 0; i < TWO_POWER_D; i++) {
      if(vertex.getAdjacentRanksDuringLastIteration()(i) == localRank) {
        int adjacentCellDescriptionIndex = vertex.getAdjacentCellDescriptionIndexInPeanoOrder(i);

        //TODO unterweg debug
        logInfo("", "Sending patch for index " << i);

        sendPatch(adjacentCellDescriptionIndex);
        localPatches++;
      } else if (vertex.getAdjacentRanksDuringLastIteration()(i) == _remoteRank) {
        neighborPatches++;
      }
    }

    logInfo("sendSubgridsForVertex", "Sending subgrids to " << _remoteRank << " at " << _position << " on level " << _level << ". local: " << localPatches << " neighbor: " << neighborPatches
        << " adjacentRanks: " << vertex.getAdjacentRanks() << " adjacentRanks(last iteration):" << vertex.getAdjacentRanksDuringLastIteration());
//    assertionEquals1(vertex.getAdjacentRanks(), vertex.getAdjacentRanksDuringLastIteration(), vertex.toString());

    //Send padding patches if remote rank has more patches than local rank
    for( ; localPatches < neighborPatches; localPatches++) {
      sendPaddingPatch(_position, _level, _subgridSize);
    }*/

//    vertex.setAdjacentRanksDuringLastIteration(vertex.getAdjacentRanks());
    for(int i = 0; i < TWO_POWER_D; i++) {
      if(vertex.getAdjacentRanks()(i) == localRank) {
        int adjacentCellDescriptionIndex = vertex.getAdjacentCellDescriptionIndexInPeanoOrder(i);
        sendPatch(adjacentCellDescriptionIndex);
      } else {
        sendPaddingPatch(_position, _level, _subgridSize);
      }
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
    /*assertionEquals(localVertex.isInside(), remoteVertex.isInside());
    assertionEquals(localVertex.isBoundary(), remoteVertex.isBoundary());

    tarch::la::Vector<TWO_POWER_D, int> neighbourVertexRanks = remoteVertex.getAdjacentRanksDuringLastIteration();

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

    logInfo("receiveSubgridsForVertex", "Receiving subgrids from " << _remoteRank << " at " << _position << " on level " << _level << ". local: " << localPatches << " neighbor: " << neighborPatches
        << " adjacentRanks: " << localVertex.getAdjacentRanks() << " adjacentRanks(last iteration):" << remoteVertex.getAdjacentRanksDuringLastIteration());
    assertionEquals1(localVertex.getAdjacentRanks(), localVertex.getAdjacentRanksDuringLastIteration(), localVertex.toString());

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

          //TODO unterweg debug
          logInfo("", "Receiving patch for index " << i);

          receivePatch(localAdjacentCellDescriptionIndex);
        } else {
          //Receive dummy message
          receivePaddingPatch();
        }
      }
    }*/

    for(int i = TWO_POWER_D - 1; i >= 0; i--) {
      int localAdjacentCellDescriptionIndex = localVertex.getAdjacentCellDescriptionIndexInPeanoOrder(i);
      int remoteAdjacentCellDescriptionIndex = remoteVertex.getAdjacentCellDescriptionIndexInPeanoOrder(i);

      if(remoteVertex.getAdjacentRanks()(i) == _remoteRank && remoteAdjacentCellDescriptionIndex != -1) {
          assertion(localAdjacentCellDescriptionIndex != -1);
          receivePatch(localAdjacentCellDescriptionIndex);
      } else {
        //Receive dummy message
        receivePaddingPatch();
      }
    }
  }
  #endif
}

void peanoclaw::parallel::NeighbourCommunicator::switchToRemote(peanoclaw::Patch& subgrid) {
  #ifdef Parallel
  subgrid.setIsRemote(true);
  tarch::la::Vector<DIMENSIONS_PLUS_ONE, double> key = createRemoteSubgridKey();

  _remoteSubgridMap[key] = subgrid.getCellDescriptionIndex();
  #endif
}
