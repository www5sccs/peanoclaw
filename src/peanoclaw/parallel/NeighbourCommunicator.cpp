/*
 * NeighbourCommunicator.cpp
 *
 *  Created on: Mar 15, 2013
 *      Author: unterweg
 */
#include "NeighbourCommunicator.h"

#include "peanoclaw/Heap.h"
#include "peanoclaw/ParallelSubgrid.h"
#include "peanoclaw/records/CellDescription.h"
#include "peanoclaw/records/Data.h"

#include "peano/utils/Loop.h"

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

void peanoclaw::parallel::NeighbourCommunicator::deleteArraysFromPatch(Patch& subgrid) {
  logTraceInWith1Argument("deleteArraysFromPatch", cellDescriptionIndex);

  if(subgrid.getUNewIndex() != -1) {
    DataHeap::getInstance().deleteData(subgrid.getUNewIndex());
  }
//    if(cellDescription.getUOldIndex() != -1) {
//      DataHeap::getInstance().deleteData(cellDescription.getUOldIndex());
//    }
//    if(cellDescription.getAuxIndex() != -1) {
//      DataHeap::getInstance().deleteData(cellDescription.getAuxIndex());
//    }
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
  const Patch& transferedSubgrid
) {
  logTraceInWith3Arguments("sendPatch", cellDescriptionIndex, _position, _level);
  #ifdef Parallel

  #ifdef Asserts
  assertion3(tarch::la::allGreater(transferedSubgrid.getSubdivisionFactor(), 0), transferedSubgrid.toString(), _position, _level);

  //Check for zeros in transfered patch
  if(transferedSubgrid.isValid() && transferedSubgrid.isLeaf()) {
    dfor(subcellIndex, transferedSubgrid.getSubdivisionFactor()) {
      assertion3(tarch::la::greater(transferedSubgrid.getValueUNew(subcellIndex, 0), 0.0), subcellIndex, transferedSubgrid, transferedSubgrid.toStringUNew());
      assertion3(tarch::la::greater(transferedSubgrid.getValueUOld(subcellIndex, 0), 0.0), subcellIndex, transferedSubgrid, transferedSubgrid.toStringUOldWithGhostLayer());
    }
  }
  #endif

  tarch::la::Vector<DIMENSIONS, double> position;
  tarch::la::Vector<DIMENSIONS, double> subgridSize;
  position    = transferedSubgrid.getPosition();
  subgridSize = transferedSubgrid.getSize();

  assertion1(!transferedSubgrid.isRemote(), transferedSubgrid);

  _statistics.sentNeighborData();
  sendCellDescription(transferedSubgrid.getCellDescriptionIndex());

  if(transferedSubgrid.getUNewIndex() != -1) {
    sendDataArray(transferedSubgrid.getUNewIndex());
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

void peanoclaw::parallel::NeighbourCommunicator::receivePatch(Patch& localSubgrid) {
  #ifdef Parallel
  logTraceInWith3Arguments("receivePatch", localCellDescriptionIndex, _position, _level);

  //CellDescription localCellDescription = CellDescriptionHeap::getInstance().getData(localCellDescriptionIndex).at(0);

  std::vector<CellDescription> remoteCellDescriptionVector = CellDescriptionHeap::getInstance().receiveData(_remoteRank, _position, _level, peano::heap::NeighbourCommunication);

  logDebug("", "Receiving patch from " << _remoteRank << " at " << localCellDescription.getPosition() << " on level " << localCellDescription.getLevel());

  if(!_onlySendSubgridsAfterChange || remoteCellDescriptionVector.size() > 0) {
    assertionEquals3(remoteCellDescriptionVector.size(), 1, _position, _level, localSubgrid);
    CellDescription remoteCellDescription = remoteCellDescriptionVector[0];
    assertionEquals2(localSubgrid.getLevel(), _level, localSubgrid, tarch::parallel::Node::getInstance().getRank());

    logDebug("", "Receiving patch: " << remoteCellDescription.toString());

    assertion6(!remoteCellDescription.getIsRemote(), localSubgrid, remoteCellDescription.toString(), _position, _level, _remoteRank, tarch::parallel::Node::getInstance().getRank());
    assertionNumericalEquals6(localSubgrid.getPosition(), remoteCellDescription.getPosition(),
        localSubgrid, remoteCellDescription.toString(), tarch::parallel::Node::getInstance().getRank(), _remoteRank, _position, _level);
    assertionNumericalEquals4(localSubgrid.getSize(), remoteCellDescription.getSize(),
        localSubgrid, remoteCellDescription.toString(), tarch::parallel::Node::getInstance().getRank(), _remoteRank);
    assertionEquals4(localSubgrid.getLevel(), remoteCellDescription.getLevel(),
        localSubgrid, remoteCellDescription.toString(), tarch::parallel::Node::getInstance().getRank(), _remoteRank);

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
    deleteArraysFromPatch(localSubgrid);
    remoteCellDescription.setCellDescriptionIndex(localSubgrid.getCellDescriptionIndex());
    remoteCellDescription.setIsRemote(true); //TODO unterweg: Remote patches are currently never destroyed.
    CellDescriptionHeap::getInstance().getData(localSubgrid.getCellDescriptionIndex()).at(0) = remoteCellDescription;
    assertionEquals(CellDescriptionHeap::getInstance().getData(localSubgrid.getCellDescriptionIndex()).size(), 1);

    //Initialize non-parallel fields
    Patch remotePatch(remoteCellDescription);
    remotePatch.initializeNonParallelFields();

    //Check for zeros in transfered patch
    #ifdef Asserts
    if(remotePatch.isValid() && remotePatch.isLeaf()) {
      dfor(subcellIndex, remotePatch.getSubdivisionFactor()) {
        assertion3(tarch::la::greater(remotePatch.getValueUNew(subcellIndex, 0), 0.0), subcellIndex, remotePatch, remotePatch.toStringUNew());
        assertion3(tarch::la::greater(remotePatch.getValueUOld(subcellIndex, 0), 0.0), subcellIndex, remotePatch, remotePatch.toStringUOldWithGhostLayer());
      }
    }
    #endif

    assertionEquals(CellDescriptionHeap::getInstance().getData(localSubgrid.getCellDescriptionIndex()).at(0).getCellDescriptionIndex(), localSubgrid.getCellDescriptionIndex());
  } else {
    //Padding patch received -> receive padding data
    DataHeap::getInstance().receiveData(_remoteRank, _position, _level, peano::heap::NeighbourCommunication);
  }

  logTraceOut("receivePatch");
  #endif
}

void peanoclaw::parallel::NeighbourCommunicator::receivePaddingPatch() {
  logTraceInWith2Arguments("receivePaddingPatch", _position, _level);
  #ifdef Parallel
  _statistics.receivedPaddingNeighborData();

  logDebug("", "Receiving padding patch from " << _remoteRank << " at " << _position << " on level " << _level);

  //Receive padding patch
  std::vector<CellDescription> remoteCellDescriptionVector = CellDescriptionHeap::getInstance().receiveData(_remoteRank, _position, _level, peano::heap::NeighbourCommunication);
  assertionEquals3(remoteCellDescriptionVector.size(), 0, _position, _level, _remoteRank);

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

void peanoclaw::parallel::NeighbourCommunicator::createOrFindRemoteSubgrid(
  Vertex& localVertex,
  int     adjacentSubgridIndexInPeanoOrder,
  const tarch::la::Vector<DIMENSIONS, double>& subgridSize
) {
  tarch::la::Vector<DIMENSIONS, double> subgridPosition
    = _position + tarch::la::multiplyComponents(subgridSize, peano::utils::dDelinearised(adjacentSubgridIndexInPeanoOrder, 2).convertScalar<double>() - 1.0);
  tarch::la::Vector<DIMENSIONS_PLUS_ONE, double> key = createRemoteSubgridKey(subgridPosition, _level);
   if(_remoteSubgridMap.find(key) != _remoteSubgridMap.end()) {
     localVertex.setAdjacentCellDescriptionIndexInPeanoOrder(adjacentSubgridIndexInPeanoOrder, _remoteSubgridMap[key]);
   } else {
    Patch outsidePatch(
      subgridPosition,
      subgridSize,
      0,
      0,
      1,
      1,
      1.0,
      _level
    );
    localVertex.setAdjacentCellDescriptionIndexInPeanoOrder(adjacentSubgridIndexInPeanoOrder, outsidePatch.getCellDescriptionIndex());
    _remoteSubgridMap[key] = outsidePatch.getCellDescriptionIndex();
   }
}

tarch::la::Vector<DIMENSIONS_PLUS_ONE, double> peanoclaw::parallel::NeighbourCommunicator::createRemoteSubgridKey(
  const tarch::la::Vector<DIMENSIONS, double> subgridPosition,
  int                                         level
) const {
  tarch::la::Vector<DIMENSIONS_PLUS_ONE, double> key;
  for(int d = 0; d < DIMENSIONS; d++) {
    key(d) = subgridPosition(d);
  }
  key(DIMENSIONS) = level;
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
    _remoteSubgridMap(remoteSubgridMap),
    _statistics(statistics),
    //En-/Disable optimizations
    _avoidMultipleTransferOfSubgridsIfPossible(false),
    _reduceNumberOfPaddingSubgrids(false),
    _onlySendSubgridsAfterChange(true) {
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

    for(int i = 0; i < TWO_POWER_D; i++) {
      bool sentSubgrid = false;
      if(vertex.getAdjacentRanks()(i) == localRank && vertex.getAdjacentCellDescriptionIndexInPeanoOrder(i) != -1) {
        Patch           localSubgrid(vertex.getAdjacentCellDescriptionIndexInPeanoOrder(i));
        ParallelSubgrid localParallelSubgrid(vertex.getAdjacentCellDescriptionIndexInPeanoOrder(i));

        if(
            !localParallelSubgrid.wasCurrentStateSent()
            || vertex.wereAdjacentRanksChanged()
            || vertex.getAgeInGridIterations() <= 1
            //TODO unterweg dissertion
            //Warum kann ich nicht einfach den Status der Feingitter restringieren, ob die sich geändert haben?
            //Gibt es da einen Zusammenhang zwischen Zeitschritten der benachbarten Grobgitter und den Zeitschritten
            //der Feingitter, den ich hier nicht erkenne?
            //TODO unterweg debug
            || !localSubgrid.isLeaf()
            || !_onlySendSubgridsAfterChange) {

          if(
              localParallelSubgrid.getAdjacentRank() == -1
              || localParallelSubgrid.getNumberOfSharedAdjacentVertices() == 1
              || !_avoidMultipleTransferOfSubgridsIfPossible
          ) {
            assertion3(
              localParallelSubgrid.getAdjacentRank() == _remoteRank || localParallelSubgrid.getAdjacentRank() == -1,
              _remoteRank,
              localSubgrid,
              vertex.getAdjacentRanks()
            );

            logDebug("sendSubgridsForVertex", "Sending subgrid to rank " << _remoteRank << ": " << localSubgrid << " for vertex " << _position);

            sendPatch(localSubgrid);

            sentSubgrid = true;
          } else {
            logDebug("sendSubgridsForVertex", "(Skipped) Sending subgrid to rank " << _remoteRank << ": " << localSubgrid
                << " for vertex " << _position << ", getNumberOfSharedAdjacentVertices=" << localParallelSubgrid.getNumberOfSharedAdjacentVertices());
          }
          localParallelSubgrid.decreaseNumberOfSharedAdjacentVertices();
        }
      }

      if(!sentSubgrid) {
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
    for(int i = TWO_POWER_D - 1; i >= 0; i--) {
      int localAdjacentCellDescriptionIndex = localVertex.getAdjacentCellDescriptionIndexInPeanoOrder(i);
      int remoteAdjacentCellDescriptionIndex = remoteVertex.getAdjacentCellDescriptionIndexInPeanoOrder(i);

      bool receivedSubgrid = false;
      if(remoteVertex.getAdjacentRanks()(i) == _remoteRank && remoteAdjacentCellDescriptionIndex != -1) {

          //Create remote subgrid if necessary
          if(localVertex.getAdjacentCellDescriptionIndexInPeanoOrder(i) == -1 && localVertex.getAdjacentRanks()(i) != 0) {
            createOrFindRemoteSubgrid(localVertex, i, adjacentSubgridSize);
            localAdjacentCellDescriptionIndex = localVertex.getAdjacentCellDescriptionIndexInPeanoOrder(i);
          }

          assertion(localAdjacentCellDescriptionIndex != -1);
          Patch           localSubgrid(localAdjacentCellDescriptionIndex);
          ParallelSubgrid localParallelSubgrid(localAdjacentCellDescriptionIndex);

          //Only receive if the incoming transfer was not skipped to avoid multiple send of the subgrid.
          if(localParallelSubgrid.getNumberOfTransfersToBeSkipped() == 0 || !_avoidMultipleTransferOfSubgridsIfPossible) {

            receivePatch(localSubgrid);
            receivedSubgrid = true;

            localSubgrid.reloadCellDescription();
            logDebug("receiveSubgridsForVertex", "Received subgrid from rank " << _remoteRank << ": " << localSubgrid
                << " for vertex " << _position);
          } else {

            assertion1(localSubgrid.isRemote(), localSubgrid);

            localParallelSubgrid.decreaseNumberOfTransfersToBeSkipped();

            logDebug("receiveSubgridsForVertex", "(Skipped) Received subgrid from rank " << _remoteRank << ": " << localSubgrid
                << " for vertex " << _position << ",getNumberOfTransfersToBeSkipped=" << localParallelSubgrid.getNumberOfTransfersToBeSkipped());
          }
      }


      if(!receivedSubgrid) {
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
  tarch::la::Vector<DIMENSIONS_PLUS_ONE, double> key = createRemoteSubgridKey(subgrid.getPosition(), subgrid.getLevel());

  _remoteSubgridMap[key] = subgrid.getCellDescriptionIndex();
  #endif
}
