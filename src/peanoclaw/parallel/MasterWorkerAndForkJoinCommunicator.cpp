/*
 * MasterWorkerAndForkJoinCommunicator.cpp
 *
 *  Created on: Mar 19, 2013
 *      Author: kristof
 */
#include "MasterWorkerAndForkJoinCommunicator.h"

#include "peanoclaw/Cell.h"
#include "peanoclaw/Heap.h"
#include "peanoclaw/Patch.h"
#include "peanoclaw/ParallelSubgrid.h"
#include "peanoclaw/records/CellDescription.h"
#include "peanoclaw/records/Data.h"

#include "tarch/parallel/Node.h"
#include "tarch/parallel/NodePool.h"

tarch::logging::Log peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator::_log("peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator");

void peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator::sendCellDescription(int cellDescriptionIndex) {
  logTraceInWith1Argument("sendCellDescription", cellDescriptionIndex);
  CellDescriptionHeap::getInstance().sendData(cellDescriptionIndex, _remoteRank, _position, _level, _messageType);
  logTraceOut("sendCellDescription");
}

void peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator::sendDataArray(int index) {
  logTraceInWith3Arguments("sendDataArray", index, _position, _level);
  DataHeap::getInstance().sendData(index, _remoteRank, _position, _level, _messageType);
  logTraceOut("sendDataArray");
}

void peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator::deleteArraysFromPatch(int cellDescriptionIndex) {
  logTraceInWith1Argument("deleteArraysFromPatch", cellDescriptionIndex);
  if(cellDescriptionIndex != -1) {
    assertion2(CellDescriptionHeap::getInstance().isValidIndex(cellDescriptionIndex), _position, _level);
    CellDescription cellDescription = CellDescriptionHeap::getInstance().getData(cellDescriptionIndex).at(0);

    if(cellDescription.getUIndex() != -1) {
      DataHeap::getInstance().deleteData(cellDescription.getUIndex());
    }
  }
  logTraceOut("deleteArraysFromPatch");
}

int peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator::receiveDataArray() {
  logTraceIn("receiveDataArray");
  std::vector<Data> remoteArray = DataHeap::getInstance().receiveData(_remoteRank, _position, _level, _messageType);

  int localIndex = DataHeap::getInstance().createData();
  std::vector<Data>& localArray = DataHeap::getInstance().getData(localIndex);

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
) : _subgridCommunicator(remoteRank, position, level, forkOrJoin ? peano::heap::ForkOrJoinCommunication : peano::heap::MasterWorkerCommunication),
    _remoteRank(remoteRank),
    _position(position),
    _level(level),
//    _cellDescriptionHeap(CellDescriptionHeap::getInstance()),
//    _dataHeap(DataHeap::getInstance()),
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

    CellDescription cellDescription = CellDescriptionHeap::getInstance().getData(cellDescriptionIndex).at(0);

    if(cellDescription.getUIndex() != -1) {
      sendDataArray(cellDescription.getUIndex());
    }
  }
  logTraceOut("sendPatch");
}

void peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator::receivePatch(int localCellDescriptionIndex) {
  logTraceInWith3Arguments("receivePatch", localCellDescriptionIndex, _position, _level);
  #ifdef Parallel

  std::vector<CellDescription> remoteCellDescriptionVector = CellDescriptionHeap::getInstance().receiveData(_remoteRank, _position, _level, _messageType);
  assertionEquals2(remoteCellDescriptionVector.size(), 1, _position, _level);
  CellDescription remoteCellDescription = remoteCellDescriptionVector[0];

  assertion3(localCellDescriptionIndex >= 0, localCellDescriptionIndex, _position, _level);
  CellDescription localCellDescription = CellDescriptionHeap::getInstance().getData(localCellDescriptionIndex).at(0);
  #ifdef Asserts
  assertionNumericalEquals2(remoteCellDescription.getPosition(), localCellDescription.getPosition(), localCellDescription.toString(), remoteCellDescription.toString());
  assertionNumericalEquals2(remoteCellDescription.getSize(), localCellDescription.getSize(), localCellDescription.toString(), remoteCellDescription.toString());
  assertionEquals2(remoteCellDescription.getLevel(), localCellDescription.getLevel(), localCellDescription.toString(), remoteCellDescription.toString());
  #endif

  //TODO unterweg debug
//  std::cout << "Received cell description: " << remoteCellDescription.toString() << std::endl;

  //Load arrays and stores according indices in cell description
  if(remoteCellDescription.getUIndex() != -1) {
    remoteCellDescription.setUIndex(receiveDataArray());
  }

  //Reset undesired values
  remoteCellDescription.setNumberOfTransfersToBeSkipped(0);

  //Copy remote cell description to local cell description
  deleteArraysFromPatch(localCellDescriptionIndex);
  remoteCellDescription.setCellDescriptionIndex(localCellDescriptionIndex);
  CellDescriptionHeap::getInstance().getData(localCellDescriptionIndex).at(0) = remoteCellDescription;
  assertionEquals(CellDescriptionHeap::getInstance().getData(localCellDescriptionIndex).size(), 1);

  assertionEquals(CellDescriptionHeap::getInstance().getData(localCellDescriptionIndex).at(0).getCellDescriptionIndex(), localCellDescriptionIndex);
  #endif
  logTraceOut("receivePatch");
}

void peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator::sendCellDuringForkOrJoin(
  const Cell& localCell,
  const tarch::la::Vector<DIMENSIONS, double>& position,
  const tarch::la::Vector<DIMENSIONS, double>& size,
  const State state
) {
  #ifdef Parallel
  bool isForking = tarch::parallel::Node::getInstance().isGlobalMaster()
                   || _remoteRank != tarch::parallel::NodePool::getInstance().getMasterRank();

  if(localCell.isInside()
      && (
           (isForking && localCell.getRankOfRemoteNode() == _remoteRank)
        || (!isForking && !localCell.isAssignedToRemoteRank())
      )
    ) {
    sendPatch(localCell.getCellDescriptionIndex());

    //Switch to remote after having sent the patch away...
    Patch patch(localCell);
    patch.setIsRemote(true);
    ParallelSubgrid parallelSubgrid(localCell);
    parallelSubgrid.resetNumberOfTransfersToBeSkipped();
  }
  #endif
}

void peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator::mergeCellDuringForkOrJoin(
  peanoclaw::Cell&                      localCell,
  const peanoclaw::Cell&                remoteCell,
  tarch::la::Vector<DIMENSIONS, double> cellSize,
  const peanoclaw::State&               state
) {
  #ifdef Parallel
  bool isForking = !tarch::parallel::Node::getInstance().isGlobalMaster()
                   && _remoteRank == tarch::parallel::NodePool::getInstance().getMasterRank();

  if(localCell.isInside()
      && (
           ( isForking && !remoteCell.isAssignedToRemoteRank())
        || (!isForking && remoteCell.getRankOfRemoteNode() == _remoteRank)
      )
    ) {
    if(localCell.isRemote(state, false, false)) {
      if(tarch::parallel::NodePool::getInstance().getMasterRank() != 0) {
        assertionEquals2(localCell.getCellDescriptionIndex(), -2, _position, _level);
      }
      Patch temporaryPatch(
        _position - cellSize * 0.5,
        cellSize,
        1,
        0,
        1,
        1,
        0.0,
        _level
      );

      receivePatch(temporaryPatch.getCellDescriptionIndex());
      temporaryPatch.reloadCellDescription();

      if(temporaryPatch.isLeaf()) {
        temporaryPatch.switchToVirtual();
      }
      if(temporaryPatch.isVirtual()) {
        temporaryPatch.switchToNonVirtual();
      }
      CellDescriptionHeap::getInstance().deleteData(temporaryPatch.getCellDescriptionIndex());
    } else {
      assertion2(localCell.getCellDescriptionIndex() != -1, _position, _level);

      Patch localPatch(localCell);

      assertion2(
        (!localPatch.isLeaf() && !localPatch.isVirtual())
        || localPatch.getUNewIndex() >= 0,
        _position,
        _level
      );

      receivePatch(localPatch.getCellDescriptionIndex());
      localPatch.loadCellDescription(localCell.getCellDescriptionIndex());

      assertion1(!localPatch.isRemote(), localPatch);

      //TODO unterweg dissertation: Wenn auf dem neuen Knoten die Adjazenzinformationen auf den
      // Vertices noch nicht richtig gesetzt sind k��nnen wir nicht gleich voranschreiten.
      // U.U. brauchen wir sogar 2 Iterationen ohne Aktion... (Wegen hin- und herlaufen).
      localPatch.setSkipNextGridIteration(2);

      assertionEquals1(localPatch.getLevel(), _level, localPatch);
    }
  }
  #endif
}

void peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator::mergeWorkerStateIntoMasterState(
  const peanoclaw::State&          workerState,
  peanoclaw::State&                masterState
) {
  #ifdef Parallel
  masterState.updateGlobalTimeIntervals(
        workerState.getStartMaximumGlobalTimeInterval(),
        workerState.getEndMaximumGlobalTimeInterval(),
        workerState.getStartMinimumGlobalTimeInterval(),
        workerState.getEndMinimumGlobalTimeInterval()
  );

  masterState.updateMinimalTimestep(workerState.getMinimalTimestep());

  //TODO unterweg debug
  logDebug("", "On rank " << tarch::parallel::Node::getInstance().getRank()
      << " master.allPatchesEvolved=" << masterState.getAllPatchesEvolvedToGlobalTimestep()
      << " worker.allPatchesEvolved=" << workerState.getAllPatchesEvolvedToGlobalTimestep());

  bool allPatchesEvolvedToGlobalTimestep = workerState.getAllPatchesEvolvedToGlobalTimestep();
  allPatchesEvolvedToGlobalTimestep &= masterState.getAllPatchesEvolvedToGlobalTimestep();

  masterState.setAllPatchesEvolvedToGlobalTimestep(allPatchesEvolvedToGlobalTimestep);

  masterState.updateLocalHeightOfWorkerTree(workerState.getLocalHeightOfWorkerTree());
  #endif
}
