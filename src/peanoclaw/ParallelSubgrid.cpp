/*
 * ParallelSubgrid.cpp
 *
 *  Created on: Aug 5, 2013
 *      Author: kristof
 */
#include "peanoclaw/ParallelSubgrid.h"

#include "peanoclaw/Region.h"
#include "peanoclaw/Heap.h"

#include "peano/utils/Globals.h"
#include "peano/utils/Loop.h"

tarch::logging::Log peanoclaw::ParallelSubgrid::_log("peanoclaw::ParallelSubgrid");

void peanoclaw::ParallelSubgrid::listRemoteRankAndAddSharedVertex(
  int  remoteRank,
  bool setRanks[THREE_POWER_D_MINUS_ONE]
) {
  #ifdef Parallel
  //Search slot to store remote rank
  for(int i = 0; i < THREE_POWER_D_MINUS_ONE; i++) {
    if(_cellDescription->getAdjacentRanks(i) == remoteRank
       || _cellDescription->getAdjacentRanks(i) == -1) {
      _cellDescription->setAdjacentRanks(i, remoteRank);
      if(!setRanks[i]) {
        setRanks[i] = true;
        _cellDescription->setNumberOfSharedAdjacentVertices(i,
          _cellDescription->getNumberOfSharedAdjacentVertices(i) + 1
        );
      }
      break;
    }
  }
  #endif
}

peanoclaw::ParallelSubgrid::ParallelSubgrid(
  CellDescription& cellDescription
) : _cellDescription(&cellDescription) {
}

peanoclaw::ParallelSubgrid::ParallelSubgrid(
  int subgridDescriptionIndex
) {
  _cellDescription = &CellDescriptionHeap::getInstance().getData(subgridDescriptionIndex).at(0);
}

peanoclaw::ParallelSubgrid::ParallelSubgrid(
  const Cell& cell
) {
  _cellDescription = &CellDescriptionHeap::getInstance().getData(cell.getCellDescriptionIndex()).at(0);
}

peanoclaw::ParallelSubgrid::ParallelSubgrid(
  Patch& subgrid
) {
  _cellDescription = &CellDescriptionHeap::getInstance().getData(subgrid.getCellDescriptionIndex()).at(0);
}

void peanoclaw::ParallelSubgrid::markCurrentStateAsSent(bool wasSent) {
  #ifdef Parallel
  _cellDescription->setCurrentStateWasSent(wasSent);
  #endif
}

void peanoclaw::ParallelSubgrid::markCurrentStateAsSentIfAppropriate() {
  #ifdef Parallel
  if(_cellDescription->getMarkStateAsSentInNextIteration()) {
    markCurrentStateAsSent(true);
    _cellDescription->setMarkStateAsSentInNextIteration(false);
  }
  #endif
}

bool peanoclaw::ParallelSubgrid::wasCurrentStateSent() const {
  #ifdef Parallel
  return _cellDescription->getCurrentStateWasSent();
  #else
  return false;
  #endif
}

void peanoclaw::ParallelSubgrid::markCurrentStateAsSentInNextIteration() {
  #ifdef Parallel
  _cellDescription->setMarkStateAsSentInNextIteration(true);
  #endif
}

tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int> peanoclaw::ParallelSubgrid::getAdjacentRanks() const {
  #ifdef Parallel
  return _cellDescription->getAdjacentRanks();
  #else
  return tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int>(-1);
  #endif
}

void peanoclaw::ParallelSubgrid::decreaseNumberOfSharedAdjacentVertices(int remoteRank) {
  #ifdef Parallel
  for(int i = 0; i < THREE_POWER_D_MINUS_ONE; i++) {
    if(_cellDescription->getAdjacentRanks(i) == remoteRank) {
      _cellDescription->setNumberOfSharedAdjacentVertices(
        i, _cellDescription->getNumberOfSharedAdjacentVertices(i) - 1
      );
      break;
    }
  }
  #endif
}

tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int> peanoclaw::ParallelSubgrid::getNumberOfSharedAdjacentVertices() const {
  #ifdef Parallel
  return _cellDescription->getNumberOfSharedAdjacentVertices();
  #else
  return 0;
  #endif
}

int peanoclaw::ParallelSubgrid::getNumberOfSharedAdjacentVertices(int remoteRank) const {
  #ifdef Parallel
  for(int i = 0; i < THREE_POWER_D_MINUS_ONE; i++) {
    if(_cellDescription->getAdjacentRanks(i) == remoteRank) {
     return _cellDescription->getNumberOfSharedAdjacentVertices(i);
    }
  }
  assertionFail("Should not occur! remoteRank=" << remoteRank << ", adjacentRanks=" << _cellDescription->getAdjacentRanks());
  #endif
  return -1;
}

tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int> peanoclaw::ParallelSubgrid::getAllNumbersOfTransfersToBeSkipped() const {
  #ifdef Parallel
  return _cellDescription->getNumberOfTransfersToBeSkipped();
  #else
  return 0;
  #endif
}

int peanoclaw::ParallelSubgrid::getNumberOfTransfersToBeSkipped() const {
  #ifdef Parallel
  int localRank = tarch::parallel::Node::getInstance().getRank();
  for(int i = 0; i < THREE_POWER_D_MINUS_ONE; i++) {
    if(_cellDescription->getAdjacentRanks(i) == localRank) {
      return _cellDescription->getNumberOfTransfersToBeSkipped(i);
    }
  }
  return 0;
  #else
  return 0;
  #endif
}

void peanoclaw::ParallelSubgrid::decreaseNumberOfTransfersToBeSkipped() {
  #ifdef Parallel
  int localRank = tarch::parallel::Node::getInstance().getRank();
  for(int i = 0; i < THREE_POWER_D_MINUS_ONE; i++) {
    if(_cellDescription->getAdjacentRanks(i) == localRank) {
      _cellDescription->setNumberOfTransfersToBeSkipped(i,
        _cellDescription->getNumberOfTransfersToBeSkipped(i) - 1
      );
      break;
    }
  }
  #endif
}

void peanoclaw::ParallelSubgrid::resetNumberOfTransfersToBeSkipped() {
  #ifdef Parallel
  _cellDescription->setNumberOfTransfersToBeSkipped(0);
  #endif
}

void peanoclaw::ParallelSubgrid::countNumberOfAdjacentParallelSubgrids (
  peanoclaw::Vertex * const            vertices,
  const peano::grid::VertexEnumerator& verticesEnumerator
) {
  #ifdef Parallel
  for(int i = 0; i < THREE_POWER_D_MINUS_ONE; i++) {
    _cellDescription->setAdjacentRanks(i, -1);
    _cellDescription->setNumberOfSharedAdjacentVertices(i, 0);
  }

  //Set adjacent ranks
  for(int vertexIndex = 0; vertexIndex < TWO_POWER_D; vertexIndex++) {
    tarch::la::Vector<DIMENSIONS, int> vertexPosition = peano::utils::dDelinearised(vertexIndex, 2);

    for(int subgridIndex = 0; subgridIndex < TWO_POWER_D; subgridIndex++) {
      tarch::la::Vector<DIMENSIONS, int> subgridPosition = (2*vertexPosition + 2*peano::utils::dDelinearised(subgridIndex, 2)) / 2 - 1;

      if(!tarch::la::equals(subgridPosition, 0)) {
        int entry = Region::linearizeManifoldPosition(subgridPosition);
        int adjacentRank = vertices[verticesEnumerator(vertexIndex)].getAdjacentRanks()(subgridIndex);

        _cellDescription->setAdjacentRanks(entry, adjacentRank);
        }
      }
  }

  //Count shared vertices
  for(int vertexIndex = 0; vertexIndex < TWO_POWER_D; vertexIndex++) {
    tarch::la::Vector<TWO_POWER_D, int> countedForRanks(-1);

    if(!vertices[verticesEnumerator(vertexIndex)].isHangingNode()) {
      for(int subgridIndex = 0; subgridIndex < TWO_POWER_D; subgridIndex++) {
        int adjacentRank = vertices[verticesEnumerator(vertexIndex)].getAdjacentRanks()(subgridIndex);

        bool counted = false;
        for(int i = 0; i < TWO_POWER_D; i++) {
          counted |= (countedForRanks(i) == adjacentRank);
        }

        if(!counted) {
          for(int i = 0; i < TWO_POWER_D; i++) {
            if(countedForRanks(i) == -1 || countedForRanks(i) == adjacentRank) {
              countedForRanks(i) = adjacentRank;
              break;
            }
          }
          for(int i = 0; i < THREE_POWER_D_MINUS_ONE; i++) {
            if(_cellDescription->getAdjacentRanks(i) == -1 || _cellDescription->getAdjacentRanks(i) == adjacentRank) {
              _cellDescription->setAdjacentRanks(i, adjacentRank);
              _cellDescription->setNumberOfSharedAdjacentVertices(i, _cellDescription->getNumberOfSharedAdjacentVertices(i) + 1);
              break;
            }
          }
        }
      }
    }
  }
  _cellDescription->setNumberOfTransfersToBeSkipped(_cellDescription->getNumberOfSharedAdjacentVertices() - 1);
  #endif
}

void peanoclaw::ParallelSubgrid::setOverlapOfRemoteGhostlayer(int subgridIndex, int overlap) {
  #ifdef Parallel
  _cellDescription->setOverlapByRemoteGhostlayer(subgridIndex, overlap);
  #endif
}

int peanoclaw::ParallelSubgrid::getOverlapOfRemoteGhostlayer(int subgridIndex) const {
  #ifdef Parallel
  return _cellDescription->getOverlapByRemoteGhostlayer(subgridIndex);
  #else
  return 0;
  #endif
}

tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int> peanoclaw::ParallelSubgrid::getOverlapOfRemoteGhostlayers() const {
  #ifdef Parallel
  return _cellDescription->getOverlapByRemoteGhostlayer();
  #else
  return 0;
  #endif
}

bool peanoclaw::ParallelSubgrid::isAdjacentToLocalSubdomain(
  const peanoclaw::Cell&               coarseGridCell,
  peanoclaw::Vertex * const            fineGridVertices,
  const peano::grid::VertexEnumerator& fineGridVerticesEnumerator
) {
  #ifdef Parallel
  bool isAdjacentToLocalDomain = !coarseGridCell.isAssignedToRemoteRank();
  for(int i = 0; i < TWO_POWER_D; i++) {
    isAdjacentToLocalDomain |= fineGridVertices[fineGridVerticesEnumerator(i)].isAdjacentToDomainOf(
        tarch::parallel::Node::getInstance().getRank()
      );
  }
  return isAdjacentToLocalDomain;
  #else
  return true;
  #endif
}

