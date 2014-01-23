/*
 * ParallelSubgrid.cpp
 *
 *  Created on: Aug 5, 2013
 *      Author: kristof
 */
#include "peanoclaw/ParallelSubgrid.h"

#include "peanoclaw/Heap.h"

#include "peano/utils/Globals.h"

tarch::logging::Log peanoclaw::ParallelSubgrid::_log("peanoclaw::ParallelSubgrid");

void peanoclaw::ParallelSubgrid::listRemoteRankAndAddSharedVertex(
  int  remoteRank,
  bool setRanks[THREE_POWER_D_MINUS_ONE]
) {
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
  _cellDescription->setCurrentStateWasSend(wasSent);
  #endif
}

bool peanoclaw::ParallelSubgrid::wasCurrentStateSent() const {
  #ifdef Parallel
  return _cellDescription->getCurrentStateWasSend();
  #else
  return false;
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

tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int> peanoclaw::ParallelSubgrid::getAdjacentRanks() const {
  #ifdef Parallel
  return _cellDescription->getAdjacentRanks();
  #else
  return tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int>(-1);
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
      _cellDescription->getNumberOfSharedAdjacentVertices(i);
    }
  }
  assertionFail("Should not occur!");
  return -1;
  #else
  return 0;
  #endif
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
      _cellDescription->setNumberOfTransfersToBeSkipped(
        _cellDescription->getNumberOfTransfersToBeSkipped() - 1
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

void peanoclaw::ParallelSubgrid::countNumberOfAdjacentParallelSubgridsAndSetGhostlayerOverlap (
  peanoclaw::Vertex * const            vertices,
  const peano::grid::VertexEnumerator& verticesEnumerator
) {
  #ifdef Parallel
  for(int i = 0; i < THREE_POWER_D_MINUS_ONE; i++) {
    _cellDescription->setAdjacentRanks(i, -1);
    _cellDescription->setNumberOfSharedAdjacentVertices(i, 0);
  }

  int entry = 0;
  for(int vertexIndex = 0; vertexIndex < TWO_POWER_D; vertexIndex++) {
    bool setRanks[THREE_POWER_D_MINUS_ONE];
    for(int i = 0; i < THREE_POWER_D_MINUS_ONE; i++) {
      setRanks[i] = false;
    }

    for(int subgridIndex = 0; subgridIndex < TWO_POWER_D; subgridIndex++) {
      int adjacentRank = vertices[verticesEnumerator(vertexIndex)].getAdjacentRanks()(subgridIndex);

      if(adjacentRank != tarch::parallel::Node::getInstance().getRank()) {

        //Count shared vertices and add adjacent ranks
        listRemoteRankAndAddSharedVertex(adjacentRank, setRanks);

        int adjacentSubgridDescriptionIndex
          = vertices[verticesEnumerator(vertexIndex)].getAdjacentCellDescriptionIndexInPeanoOrder(subgridIndex);
        if(adjacentSubgridDescriptionIndex == -1) {
          //No adjacent subgrid present. -> Not copied from the remote rank, yet. -> Copy complete ghostlayer.

        } else {
          Patch adjacentSubgrid(adjacentSubgridDescriptionIndex);
          tarch::la::Vector<DIMENSIONS, double> continuousGhostlayerWidth
            = (double)adjacentSubgrid.getGhostlayerWidth() * adjacentSubgrid.getSubcellSize();
        }
      }
    }
  }

  _cellDescription->setNumberOfTransfersToBeSkipped(_cellDescription->getNumberOfSharedAdjacentVertices() - 1);
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

