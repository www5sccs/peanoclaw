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

int peanoclaw::ParallelSubgrid::getNumberOfTransfersToBeSkipped() const {
  #ifdef Parallel
  return _cellDescription->getNumberOfSkippedTransfers();
  #else
  return 0;
  #endif
}

void peanoclaw::ParallelSubgrid::decreaseNumberOfTransfersToBeSkipped() {
  #ifdef Parallel
  _cellDescription->setNumberOfSkippedTransfers(_cellDescription->getNumberOfSkippedTransfers() - 1);
  #endif
}

void peanoclaw::ParallelSubgrid::resetNumberOfTransfersToBeSkipped() {
  #ifdef Parallel
  _cellDescription->setNumberOfSkippedTransfers(0);
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
  for(int vertexIndex = 0; vertexIndex < TWO_POWER_D; vertexIndex++) {
    bool setRanks[THREE_POWER_D_MINUS_ONE];
    for(int i = 0; i < THREE_POWER_D_MINUS_ONE; i++) {
      setRanks[i] = false;
    }

    for(int subgridIndex = 0; subgridIndex < TWO_POWER_D; subgridIndex++) {
      int adjacentRank = vertices[verticesEnumerator(vertexIndex)].getAdjacentRanks()(subgridIndex);

      if(adjacentRank != tarch::parallel::Node::getInstance().getRank()) {
        //Search slot to store remote rank
        for(int i = 0; i < THREE_POWER_D_MINUS_ONE; i++) {
          if(_cellDescription->getAdjacentRanks(i) == adjacentRank
             || _cellDescription->getAdjacentRanks(i) == -1) {
            _cellDescription->setAdjacentRanks(i, adjacentRank);
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
    }
  }
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

void peanoclaw::ParallelSubgrid::setAdjacentRanksAndRemoteGhostlayerOverlap(
  peanoclaw::Vertex * const            fineGridVertices,
  const peano::grid::VertexEnumerator& fineGridVerticesEnumerator
) {
  #ifdef Parallel
  int entry = 0;
  for(int i = 0; i < TWO_POWER_D; i++) {
    for(int j = i; j < TWO_POWER_D; j++) {
      int adjacentRank = fineGridVertices[fineGridVerticesEnumerator(i)].getAdjacentRanks()(j);
      int adjacentCellDescriptionIndex = fineGridVertices[fineGridVerticesEnumerator(i)].getAdjacentCellDescriptionIndexInPeanoOrder(j);
      if(i != TWO_POWER_D-1
          && adjacentRank != tarch::parallel::Node::getInstance().getRank()
          && adjacentCellDescriptionIndex != -1) {
        ParallelSubgrid remoteSubgrid(
          adjacentCellDescriptionIndex
        );

        _cellDescription->setAdjacentRanks(entry, adjacentRank);

        entry++;
      }
    }
  }
  #endif
}
