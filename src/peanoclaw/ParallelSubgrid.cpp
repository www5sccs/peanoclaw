/*
 * ParallelSubgrid.cpp
 *
 *  Created on: Aug 5, 2013
 *      Author: kristof
 */
#include "peanoclaw/ParallelSubgrid.h"

#include "peano/heap/Heap.h"
#include "peano/utils/Globals.h"

tarch::logging::Log peanoclaw::ParallelSubgrid::_log("peanoclaw::ParallelSubgrid");

peanoclaw::ParallelSubgrid::ParallelSubgrid(
  CellDescription& cellDescription
) : _cellDescription(&cellDescription) {
}

peanoclaw::ParallelSubgrid::ParallelSubgrid(
  int subgridDescriptionIndex
) {
  _cellDescription = &peano::heap::Heap<CellDescription>::getInstance().getData(subgridDescriptionIndex).at(0);
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

void peanoclaw::ParallelSubgrid::decreaseNumberOfSharedAdjacentVertices() {
  #ifdef Parallel
  _cellDescription->setNumberOfSharedAdjacentVertices(_cellDescription->getNumberOfSharedAdjacentVertices() - 1);
  #endif
}

int peanoclaw::ParallelSubgrid::getAdjacentRank() const {
  #ifdef Parallel
  return _cellDescription->getAdjacentRank();
  #else
  return -1;
  #endif
}

int peanoclaw::ParallelSubgrid::getNumberOfSharedAdjacentVertices() const {
  #ifdef Parallel
  return _cellDescription->getNumberOfSharedAdjacentVertices();
  #else
  return 0;
  #endif
}

void peanoclaw::ParallelSubgrid::countNumberOfAdjacentParallelSubgridsAndResetExclusiveFlag (
  peanoclaw::Vertex * const            vertices,
  const peano::grid::VertexEnumerator& verticesEnumerator
) {
  #ifdef Parallel
  int adjacentRank = -1;
  int localRank = tarch::parallel::Node::getInstance().getRank();
  int numberOfSharedAdjacentVertices = 0;
  for(int i = 0; i < TWO_POWER_D; i++) {
    Vertex& vertex = vertices[verticesEnumerator(i)];

    for(int j = 0; j < TWO_POWER_D; j++) {
      if(vertex.getAdjacentRanks()(j) != localRank) {
        if(adjacentRank == -1 || vertex.getAdjacentRanks()(j) == adjacentRank) {
          adjacentRank = vertex.getAdjacentRanks()(j);
          numberOfSharedAdjacentVertices++;
        } else {
          adjacentRank = -1;
          numberOfSharedAdjacentVertices = -1;
        }
        break;
      }
    }

    if(numberOfSharedAdjacentVertices == -1) {
      //There are more than one adjacent ranks
      break;
    }
  }
  _cellDescription->setAdjacentRank(adjacentRank);
  _cellDescription->setNumberOfSharedAdjacentVertices(numberOfSharedAdjacentVertices);
  _cellDescription->setIsExclusiveMessageForSubgrid(false);
  #endif
}
