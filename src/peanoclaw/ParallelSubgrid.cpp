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
  _cellDescription->setCurrentStateWasSend(wasSent);
}

bool peanoclaw::ParallelSubgrid::wasCurrentStateSent() const {
  return _cellDescription->getCurrentStateWasSend();
}

void peanoclaw::ParallelSubgrid::decreaseNumberOfSharedAdjacentVertices() {
  _cellDescription->setNumberOfSharedAdjacentVertices(_cellDescription->getNumberOfSharedAdjacentVertices() - 1);
}

int peanoclaw::ParallelSubgrid::getAdjacentRank() const {
  return _cellDescription->getAdjacentRank();
}

int peanoclaw::ParallelSubgrid::getNumberOfSharedAdjacentVertices() const {
  return _cellDescription->getNumberOfSharedAdjacentVertices();
}

void peanoclaw::ParallelSubgrid::countNumberOfAdjacentParallelSubgrids (
  peanoclaw::Vertex * const            vertices,
  const peano::grid::VertexEnumerator& verticesEnumerator
) {
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
}
