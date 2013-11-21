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

//void peanoclaw::ParallelSubgrid::markCurrentStateAsSent(bool wasSent) {
//  #ifdef Parallel
//  _cellDescription->setCurrentStateWasSend(wasSent);
//  #endif
//}
//
//bool peanoclaw::ParallelSubgrid::wasCurrentStateSent() const {
//  #ifdef Parallel
//  return _cellDescription->getCurrentStateWasSend();
//  #else
//  return false;
//  #endif
//}

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

void peanoclaw::ParallelSubgrid::countNumberOfAdjacentParallelSubgrids (
  peanoclaw::Vertex * const            vertices,
  const peano::grid::VertexEnumerator& verticesEnumerator
) {
  #ifdef Parallel
//  assertion1(_cellDescription->getNumberOfSharedAdjacentVertices() <= 0, _cellDescription->toString());

  int adjacentRank = -1;
  int localRank = tarch::parallel::Node::getInstance().getRank();
  int numberOfSharedAdjacentVertices = 0;
  for(int i = 0; i < TWO_POWER_D; i++) {
    Vertex& vertex = vertices[verticesEnumerator(i)];

    if(!vertex.isHangingNode()) {
      bool oneAdjacentRemoteRank = false;
      bool moreThanOneAdjacentRemoteRanks = false;

      for(int j = 0; j < TWO_POWER_D; j++) {
        if(
            vertex.getAdjacentRanks()(j) != localRank
            && vertex.getAdjacentRanks()(j) != 0
          ) {
          if(adjacentRank == -1) {
            adjacentRank = vertex.getAdjacentRanks()(j);
            oneAdjacentRemoteRank = true;
          } else if(vertex.getAdjacentRanks()(j) == adjacentRank) {
            oneAdjacentRemoteRank = true;
          } else {
            moreThanOneAdjacentRemoteRanks = true;
            break;
          }
        }
      }

      if(oneAdjacentRemoteRank && !moreThanOneAdjacentRemoteRanks) {
        numberOfSharedAdjacentVertices++;
      } else if (moreThanOneAdjacentRemoteRanks) {
        adjacentRank = -1;
        numberOfSharedAdjacentVertices = -1;
      }
    }

    if(numberOfSharedAdjacentVertices == -1) {
      //There are more than one adjacent ranks
      break;
    }
  }

  assertion2(numberOfSharedAdjacentVertices <= TWO_POWER_D, numberOfSharedAdjacentVertices, _cellDescription->toString());

  _cellDescription->setAdjacentRank(adjacentRank);
  _cellDescription->setNumberOfSharedAdjacentVertices(numberOfSharedAdjacentVertices);
  _cellDescription->setNumberOfSkippedTransfers(std::max(0, numberOfSharedAdjacentVertices-1));
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

//bool isAdjacentToRemoteSubdomain(
//  peanoclaw::Vertex * const            fineGridVertices,
//  const peano::grid::VertexEnumerator& fineGridVerticesEnumerator
//) {
//  #ifdef Parallel
//  bool isAdjacentToRemoteRank = false;
//  for(int i = 0; i < TWO_POWER_D; i++) {
//    isAdjacentToRemoteRank |= fineGridVertices[fineGridVerticesEnumerator(i)].isAdjacentToRemoteRank();
//  }
//  return isAdjacentToRemoteRank;
//  #else
//  return false;
//  #endif
//}
