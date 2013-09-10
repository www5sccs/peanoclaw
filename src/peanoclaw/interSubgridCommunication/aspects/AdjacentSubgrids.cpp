/*
 * AdjacentSubgrids.cpp
 *
 *  Created on: Jul 24, 2013
 *      Author: kristof
 */
#include "peanoclaw/interSubgridCommunication/aspects/AdjacentSubgrids.h"

#include "peanoclaw/Patch.h"
#include "peanoclaw/ParallelSubgrid.h"

#include "peano/heap/Heap.h"

tarch::logging::Log peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids::_log("peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids");

tarch::la::Vector<DIMENSIONS_PLUS_ONE, double> peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids::createVertexKey() const {
  tarch::la::Vector<DIMENSIONS_PLUS_ONE, double> vertexKey;
  for(int d = 0; d < DIMENSIONS; d++) {
    vertexKey(d) = _position(d);
  }
  vertexKey(DIMENSIONS) = _level;
  return vertexKey;
}

peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids::AdjacentSubgrids(
  peanoclaw::Vertex&                    vertex,
  VertexMap&                            vertexMap,
  tarch::la::Vector<DIMENSIONS, double> position,
  int                                   level
) : _vertex(vertex), _vertexMap(vertexMap), _position(position), _level(level)
{}

void peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids::createdAdjacentSubgrid(
  int cellDescriptionIndex,
  int subgridIndex
) {
  _vertex.setAdjacentCellDescriptionIndex(subgridIndex, cellDescriptionIndex);
}

void peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids::convertPersistentToHangingVertex() {
  //  Retrieve or create hanging vertex description
  tarch::la::Vector<DIMENSIONS_PLUS_ONE, double> hangingVertexPosition = createVertexKey();

  if( _vertexMap.find(hangingVertexPosition) == _vertexMap.end() ) {
    VertexDescription vertexDescription;
    _vertexMap[hangingVertexPosition] = vertexDescription;
  }

  VertexDescription& vertexDescription = _vertexMap[hangingVertexPosition];
  vertexDescription.setTouched(true);

  //Copy adjacency information from destroyed vertex to hanging vertex description
  for(int i = 0; i < TWO_POWER_D; i++) {
    vertexDescription.setIndicesOfAdjacentCellDescriptions(i, _vertex.getAdjacentCellDescriptionIndex(i));
  }
}

void peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids::convertHangingVertexToPersistentVertex() {
  //Copy adjacent cell indices from former hanging vertex description, if available.
  tarch::la::Vector<DIMENSIONS_PLUS_ONE, double> vertexPosition = createVertexKey();

  if( _vertexMap.find(vertexPosition) != _vertexMap.end() ) {
    VertexDescription& hangingVertexDescription = _vertexMap[vertexPosition];

    for(int i = 0; i < TWO_POWER_D; i++) {
      //Skip setting the index if a valid index is already set. It seems that sometimes createCell(...) is
      //triggered before createInnerVertex(...)
      if(_vertex.getAdjacentCellDescriptionIndex(i) == -1) {
        int hangingVertexIndex = hangingVertexDescription.getIndicesOfAdjacentCellDescriptions(i);

        if(hangingVertexIndex != -1) {
          Patch patch(peano::heap::Heap<CellDescription>::getInstance().getData(hangingVertexIndex).at(0));
          if(patch.getLevel() == _level) {
            _vertex.setAdjacentCellDescriptionIndex(i, hangingVertexIndex);
          }
        }
      }
    }
  }

  _vertex.setWasCreatedInThisIteration(true);
}

void peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids::createHangingVertex(
  peanoclaw::Vertex * const                coarseGridVertices,
  const peano::grid::VertexEnumerator&     coarseGridVerticesEnumerator,
  const tarch::la::Vector<DIMENSIONS,int>& fineGridPositionOfVertex,
  tarch::la::Vector<DIMENSIONS, double> domainOffset,
  tarch::la::Vector<DIMENSIONS, double> domainSize,
  peanoclaw::interSubgridCommunication::GridLevelTransfer& gridLevelTransfer
) {
  if(!tarch::la::oneGreater(domainOffset, _position)
      && !tarch::la::oneGreater(_position, domainOffset + domainSize)) {
    //Project adjacency information down from coarse grid vertex
    gridLevelTransfer.fillAdjacentPatchIndicesFromCoarseVertices(
      coarseGridVertices,
      coarseGridVerticesEnumerator,
      _vertex,
      fineGridPositionOfVertex
    );

    //Retrieve or create hanging vertex description
    tarch::la::Vector<DIMENSIONS_PLUS_ONE, double> hangingVertexPosition;
    for(int d = 0; d < DIMENSIONS; d++) {
      hangingVertexPosition(d) = _position(d);
    }
    hangingVertexPosition(DIMENSIONS) = _level;

    if( _vertexMap.find(hangingVertexPosition) == _vertexMap.end() ) {
      VertexDescription vertexDescription;
      vertexDescription.setTouched(true);
      for(int i = 0; i < TWO_POWER_D; i++) {
        vertexDescription.setIndicesOfAdjacentCellDescriptions(i, -1);
      }
      _vertexMap[hangingVertexPosition] = vertexDescription;
    } else {
      //A vertex on this position existed earlier...
      _vertex.setWasCreatedInThisIteration(false);
    }

    VertexDescription& hangingVertexDescription = _vertexMap[hangingVertexPosition];
    hangingVertexDescription.setTouched(true);

    //Copy indices from coarse level
    gridLevelTransfer.fillAdjacentPatchIndicesFromCoarseVertices(
      coarseGridVertices,
      coarseGridVerticesEnumerator,
      _vertex,
      fineGridPositionOfVertex
    );

    //Remove deleted indices
    for(int i = 0; i < TWO_POWER_D; i++) {
      //From hanging vertex description
      if(hangingVertexDescription.getIndicesOfAdjacentCellDescriptions(i) != -1
          && !peano::heap::Heap<CellDescription>::getInstance().isValidIndex(hangingVertexDescription.getIndicesOfAdjacentCellDescriptions(i))) {
        hangingVertexDescription.setIndicesOfAdjacentCellDescriptions(i, -1);
      }
      if(_vertex.getAdjacentCellDescriptionIndex(i) != -1
          && !peano::heap::Heap<CellDescription>::getInstance().isValidIndex(_vertex.getAdjacentCellDescriptionIndex(i))) {
        _vertex.setAdjacentCellDescriptionIndex(i, -1);
      }
    }

    //TODO If the coarse grid vertices are also hanging, a deleted patch index two or more
    //levels coarser than this hanging vertex might not be recognized, yet.
    //Merging adjacency information from stored hanging vertex description and hanging vertex
    //The data stored on the hanging vertex itself must come from the coarser vertex, since
    //the hanging vertex has just been created. So, this data is more recent, when the data
    //in the hanging vertex description describes a patch on a coarser level than the hanging
    //vertex (Should be solved by the check before).
    for(int i = 0; i < TWO_POWER_D; i++) {
      assertion(hangingVertexDescription.getIndicesOfAdjacentCellDescriptions(i) >= -1);
      if(hangingVertexDescription.getIndicesOfAdjacentCellDescriptions(i) != -1) {
        CellDescription& cellDescription = peano::heap::Heap<CellDescription>::getInstance().getData(hangingVertexDescription.getIndicesOfAdjacentCellDescriptions(i)).at(0);
        if(cellDescription.getLevel() == _level) {
          _vertex.setAdjacentCellDescriptionIndex(i, hangingVertexDescription.getIndicesOfAdjacentCellDescriptions(i));
        }
      }
    }
  }
}

void peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids::destroyHangingVertex(
  tarch::la::Vector<DIMENSIONS, double>    domainOffset,
  tarch::la::Vector<DIMENSIONS, double>    domainSize
) {
  if(!tarch::la::oneGreater(domainOffset, _position) && !tarch::la::oneGreater(_position, domainOffset + domainSize)) {

    //Retrieve hanging vertex description
    tarch::la::Vector<DIMENSIONS_PLUS_ONE, double> hangingVertexPosition;
    for(int d = 0; d < DIMENSIONS; d++) {
      hangingVertexPosition(d) = _position(d);
    }
    hangingVertexPosition(DIMENSIONS) = _level;

    assertionMsg(_vertexMap.find(hangingVertexPosition) != _vertexMap.end(), "Hanging vertex description was not created for vertex " << _vertex);

    VertexDescription& hangingVertexDescription = _vertexMap[hangingVertexPosition];
    hangingVertexDescription.setTouched(true);

    //Copy adjacency information from hanging vertex to hanging vertex description
    for(int i = 0; i < TWO_POWER_D; i++) {
      hangingVertexDescription.setIndicesOfAdjacentCellDescriptions(
        i,
        _vertex.getAdjacentCellDescriptionIndex(i)
      );
    }
//    hangingVertexDescription.setLastUpdateIterationParity(_iterationParity);
  }
}

void peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids::storeAdjacencyInformation() {
  #ifdef Parallel
  //Set all adjacent patches to unsent if the adjacency information has changed
  for(int i = 0; i < TWO_POWER_D; i++) {
    if(_vertex.getAdjacentRanks()(i) != _vertex.getAdjacentRanksDuringLastIteration()(i)) {
      for(int j = 0; j < TWO_POWER_D; j++) {
        if(_vertex.getAdjacentCellDescriptionIndex(j) != -1) {
          ParallelSubgrid adjacentSubgrid(_vertex.getAdjacentCellDescriptionIndex(j));
          adjacentSubgrid.markCurrentStateAsSent(false);
        }
      }
    }
  }
  #endif
}

void peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids::regainTwoIrregularity(
  peanoclaw::Vertex * const            coarseGridVertices,
  const peano::grid::VertexEnumerator& coarseGridVerticesEnumerator
) {
  //Regain 2-irregularity if necessary
  if(_vertex.getRefinementControl() == peanoclaw::Vertex::Records::Refined
      || _vertex.getRefinementControl() == peanoclaw::Vertex::Records::Refining) {
    tarch::la::Vector<DIMENSIONS, int> coarseGridPositionOfVertex(0);
    for(int d = 0; d < DIMENSIONS; d++) {
      if(_position(d) > 1) {
        coarseGridPositionOfVertex(d) = 1;
      }
    }

    peanoclaw::Vertex& coarseVertex = coarseGridVertices[coarseGridVerticesEnumerator(coarseGridPositionOfVertex)];
    if(coarseVertex.getRefinementControl() == peanoclaw::Vertex::Records::Unrefined
        && !coarseVertex.isHangingNode()) {
      coarseVertex.refine();
    }
  }

  //Mark vertex as "old" (i.e. older than just created ;-))
  _vertex.setWasCreatedInThisIteration(false);
}
