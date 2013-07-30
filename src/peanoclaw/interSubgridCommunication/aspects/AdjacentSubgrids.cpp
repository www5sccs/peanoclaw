/*
 * AdjacentSubgrids.cpp
 *
 *  Created on: Jul 24, 2013
 *      Author: kristof
 */
#include "peanoclaw/interSubgridCommunication/aspects/AdjacentSubgrids.h"

#include "peanoclaw/Patch.h"

#include "peano/heap/Heap.h"

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
  tarch::la::Vector<DIMENSIONS_PLUS_ONE, double> hangingVertexPosition;
  for(int d = 0; d < DIMENSIONS; d++) {
    hangingVertexPosition(d) = _position(d);
  }
  hangingVertexPosition(DIMENSIONS) = _level;

  if( _vertexMap.find(hangingVertexPosition) == _vertexMap.end() ) {
    VertexDescription vertexDescription;
//    if(_iterationParity == peanoclaw::records::VertexDescription::EVEN) {
//      vertexDescription.setLastUpdateIterationParity(peanoclaw::records::VertexDescription::ODD);
//    } else {
//      vertexDescription.setLastUpdateIterationParity(peanoclaw::records::VertexDescription::EVEN);
//    }
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
  for(int i = 0; i < TWO_POWER_D; i++) {
    _vertex.setAdjacentCellDescriptionIndex(i, -1);
  }

  //Copy adjacent cell indices from former hanging vertex description, if available.
  tarch::la::Vector<DIMENSIONS_PLUS_ONE, double> vertexPosition;
  for(int d = 0; d < DIMENSIONS; d++) {
    vertexPosition(d) = _position(d);
  }
  vertexPosition(DIMENSIONS) = _level;

  if( _vertexMap.find(vertexPosition) != _vertexMap.end() ) {
    VertexDescription& hangingVertexDescription = _vertexMap[vertexPosition];

    for(int i = 0; i < TWO_POWER_D; i++) {
      int hangingVertexIndex = hangingVertexDescription.getIndicesOfAdjacentCellDescriptions(i);
      int persistentVertexIndex = -1;
      if(hangingVertexIndex != -1) {
        Patch patch(peano::heap::Heap<CellDescription>::getInstance().getData(hangingVertexIndex).at(0));
        if(patch.getLevel() == _level) {
          persistentVertexIndex = hangingVertexIndex;
        }
      }
      _vertex.setAdjacentCellDescriptionIndex(i, persistentVertexIndex);
    }
  } else {
    for(int i = 0; i < TWO_POWER_D; i++) {
      _vertex.setAdjacentCellDescriptionIndex(i, -1);
    }
  }
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
//      if(_iterationParity == peanoclaw::records::VertexDescription::EVEN) {
//        vertexDescription.setLastUpdateIterationParity(peanoclaw::records::VertexDescription::ODD);
//      } else {
//        vertexDescription.setLastUpdateIterationParity(peanoclaw::records::VertexDescription::EVEN);
//      }
      _vertexMap[hangingVertexPosition] = vertexDescription;
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
