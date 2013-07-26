/*
 * AdjacentSubgrids.cpp
 *
 *  Created on: Jul 24, 2013
 *      Author: kristof
 */
#include "peanoclaw/interSubgridCommunication/AdjacentSubgrids.h"

peanoclaw::interSubgridCommunication::AdjacentSubgrids::AdjacentSubgrids(
  peanoclaw::Vertex&                    vertex,
  VertexMap&                            vertexMap,
  tarch::la::Vector<DIMENSIONS, double> position,
  int                                   level
) : _vertex(vertex), _vertexMap(vertexMap), _position(position), _level(level)
{}

void peanoclaw::interSubgridCommunication::AdjacentSubgrids::createdAdjacentSubgrid(
  int cellDescriptionIndex,
  int subgridIndex
) {
  _vertex.setAdjacentCellDescriptionIndex(subgridIndex, cellDescriptionIndex);
}

void peanoclaw::interSubgridCommunication::AdjacentSubgrids::convertPersistentToHangingVertex() {
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
