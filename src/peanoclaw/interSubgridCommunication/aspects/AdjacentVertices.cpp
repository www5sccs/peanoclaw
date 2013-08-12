/*
 * AdjacentVertices.cpp
 *
 *  Created on: Aug 8, 2013
 *      Author: kristof
 */
#include "peanoclaw/interSubgridCommunication/aspects/AdjacentVertices.h"

peanoclaw::interSubgridCommunication::aspects::AdjacentVertices::AdjacentVertices(
  peanoclaw::Vertex*             vertices,
  peano::grid::VertexEnumerator& verticesEnumerator
) : _vertices(vertices), _verticesEnumerator(verticesEnumerator) {
}
