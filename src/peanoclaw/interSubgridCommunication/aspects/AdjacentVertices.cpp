/*
 * AdjacentVertices.cpp
 *
 *  Created on: Aug 8, 2013
 *      Author: kristof
 */
#include "peanoclaw/interSubgridCommunication/aspects/AdjacentVertices.h"

#include "peanoclaw/Patch.h"

tarch::logging::Log peanoclaw::interSubgridCommunication::aspects::AdjacentVertices::_log("peanoclaw::interSubgridCommunication::aspects::AdjacentVertices");

peanoclaw::interSubgridCommunication::aspects::AdjacentVertices::AdjacentVertices(
  peanoclaw::Vertex* const             vertices,
  const peano::grid::VertexEnumerator& verticesEnumerator
) : _vertices(vertices), _verticesEnumerator(verticesEnumerator) {
}

bool peanoclaw::interSubgridCommunication::aspects::AdjacentVertices::refineIfNecessary(
  Patch&                                patch,
  tarch::la::Vector<DIMENSIONS, double> minimalMeshWidth
) {
  //Check for error in refinement criterion
  if(!tarch::la::allGreater(minimalMeshWidth, tarch::la::Vector<DIMENSIONS,double>(0.0))) {
    logWarning("createCell(...)", "A demanded mesh width of 0.0 leads to an infinite refinement. Is the refinement criterion correct?");
  }
  assertion(tarch::la::allGreater(minimalMeshWidth, 0.0));

  //Refine if necessary
  if(tarch::la::oneGreater(patch.getSubcellSize(), tarch::la::Vector<DIMENSIONS, double>(minimalMeshWidth))) {
    for(int i = 0; i < TWO_POWER_D; i++) {
      if (_vertices[_verticesEnumerator(i)].getRefinementControl() == Vertex::Records::Unrefined
          && !_vertices[_verticesEnumerator(i)].isHangingNode()) {
        _vertices[_verticesEnumerator(i)].refine();
      }
    }
  }

  //Switch to refined patch if necessary
  bool refinementTriggered = false;
  for(int i = 0; i < TWO_POWER_D; i++) {
    if(_vertices[_verticesEnumerator(i)].getRefinementControl()
        == Vertex::Records::Refining) {
      refinementTriggered = true;
    }
  }
  if(refinementTriggered) {
    assertion1(patch.isLeaf(), patch.toString());
    patch.switchToVirtual();
    patch.switchToNonVirtual();
    assertion1(!patch.isLeaf() && !patch.isVirtual(), patch);
  }

  return refinementTriggered;
}
