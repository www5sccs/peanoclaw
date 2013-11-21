/*
 * AdjacentSubgrids.cpp
 *
 *  Created on: Jul 24, 2013
 *      Author: kristof
 */
#include "peanoclaw/interSubgridCommunication/aspects/AdjacentSubgrids.h"

#include "peanoclaw/interSubgridCommunication/aspects/CornerAdjacentPatchTraversal.h"
#include "peanoclaw/interSubgridCommunication/aspects/EdgeAdjacentPatchTraversal.h"

#include "peanoclaw/Heap.h"
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
          Patch patch(CellDescriptionHeap::getInstance().getData(hangingVertexIndex).at(0));
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
    fillAdjacentPatchIndicesFromCoarseVertices(
      coarseGridVertices,
      coarseGridVerticesEnumerator,
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
    }

    VertexDescription& hangingVertexDescription = _vertexMap[hangingVertexPosition];
    hangingVertexDescription.setTouched(true);

    //Copy indices from coarse level
    fillAdjacentPatchIndicesFromCoarseVertices(
      coarseGridVertices,
      coarseGridVerticesEnumerator,
      fineGridPositionOfVertex
    );

    //Remove deleted indices
    for(int i = 0; i < TWO_POWER_D; i++) {
      //From hanging vertex description
      if(hangingVertexDescription.getIndicesOfAdjacentCellDescriptions(i) != -1
          && !CellDescriptionHeap::getInstance().isValidIndex(hangingVertexDescription.getIndicesOfAdjacentCellDescriptions(i))) {
        hangingVertexDescription.setIndicesOfAdjacentCellDescriptions(i, -1);
      }
      if(_vertex.getAdjacentCellDescriptionIndex(i) != -1
          && !CellDescriptionHeap::getInstance().isValidIndex(_vertex.getAdjacentCellDescriptionIndex(i))) {
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
        CellDescription& cellDescription = CellDescriptionHeap::getInstance().getData(hangingVertexDescription.getIndicesOfAdjacentCellDescriptions(i)).at(0);
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

void peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids::regainTwoIrregularity(
  peanoclaw::Vertex * const                coarseGridVertices,
  const peano::grid::VertexEnumerator&     coarseGridVerticesEnumerator,
  const tarch::la::Vector<DIMENSIONS,int>& fineGridPositionOfVertex
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
        && !coarseVertex.isHangingNode()
        && !coarseVertex.isOutside()) {

      coarseVertex.refine();
    }
  }
}

void peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids::fillAdjacentPatchIndicesFromCoarseVertices(
  const peanoclaw::Vertex*                 coarseGridVertices,
  const peano::grid::VertexEnumerator&     coarseGridVerticesEnumerator,
  const tarch::la::Vector<DIMENSIONS,int>& localPositionOfHangingNode
) {
  logTraceInWith1Argument( "fillAdjacentPatchIndicesFromCoarseVertices(...)", localPositionOfHangingNode );

  tarch::la::Vector<DIMENSIONS,int>   fromCoarseGridVertex;
  tarch::la::Vector<DIMENSIONS,int>   coarseGridVertexAdjacentPatchIndex;

  dfor2(k)
    for (int d=0; d<DIMENSIONS; d++) {
      if (localPositionOfHangingNode(d)==0) {
        fromCoarseGridVertex(d)          = 0;
        coarseGridVertexAdjacentPatchIndex(d) = k(d);
      }
      else if (localPositionOfHangingNode(d)==3) {
        fromCoarseGridVertex(d)          = 1;
        coarseGridVertexAdjacentPatchIndex(d) = k(d);
      }
      else if (k(d)==0) {
        fromCoarseGridVertex(d)          = 0;
        coarseGridVertexAdjacentPatchIndex(d) = 1;
      }
      else {
        fromCoarseGridVertex(d)          = 1;
        coarseGridVertexAdjacentPatchIndex(d) = 0;
      }
    }
    int coarseGridVertexIndex = coarseGridVerticesEnumerator(peano::utils::dLinearised(fromCoarseGridVertex,2));
    int coarseGridVertexEntry = TWO_POWER_D_MINUS_ONE-peano::utils::dLinearised(coarseGridVertexAdjacentPatchIndex,2);
    _vertex.setAdjacentCellDescriptionIndex(
      TWO_POWER_D_MINUS_ONE-kScalar,
      coarseGridVertices[coarseGridVertexIndex].getAdjacentCellDescriptionIndex(coarseGridVertexEntry)
    );
  enddforx

  logTraceOut( "fillAdjacentPatchIndicesFromCoarseVertices(...)" );
}

void peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids::refineOnParallelAndAdaptiveBoundary() {
  logTraceIn("refineOnParallelBoundary(...)");
  #ifdef Parallel
  assertion2(!_vertex.isHangingNode(), _vertex, _position);

  if(_vertex.getRefinementControl() == Vertex::Records::Unrefined && _vertex.isAdjacentToRemoteRank()) {
    //Fill ghost layers of adjacent cells
    //Get adjacent cell descriptions
    CellDescription* cellDescriptions[TWO_POWER_D];
    for(int cellIndex = 0; cellIndex < TWO_POWER_D; cellIndex++) {
      if(_vertex.getAdjacentCellDescriptionIndex(cellIndex) != -1) {
        cellDescriptions[cellIndex] = &CellDescriptionHeap::getInstance().getData(_vertex.getAdjacentCellDescriptionIndex(cellIndex)).at(0);
      }
    }

    Patch patches[TWO_POWER_D];
    dfor2(cellIndex)
      if(_vertex.getAdjacentCellDescriptionIndex(cellIndexScalar) != -1) {
        patches[cellIndexScalar] = Patch(
          *cellDescriptions[cellIndexScalar]
        );
      }
    enddforx

    //Until now we just created the patches. Refactor this?
    CheckIntersectingParallelAndAdaptiveBoundaryFunctor functor(_vertex.getAdjacentRanks());
    peanoclaw::interSubgridCommunication::aspects::
      EdgeAdjacentPatchTraversal<CheckIntersectingParallelAndAdaptiveBoundaryFunctor>(patches, functor);
    #ifdef Dim3
    peanoclaw::interSubgridCommunication::aspects::
      CornerAdjacentPatchTraversal<CheckIntersectingParallelAndAdaptiveBoundaryFunctor>(patches, functor);
    #endif

    if(functor.doesParallelBoundaryCoincideWithAdaptiveBoundaryCorner()) {
      _vertex.refine();
    }
  }

  #endif
  logTraceOut("refineOnParallelBoundary(...)");
}

void peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids::checkForChangesInAdjacentRanks() {
  _vertex.setWhetherAdjacentRanksChanged(false);
  for(int i = 0; i < TWO_POWER_D; i++) {
    if(_vertex.getAdjacentRanks()(i) != _vertex.getAdjacentRanksInFormerGridIteration()(i)) {
      _vertex.setWhetherAdjacentRanksChanged(true);
      break;
    }
  }
  _vertex.setAdjacentRanksInFormerGridIteration(_vertex.getAdjacentRanks());
}

peanoclaw::interSubgridCommunication::aspects::CheckIntersectingParallelAndAdaptiveBoundaryFunctor::CheckIntersectingParallelAndAdaptiveBoundaryFunctor(
  const tarch::la::Vector<TWO_POWER_D, int>& adjacentRanks
) : _adjacentRanks(adjacentRanks),
    _numberOfDiagonallyAdjacentSubgrids(0),
    _numberOfDiagonallyAdjacentRefinedSubgrids(0)
{
}

void peanoclaw::interSubgridCommunication::aspects::CheckIntersectingParallelAndAdaptiveBoundaryFunctor::operator() (
  peanoclaw::Patch&                         patch1,
  int                                       index1,
  peanoclaw::Patch&                         patch2,
  int                                       index2,
  const tarch::la::Vector<DIMENSIONS, int>& direction
) {
  _numberOfDiagonallyAdjacentSubgrids++;

  if(_adjacentRanks[TWO_POWER_D - index1 - 1] != _adjacentRanks[TWO_POWER_D - index2 - 1]
    && patch1.isValid() && !patch1.isLeaf()
    && patch2.isValid() && !patch2.isLeaf()) {
    _numberOfDiagonallyAdjacentRefinedSubgrids++;
  }
}

bool peanoclaw::interSubgridCommunication::aspects::CheckIntersectingParallelAndAdaptiveBoundaryFunctor::doesParallelBoundaryCoincideWithAdaptiveBoundaryCorner() const {
  return _numberOfDiagonallyAdjacentRefinedSubgrids != 0
      && _numberOfDiagonallyAdjacentRefinedSubgrids != _numberOfDiagonallyAdjacentSubgrids;
}
