/*
 * AdjacentSubgrids.h
 *
 *  Created on: Jul 24, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_INTERSUBGRIDCOMMUNICATION_ASPECTS_ADJACENTSUBGRIDS_H_
#define PEANOCLAW_INTERSUBGRIDCOMMUNICATION_ASPECTS_ADJACENTSUBGRIDS_H_

#include "peanoclaw/Vertex.h"
#include "peanoclaw/interSubgridCommunication/GridLevelTransfer.h"
#include "peanoclaw/records/VertexDescription.h"

#include "tarch/la/Vector.h"
#include "tarch/la/VectorCompare.h"
#include "tarch/logging/Log.h"

#include <map>


#define DIMENSIONS_PLUS_ONE (DIMENSIONS+1)

namespace peanoclaw {
  namespace interSubgridCommunication {
    namespace aspects {
      class AdjacentSubgrids;
      class CheckIntersectingParallelAndAdaptiveBoundaryFunctor;
    }
  }
}

/**
 * Encapsulation of the adjacent subgrids/peano cells stored within
 * a vertex. This class provides the functionality to keep this
 * adjacency information consistent.
 */
class peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids {

  public:
    typedef peanoclaw::records::CellDescription CellDescription;
    typedef peanoclaw::records::VertexDescription VertexDescription;
    typedef std::map<tarch::la::Vector<DIMENSIONS_PLUS_ONE,double>, VertexDescription, tarch::la::VectorCompare<DIMENSIONS_PLUS_ONE> > VertexMap;

  private:
    /**
     * Logging device.
     */
    static tarch::logging::Log            _log;
    peanoclaw::Vertex&                    _vertex;
    VertexMap&                            _vertexMap;
    tarch::la::Vector<DIMENSIONS, double> _position;
    int                                   _level;

    tarch::la::Vector<DIMENSIONS_PLUS_ONE, double> createVertexKey() const;

  public:
    AdjacentSubgrids(
      peanoclaw::Vertex& vertex,
      VertexMap& vertexMap,
      tarch::la::Vector<DIMENSIONS, double> position,
      int level
    );

    /**
     * Called when a subgrid adjacent to the corresponding vertex of
     * this AdjacentSubgrids object has been created.
     *
     * @param cellDescriptionIndex The global index referencing the cell
     * description of the newly created subgrid.
     * @param subgridIndex The local index that describes the position of the
     * created subgrid with respect to the current vertex.
     */
    void createdAdjacentSubgrid(int cellDescriptionIndex, int subgridIndex);

    /**
     * Called when a persistent vertex is destroyed, since it may be turned
     * to a hanging vertex.
     */
    void convertPersistentToHangingVertex();

    /**
     * Called when a persistent vertex is created, since it may be constructed
     * from a hanging vertex.
     */
    void convertHangingVertexToPersistentVertex();

    /**
     * Called when creating a hanging vertex.
     */
    void createHangingVertex(
      peanoclaw::Vertex * const                                coarseGridVertices,
      const peano::grid::VertexEnumerator&                     coarseGridVerticesEnumerator,
      const tarch::la::Vector<DIMENSIONS,int>&                 fineGridPositionOfVertex,
      tarch::la::Vector<DIMENSIONS, double>                    domainOffset,
      tarch::la::Vector<DIMENSIONS, double>                    domainSize,
      peanoclaw::interSubgridCommunication::GridLevelTransfer& gridLevelTransfer
    );

    /**
     * Called when destroying a hanging vertex.
     */
    void destroyHangingVertex(
      tarch::la::Vector<DIMENSIONS, double>    domainOffset,
      tarch::la::Vector<DIMENSIONS, double>    domainSize
    );

    /**
     * Triggers the refinement of vertices in such a way that the
     * grid is always at least 2-irregular.
     */
    void regainTwoIrregularity(
      peanoclaw::Vertex * const            coarseGridVertices,
      const peano::grid::VertexEnumerator& coarseGridVerticesEnumerator
    );

    /**
     * TODO unterweg dissertation
     * Triggers a refine when a vertex resides on both, a parallel
     * and an adaptive boundary and two subgrids have to communicate
     * over their vertices (or edges in 3D).
     *
     * Loops through dimensions and check whether the adaptive boundary
     * is completely perpendicular to this dimension. If for one
     * dimension this does not hold, the vertex needs to be refined.
     */
    void refineOnParallelAndAdaptiveBoundary();
    
    /*
     * In an adaptive grid, not all of the $2^d$ adjacent cells exist for hanging
     * vertices. Since each vertex is supposed to hold the adjacent vertices in
     * order to fill the ghostlayers of the patches appropriately, the adjacent
     * indices of hanging vertices need to be filled by the data of the vertices
     * on the next coarser grid. This filling is implemented in this method.
     *
     * !!! The idea
     * Each vertex holds $2^d$ indices. In the vertices they are numbered from 0
     * to $2^d-1$. However, in this method they are considered to exist in a
     * n-dimensional array. In 2d this would look like
     *
     * (0,1)|(1,1)
     * -----v-----
     * (0,0)|(1,0)
     *
     * The linearization looks as follow:
     *
     *   1  |  0
     * -----v-----
     *   3  |  2
     *
     * In the following the term "fine grid" refers to the $4^d$ vertices
     * belonging to the $3^d$ fine grid cells which overlap with the coars grid
     * cell.
     *
     * On the coarse grid cell we again consider the vertices being arranged in a
     * n-dimensional array:
     *
     * (0,1)-----(1,1)
     *   |          |
     *   |          |
     *   |          |
     * (0,0)-----(1,0)
     *
     * Each of them hold again the $2^d$ adjacent indices, while those which refer
     * to a refined cell are set to -1. A hanging vertex therefore gets the
     * adjacent indices from the nearest coarse grid vertex. If they coincide the
     * data can just be used directly. If not, it depends on which boundary of the
     * coarse grid cell the hanging vertex resides. Here the (single) index
     * outside of the coarse grid cell is assigned for all indices of the hanging
     * vertex pointing in the direction of this neighboring coarse grid cell.
     *
     * !!! The algorithm
     * It gets a hanging vertex and performs a loop over the $2^d$ adjacent-patch-
     * indices.
     * In each loop iteration it computes the n-dimensional index of the coarse
     * grid vertex (fromCoarseGridVertex) from which the data has to be copied.
     * For each dimension d with $0\le d <n$:
     *  - If the fine grid position of the hanging vertex in dimension $d$ is 0 set
     *    $fromCoarseGridVertex(d)$ to 0. If it is equals 3 then set
     *    $fromCoarseGridVertex(d)$ to 1. By this we ensure that we always choose
     *    the nearest coarse grid vertex in dimension $d$. If the hanging vertex
     *    resides in a corner of the fine grid this approach always chooses the
     *    coarse grid vertex that is located on the same position.
     *  - If the fine grid position of the hanging vertex in dimension $d$ is
     *    neither 0 nor 3 then the value of $fromCoarseGridVertex(d)$ depends on
     *    the adjacent-patch-index $k$ that has to be set currently. $k(d)$ can
     *    either be 0 or 1. If $k(d)$ is 0 than we want to get data from the
     *    in this dimension "lower" coarse grid vertex, so we set
     *    $fromCoarseGridVertex(d)$ to 0 as well. In the case of $k(d)=1$ we set
     *    $fromCoarseGridVertex(d)$ to 1, accordingly. This actually doesn't
     *    matter since the appropriate adjacent-patch-indices of the to coarse
     *    grid vertices have to be the same, since they are pointing to the same
     *    adjacent cell.
     * The determination of the correct adjacent-patch-index of the coarse grid
     * vertex (coarseGridVertexAdjacentPatchIndex) is done in a similar way. So,
     * for the adjacent-patch-index $k$ on the hanging vertex:
     *  - As stated before, if the fine and coarse grid vertices coincide we can
     *    just copy the adjacent-patch-index. Therefore, if the fine grid position
     *    of the hanging vertex in dimension $d$ is equal to 0 or to 3, we set
     *    $coarseGridVertexAdjacentPatchIndex(d)$ to $k(d)$.
     *  - Otherwise, we just set $coarseGridVertexAdjacentPatchIndex(d)$ to the
     *    inverted $k(d)$. I.e. if $k(d) = 0$ we set
     *    $coarseGridVertexAdjacentPatchIndex(d)$ to 1 and the other way around.
     *
     */
    void fillAdjacentPatchIndicesFromCoarseVertices(
      const peanoclaw::Vertex* coarseGridVertices,
      const peano::grid::VertexEnumerator&      coarseGridVerticesEnumerator,
      const tarch::la::Vector<DIMENSIONS,int>&                   localPositionOfHangingNode
    );
};

/**
 * Used to determine whether a parallel boundary coincides with a corner
 * of an adaptive boundary, such that a refinement is required.
 */
class peanoclaw::interSubgridCommunication::aspects::CheckIntersectingParallelAndAdaptiveBoundaryFunctor {

  private:
    const tarch::la::Vector<DIMENSIONS_TIMES_TWO, int>& _adjacentRanks;
    bool                                                _parallelBoundaryCoincidesWithAdaptiveBoundaryCorner;

  public:
    CheckIntersectingParallelAndAdaptiveBoundaryFunctor(
      const tarch::la::Vector<DIMENSIONS_TIMES_TWO, int>& adjacentRanks
    );

    void operator() (
      peanoclaw::Patch&                         patch1,
      int                                       index1,
      peanoclaw::Patch&                         patch2,
      int                                       index2,
      const tarch::la::Vector<DIMENSIONS, int>& direction
    );

    bool doesParallelBoundaryCoincideWithAdaptiveBoundaryCorner() const;

};

#endif /* PEANOCLAW_INTERSUBGRIDCOMMUNICATION_ADJACENTSUBGRIDS_H_ */
