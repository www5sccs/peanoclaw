/*
 * GridLevelTransfer.h
 *
 *  Created on: Feb 29, 2012
 *      Author: Kristof Unterweger
 */

#ifndef PEANO_APPLICATIONS_PEANOCLAW_GRIDLEVELTRANSFER_H_
#define PEANO_APPLICATIONS_PEANOCLAW_GRIDLEVELTRANSFER_H_

#include "peanoclaw/records/CellDescription.h"
#include "peanoclaw/records/Data.h"

#include <vector>
#include "peano/utils/Globals.h"
#include "tarch/la/Vector.h"
#include "tarch/logging/Log.h"

namespace peanoclaw {
  class Numerics;
  class Patch;
  class Vertex;

  namespace interSubgridCommunication {
    class GridLevelTransfer;
  }

  namespace tests {
    class GridLevelTransferTest;
  }
}
 
namespace peano {
    namespace grid {
      class VertexEnumerator;
    }
}

/**
 * In this class all functionality for transferring data
 * up or down through the spacetree is implemented.
 *
 * Every spacetree node is considered to contain a patch. However,
 * since we're implementing a mostly non-overlapping approach, not
 * all of these patches hold valid grid data. But all of them
 * contain time information. So, each patch is thought of spanning
 * a time interval.
 *
 * @see peanoclaw::Patch for further information about this.
 *
 * !!! Virtual Patches
 * There are three possible situations when a virtual patch needs to be created.
 *
 * !! Refined patch along a refinement boundary
 *
 * @image html virtual-patches-conditions-0.png
 *
 * When a patch is refined but has an unrefined neighbor on the same level, it
 * needs to become a virtual patch in this iteration. In this situation at least
 * one vertex needs to be in the state Unrefined (as long as the grid is static).
 * If the grid is not static we are in the next situation.
 *
 * !! Refined patch neighboring a refining patch
 *
 * @image html virtual-patches-conditions-1.png
 *
 * If an already refined patch has a refining neighbor, it needs to become a virtual
 * patch during this grid iteration. In this state at least on adjacent vertex is
 * in the state Refining.
 *
 * !! Erasing patch
 *
 * @image html virtual-patches-conditions-2.png
 *
 * If fine patches are about to be erased, the containing coarse patch needs to
 * become a virtual patch that later will be the new coarse patch at this part
 * of the grid. In this case at least one vertex is in the state Erasing.
 *
 * !! Condition for creating virtual patches
 *
 * In all three cases at least one vertex is in one of the states Unrefined,
 * Refining, Erasing. Thus, at least one vertex is not in the state Refined
 * (Since the check is done in the enterCell event, the states
 * Refinement_Triggered and Erasing_Triggered already have been switched to
 * Refining and Erasing). This is the used condition for determining wether
 * a refined patch should become a virtual one.
 *
 */
class peanoclaw::interSubgridCommunication::GridLevelTransfer {

  private:
    /**
     * Logging device.
     */
    static tarch::logging::Log _log;

    friend class peanoclaw::tests::GridLevelTransferTest;

    typedef class peanoclaw::records::Data Data;
    typedef class peanoclaw::records::CellDescription CellDescription;

    /**
     * Virtual patches are overlapping patches to the current processed patch. They are held
     * on each level smaller than the current one, so for each coarser cell that overlaps with
     * the current cell.
     *
     * We use these virtual patches for restricting ghostlayer data upwards in the spacetree.
     */
    std::vector<int> _virtualPatchDescriptionIndices;

    /**
     * For each virtual patch in _virtualPatchDescriptionIndices this vector holds the
     * minimum time constraint from the neighbor patches. This can be used to skip restriction
     * steps if the fine patch does not match this time constraint after updating.
     */
    std::vector<double> _virtualPatchTimeConstraints;

    peanoclaw::Numerics& _numerics;

    /**
     * Stores the maximum number of virtual patches which were held by this GridLevelTransfer
     * object. So, this is the maximum size of _virtualPatchDescriptionIndices during the
     * simulation run.
     */
    int _maximumNumberOfSimultaneousVirtualPatches;

    /**
     * Indicates wether the solver uses dimensional splitting. Thus, the timestepping
     * algorithm only has to take care about the 2d direct neighbors.
     */
    bool _useDimensionalSplitting;

    /**
     * Vetos the coarsening on adjacent vertices if necessary and
     * triggers the synchronization of subgrids.
     */
    void vetoCoarseningIfNecessary (
      Patch&                                                      finePatch,
      peanoclaw::Vertex * const fineGridVertices,
      const peano::grid::VertexEnumerator&       fineGridVerticesEnumerator
    );

  public:
    GridLevelTransfer(
      bool useDimensionalSplitting,
      peanoclaw::Numerics& numerics
    );

    virtual ~GridLevelTransfer();

    /**
     *
     */
    void stepDown(
      int                                                         coarseCellDescriptionIndex,
      Patch&                                                      finePatch,
      peanoclaw::Vertex * const fineGridVertices,
      const peano::grid::VertexEnumerator&       fineGridVerticesEnumerator,
      bool                                                        isInitializing
    );

    /**
     * @param isPeanoCellLeaf We need this flag, since during a refinement/coarsening the semantics
     * of Patch::isLeaf() and Cell::isLeaf() may differ. The first describes, wether a patch holds
     * valid grid data and is not virtual, while the latter describes the actual spacetree state of
     * a Peano cell. Hence, for setting the Patch state correctly we here need to know about the
     * state of the containing Peano cell.
     */
    void stepUp(
      int                                                         coarseCellDescriptionIndex,
      Patch&                                                      finePatch,
      bool                                                        isPeanoCellLeaf,
      peanoclaw::Vertex * const fineGridVertices,
      const peano::grid::VertexEnumerator&       fineGridVerticesEnumerator
    );

    /**
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
      peanoclaw::Vertex&       fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,int>&                   localPositionOfHangingNode
    );

    /**
     * Restricts the refinement flags from a hanging vertex to the nearest coarse
     * vertex.
     *
     * The policy here is, that if one hanging vertex has a flag set, the appropriate
     * coarse vertices get the flag set as well. A refine-flag overrides a erase-flag.
     */
    void restrictRefinementFlagsToCoarseVertices(
      peanoclaw::Vertex* coarseGridVertices,
      const peano::grid::VertexEnumerator&      coarseGridVerticesEnumerator,
      const peanoclaw::Vertex&       fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,int>&                   localPositionOfHangingNode
    );

    /**
     * Prepares a refined patch for finding the correct time interval.
     * Basically this method sets current time to -infinity and timestepSize
     * to +infinity.
     */
//    void prepareCoarsePatchForUpdatingTimeinterval(Patch& coarsePatch);

    /**
     * Sets the time interval that is spanned by a coarse patch according on
     * the time intervals spent by the overlapping fine patches. The
     * dependency between them is that the coarse patch always spans the time
     * interval which is covered by all subpatches. I.e. currentTime is the
     * maximum of all the currentTimes of all subpatches and
     * currentTime+timestepSize is the minimum of all currentTime+timestepSize
     * of all subpatches.
     */
//    void updateCoarseTimeintervalFromFineTimeInterval(
//        Patch& coarsePatch,
//        const Patch& finePatch
//    );

    /**
     * Updates the minimal neighbor time for a patch depending on the minimal
     * neighbor time of the coarser patch.
     */
    void updateMinimalNeighborTimeFromCoarserPatch(
        const Patch& coarsePatch,
        Patch& finePatch
    );
};

#endif /* PEANO_APPLICATIONS_PEANOCLAW_GRIDLEVELTRANSFER_H_ */
