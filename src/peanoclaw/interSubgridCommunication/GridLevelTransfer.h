/*
w * GridLevelTransfer.h
 *
 *  Created on: Feb 29, 2012
 *      Author: Kristof Unterweger
 */

#ifndef PEANO_APPLICATIONS_PEANOCLAW_GRIDLEVELTRANSFER_H_
#define PEANO_APPLICATIONS_PEANOCLAW_GRIDLEVELTRANSFER_H_

#include "peanoclaw/ParallelSubgrid.h"
#include "peanoclaw/Patch.h"
#include "peanoclaw/records/CellDescription.h"
#include "peanoclaw/records/Data.h"

#include <map>
#include <vector>
#include "peano/utils/Globals.h"
#include "tarch/la/Vector.h"
#include "tarch/la/VectorCompare.h"
#include "tarch/logging/Log.h"
#include "tarch/multicore/BooleanSemaphore.h"

#define DIMENSIONS_PLUS_ONE (DIMENSIONS+1)

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

    tarch::multicore::BooleanSemaphore _virtualPatchListSemaphore;

    friend class peanoclaw::tests::GridLevelTransferTest;

    typedef class peanoclaw::records::Data Data;
    typedef class peanoclaw::records::CellDescription CellDescription;
    typedef std::map<tarch::la::Vector<DIMENSIONS_PLUS_ONE,double>, int, tarch::la::VectorCompare<DIMENSIONS_PLUS_ONE> > VirtualSubgridMap;

    /**
     * Virtual patches are overlapping patches to the current processed patch. They are held
     * on each level smaller than the current one, so for each coarser cell that overlaps with
     * the current cell.
     *
     * We use these virtual patches for restricting ghostlayer data upwards in the spacetree.
     */
    //std::vector<int> _virtualPatchDescriptionIndices;
    VirtualSubgridMap _virtualPatchDescriptionIndices;

    /**
     * For each virtual patch in _virtualPatchDescriptionIndices this vector holds the
     * minimum time constraint from the neighbor patches. This can be used to skip restriction
     * steps if the fine patch does not match this time constraint after updating.
     */
//    std::vector<double> _virtualPatchTimeConstraints;

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
     *
     * TODO unterweg dissertation
     */
    void vetoCoarseningIfNecessary (
      Patch&                               patch,
      peanoclaw::Vertex * const            fineGridVertices,
      const peano::grid::VertexEnumerator& fineGridVerticesEnumerator
    );

    /**
     * Retrieves whether a patch surrounded by the given vertices
     * is adjacent to a subdomain residing on a different MPI
     * rank.
     */
    bool isPatchAdjacentToRemoteRank (
      peanoclaw::Vertex * const            fineGridVertices,
      const peano::grid::VertexEnumerator& fineGridVerticesEnumerator
    );

    /**
     * Determines, whether the given subgrid should be turned into a
     * virtual subgrid.
     *
     * TODO unterweg debug
     */
    bool shouldBecomeVirtualSubgrid(
      const Patch&                         fineSubgrid,
      peanoclaw::Vertex * const            fineGridVertices,
      const peano::grid::VertexEnumerator& fineGridVerticesEnumerator,
      bool                                 isInitializing,
      bool                                 isPeanoCellLeaf
    );

    /**
     * Turns the given subgrid into a virtual subgrid and adds it to
     * the list of virtual subgrids for restriction.
     */
    void switchToAndAddVirtualSubgrid(
      Patch& subgrid
    );

    /**
     * Restricts to virtual subgrids that overlap with the current
     * subgrid.
     *
     * While basically this should restrict to all
     * overlapping virtual subgrids for optimization reasons this is
     * not true. Restriction can be omitted if the virtual subgrid
     * (or the part of it that is overlapped by the current subgrid)
     * for sure will not be required in the current grid iteration.
     *
     * This, for example, is true if the timestepping of all
     * affected coarse subgrids will be blocked by the current subgrid.
     *
     * It also is true if no ghostlayer overlaps with the current
     * subgrid.
     *
     * TODO unterweg dissertation Optimierung für Restriktion siehe oben.
     *
     */
    void restrictToOverlappingVirtualSubgrids(
      const Patch&           subgrid,
      const ParallelSubgrid& fineParallelSubgrid
    );

    /**
     * Fills the surrounding ghostlayers of a virtual subgrid and
     * switches the subgrid to non-virtual, while removing it
     * from the virtual subgrids-list.
     */
    void finalizeVirtualSubgrid(
      Patch&                               subgrid,
      peanoclaw::Vertex * const            fineGridVertices,
      const peano::grid::VertexEnumerator& fineGridVerticesEnumerator,
      bool                                 isPeanoCellLeaf
    );

    tarch::la::Vector<DIMENSIONS_PLUS_ONE, double> createVirtualSubgridKey(
      tarch::la::Vector<DIMENSIONS, double> position,
      int                                   level
    ) const;

  public:
    GridLevelTransfer(
      bool useDimensionalSplitting,
      peanoclaw::Numerics& numerics
    );

    virtual ~GridLevelTransfer();

#ifdef Parallel
    /**
     * Called whenever a patch is merged from a remote MPI rank during
     * master/worker communication.
     *
     * @param cellDescriptionIndex The index for the current patch.
     * @param needsToHoldGridData Indicates whether this patch is supposed to
     * hold grid data. Either as a leaf patch of as a virtual patch.
     */
    void updatePatchStateDuringMergeWithWorker(
      int  localCellDescriptionIndex,
      int  remoteCellDescriptionIndex
    );
#endif

    /**
     * Performs the operations necessary when stepping into a cell.
     */
    void stepDown(
      int                                  coarseCellDescriptionIndex,
      Patch&                               fineSubgrid,
      peanoclaw::Vertex * const            fineGridVertices,
      const peano::grid::VertexEnumerator& fineGridVerticesEnumerator,
      bool                                 isInitializing,
      bool                                 isPeanoCellLeaf
    );

    /**
     * @param isPeanoCellLeaf We need this flag, since during a refinement/coarsening the semantics
     * of Patch::isLeaf() and Cell::isLeaf() may differ. The first describes, wether a patch holds
     * valid grid data and is not virtual, while the latter describes the actual spacetree state of
     * a Peano cell. Hence, for setting the Patch state correctly we here need to know about the
     * state of the containing Peano cell.
     */
    void stepUp(
      int                                  coarseCellDescriptionIndex,
      Patch&                               finePatch,
      ParallelSubgrid&                     fineParallelSubgrid,
      bool                                 isPeanoCellLeaf,
      peanoclaw::Vertex * const            fineGridVertices,
      const peano::grid::VertexEnumerator& fineGridVerticesEnumerator
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
     * Restricts the data of a subgrid that is about to be destroyed
     * to the overlapping coarse subgrid.
     */
    void restrictDestroyedSubgrid(
      const Patch&                         destroyedSubgrid,
      Patch&                               coarseSubgrid,
      peanoclaw::Vertex * const            fineGridVertices,
      const peano::grid::VertexEnumerator& fineGridVerticesEnumerator
    );
};

#endif /* PEANO_APPLICATIONS_PEANOCLAW_GRIDLEVELTRANSFER_H_ */
