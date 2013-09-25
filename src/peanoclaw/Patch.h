/*
 * Patch.h
 *
 *      Author: Kristof Unterweger
 */

#ifndef PEANOCLAW_PATCH_H_
#define PEANOCLAW_PATCH_H_

#include <vector>

#include "records/Data.h"
#include "records/CellDescription.h"

#include "peano/heap/Heap.h"
#include "peano/utils/Globals.h"
#include "tarch/la/Vector.h"
#include "tarch/logging/Log.h"

//Optimizations
#define PATCH_DISABLE_RANGE_CHECK
#define PATCH_INLINE_GETTERS_AND_SETTERS

namespace peanoclaw {
  class Area;
  class Cell;
  class Patch;
}

/**
 * Hides the storage of Patchdata from the user. I.e. this maps data
 * accesses from a subcell-pattern to the actual array stored within
 * the heap data of the Peano cells.
 *
 * It also encaspulates the fact that there are always two copies of
 * the data, here called uNew and uOld. Since uOld also holds the
 * boundary conditions, the size of both arrays does not match (uNew
 * is subdivisionFactor x subdivisionFactor, while uOld is
 * (subdivisionFactor+ghostLayerWidth*2) x (subdivisionFactor +
 * ghostLayerWidth*2)).
 *
 * To make the access pattern more intuitive, the ghostlayer is neglected
 * when indexing the arrays. I.e. indices in one dimension are in the
 * interval [0..subdivisionFactor-1] for uNew and in
 * [-ghostLayerWidth..subdivisionFactor+ghostLayerWidth*2-1] for uOld. By
 * this the subcell (x,y) is accessed by the same indices for uOld and uNew.
 *
 * !!! States
 * Each spacetree node is considered of holding a patch. However, not each
 * patch actually holds valid grid data. Patches can have three states:
 *
 * !! Leaf
 * A leaf patch is a patch located in a leaf node of the spacetree grid.
 * That means it contains valid arrays for the current and the last
 * timestep. The currentTime and timestepSize fields span the time interval
 * that corresponds to the stored grid data.
 *
 * !! Refined
 * A refined patch is a patch that is located in an inner node of the
 * spacetree. It does not contain valid arrays neither for the current
 * nor the last timestep. The only exception is a virtual patch.
 *
 * The question is what the time interval represents for a refined patch.
 *
 * The currentTime and timestepSize correspond to the time interval spanned
 * by all overlapped fine cells. I.e. this is the set of points that is
 * covered by all child patches of this patch. In the figure below this is
 * called "minimal fine patch time interval". By this approach all time
 * stepping restrictions on neighboring patches can be done in a natural
 * way, since the restrictions by finer patches can be applied by just
 * taking the restriction of the time interval spanned by a refined patch.
 *
 * Since this time interval is gathered during the traversal of the child
 * patches, this information refers to the last grid iteration in the
 * enterCell and touchVertexFirstTime events and refers to the current grid
 * iteration in the leaveCell, touchVertexLastTime, and destroyCell events.
 * In all other events this information is not reliable.
 *
 * There is one more relevant time interval for a refined patch: The maximum
 * neighbor time interval spans the time interval between the oldest
 * currentTime and the newest currentTime+timestepSize of all neighbor
 * patches. This time is valid from including enterCell.
 *
 * The images afterwards show the idea of the mentioned time intervals. The
 * first one shows a situation in 2d with a refined patch. The fact that
 * this refinement pattern is illegal in Peano is here ignored for the sake
 * of simplicity. The second image shows the same example but in 1d to
 * clarify the different time intervals.
 *
 * !! Virtual
 * A virtual patch is a special kind of refined patch. Virtual patches hold
 * valid grid data, which are created by restricting the grid data of
 * underlying fine patches. This is used for filling ghostlayers of
 * neighboring coarse patches. Since the child patches span different time
 * intervals, some kind of mapping needs to be done to get a time interval
 * that is spanned by the virtual patch. For this we use the "maximum
 * neighbor time interval" and inter-/extrapolate the fine patch data to
 * this interval. After the iteration of the child patches the actual
 * time interval for the refined patch is known, so the data can be
 * interpolated from the "maximum neighbor time interval" to the
 * "minimal fine patch time interval".
 *
 * @image html refined-patch-time-intervals-2d.png
 *
 * @image html refined-patch-time-intervals-1d.png
 *
 * The second image also shows the minimal neighbor time. This is the
 * minimum over the upper limit of the time intervals of the neighboring
 * patches. This restricts the time stepping for this patch, if it is a
 * leaf patch, or the underlying fine patches, if the current patch is
 * refined.
 *
 * TODO: Rewrite the Patch class so that it directly works on the CellDescriptions.
 * So, we have just a frontend and each Patch contains a valid CellDescription object.
 */
class peanoclaw::Patch {

public:
  typedef peanoclaw::records::CellDescription CellDescription;
  typedef peanoclaw::records::Data Data;

private:
  /**
   * Logging device.
   */
  static tarch::logging::Log _log;

  CellDescription* _cellDescription;

  std::vector<Data>* _uNew;

//  std::vector<Data>* _uOldWithGhostlayer;
  int                _uOldWithGhostlayerArrayIndex;
//  std::vector<Data>* _auxArray;
  int                _auxArrayIndex;

  int uNewStrideCache[DIMENSIONS+1];
  int uOldStrideCache[DIMENSIONS+1];

  tarch::la::Vector<DIMENSIONS, double> _subcellSize;

  /**
   * Linearizes the given subcell index to a columnwise stored array index.
   *
   * This method assumes that the unknown is the first index of the d-dimensional array,
   * while the subcell index-components follow as indices.
   */
  int linearize(
    int unknown,
    const tarch::la::Vector<DIMENSIONS, int>& subcellIndex
  ) const;

  /**
   * Does a similar linearization like the method linearize(...), but it assumes
   * that the array is surrounded by a ghostlayer.
   */
  int linearizeWithGhostlayer(
    int unknown,
    const tarch::la::Vector<DIMENSIONS, int>& subcellIndex
  ) const
  #ifdef PATCH_INLINE_GETTERS_AND_SETTERS
  {
    int index = 0;
    int ghostlayerWidth = _cellDescription->getGhostLayerWidth();

    for(int d = 0; d < DIMENSIONS; d++) {
      index += (subcellIndex(d) + ghostlayerWidth) * uOldStrideCache[d+1];
    }
    index += unknown * uOldStrideCache[0];
    return index;
  }
  #else
  ;
  #endif

  /**
   * Fills caches and precomputes data.
   */
  void fillCaches();

  /**
   * Switches the values in the given area to the minimal fine grid time interval.
   */
  void switchAreaToMinimalFineGridTimeInterval(
    const peanoclaw::Area& area,
    double factorForUOld,
    double factorForUNew
  );

public:
  /**
   * Dummy constructor for arrays
   */
  Patch();

  Patch(
    const peanoclaw::Cell& cell
  );

  Patch(
    CellDescription& cellDescription
  );

  Patch(
    int cellDescriptionIndex
  );

  /**
   * Creates a new patch by allocating the necessary resources
   * on the heap.
   */
  Patch(
    const tarch::la::Vector<DIMENSIONS, double>& position,
    const tarch::la::Vector<DIMENSIONS, double>& size,
    int    unknownsPerCell,
    int    auxiliarFieldsPerSubcell,
    const tarch::la::Vector<DIMENSIONS, int>&    subdivisionFactor,
    int    ghostLayerWidth,
    double initialTimestepSize,
    int    level
  );

  ~Patch();

  /**
   * Loads data from the given cell description object. This has
   * to be done after data got changed in the cell description
   * or when a different cell description is now describing this
   * patch than when constructing the patch.
   *
   * @todo: Remove this method and do all manipulation of
   * cell descriptions within the patch class.
   */
  void loadCellDescription(int cellDescriptionIndex);

  void reloadCellDescription();

  /**
   * Deletes the complete patch. After this method the patch itself
   * is not valid anymore.
   */
  void deleteData();

  const tarch::la::Vector<DIMENSIONS, double> getSize() const;

  const tarch::la::Vector<DIMENSIONS, double> getPosition() const;

  int getUnknownsPerSubcell() const;

  int getAuxiliarFieldsPerSubcell() const;

  tarch::la::Vector<DIMENSIONS, int> getSubdivisionFactor() const;

  int getGhostLayerWidth() const;

  /**
   * The current time refers to the lower end of the time interval spanned
   * by this patch and thus to the values stored in uOld.
   * I.e. getCurrentTime() + getTimestepSize() refers to the upper end of
   * the time interval and thus to the values stored in uNew.
   */
  double getCurrentTime() const;

  /**
   * Sets the current time for this patch.
   */
  void setCurrentTime(double currentTime);

  /**
   * Returns the time that refers to the grid values in uOld. For leaf
   * patches this returns the same like getCurrentTime(), but for
   * virtual patches this refers to the maximal neighbor time interval.
   */
  double getTimeUOld() const;

  /**
   * Returns the time that refers to the grid values in uNew. For leaf
   * patches this returns the same like getCurrentTime()+getTimestepSize(),
   * but for virtual patches this refers to the maximal neighbor time
   * interval.
   */
  double getTimeUNew() const;

  /**
   * Updates the current time of this patch by the timestep size for this patch.
   */
  void advanceInTime();

  /**
   * The length of the time interval spanned by this patch.
   */
  double getTimestepSize() const;

  /**
   * Sets the length of the time interval spanned by this patch.
   */
  void setTimestepSize(double timestepSize);

  /**
   * Returns the guess for the size of the next timestep.
   */
  double getEstimatedNextTimestepSize() const;

  /**
   * Sets the guess for the next timestep.
   */
  void setEstimatedNextTimestepSize(double estimatedNextTimestepSize);

  /**
   * Returns the minimum of the points in time on which all (also
   * coarser) neighbors reside.
   */
  double getMinimalNeighborTimeConstraint() const;

  /**
   * Returns the minimum of the points in time on which all leaf
   * neighbors reside.
   */
  double getMinimalLeafNeighborTimeConstraint() const;

  /**
   * Updates the minimum of the points in time on which all (also
   * coarser) neighbors reside. If the given neighbor time is
   * larger than the current minimum this method doesn't change
   * anything.
   */
  void updateMinimalNeighborTimeConstraint(double neighborTime, int neighborIndex);

  /**
   * Updates the minimum of the points in time on which all (also
   * coarser) neighbors reside which are leaf patches. If the given
   * neighbor time is larger than the current minimum this method
   * doesn't change anything.
   */
  void updateMinimalLeafNeighborTimeConstraint(double leafNeighborTime);

  /**
   * Resets the minimal neighbor time to start a new minimum
   * search.
   */
  void resetMinimalNeighborTimeConstraint();

  /**
   * Returns the index of the subgrid that implies the time constraint
   * on this subgrid.
   */
  int getConstrainingNeighborIndex() const;

  /**
   * Resets the maximal time interval that is overlapped by
   * at least one neighbor.
   */
  void resetMaximalNeighborTimeInterval();

  /**
   * Updates the timeinterval that is overlapped by at least
   * one neighbor patch.
   */
  void updateMaximalNeighborTimeInterval(double neighborTime, double neighborTimestepSize);

  /**
   * Sets the time interval spanned by this patch to the maximal
   * time interval spanned by all neighbor patches. This is useful
   * for virtual patches for which the grid data of the fine patches
   * are restricted to the maximal neighbor time interval.
   */
//  void setTimeIntervalToMaximalNeighborTimeInterval();

  /**
   * Returns wether this patch currently is allowed to advance
   * in time, depending on the neighbor patches.
   */
  bool isAllowedToAdvanceInTime() const;

  /**
   * Returns the level on which this patch resides.
   */
  int getLevel() const;

  /**
   * Prepares the values for the maximum/minimum search in the fine
   * grid time intervals.
   */
  void resetMinimalFineGridTimeInterval();

  /**
   * Updates the values for the maximum/minimum search in the fine
   * grid time intervals.
   */
  void updateMinimalFineGridTimeInterval(double fineGridTime, double fineGridTimestepSize);

  /**
   * Returns the maximum time that is covered by this patch. This
   * time constrains the timestepping of theneighboring patches.
   * Usually this method returns $currentTime + timestepSize$, but
   * for a virtual patch it returns
   * $maximumFineGridTime + minimumFineGridTimestepSize$, because
   * a virtual patch stores the extrapolated grid data with respect
   * to the maximum neighbor time interval, but the time stepping
   * constraint is due to the minimum time of the fine grids covered
   * by the virtual patch.
   */
  double getTimeConstraint() const;

  /**
   * Sets, whether the fine grids of this Patch should synchronize
   * in time to be coarsened.
   */
  void setFineGridsSynchronize(bool synchronizeFineGrids);

  /**
   * Returns, whether the fine grids of this Patch should synchronize
   * in time to be coarsened.
   */
  bool shouldFineGridsSynchronize() const;

  /**
   * Sets, whether this patch will coarsen in this iteration
   * or not.
   */
  void setWillCoarsen(bool willCoarsen);

  /**
   * Returns, whether this patch will coarsen in this iteration
   * or not.
   */
  bool willCoarsen();

  /**
   * Sets the time interval of this patch to the minimal time
   * interval covered by all fine patches overlapped by this patch
   * and interpolates the patch values accordingly.
   */
  void switchValuesAndTimeIntervalToMinimalFineGridTimeInterval();

  /**
   * Sets, wether this patch should be skipped during the next grid
   * iteration with regard to the adaptive timestepping.
   */
  void setSkipNextGridIteration(int numberOfIterationsToSkip);

  /**
   * Returns wether this patch should be skipped during the next grid
   * iterations with regard to the adaptive timestepping.
   */
  bool shouldSkipNextGridIteration() const;

  /**
   * Reduces the number of grid iterations that should be skipped.
   */
  void reduceGridIterationsToBeSkipped();

  //GETTER/SETTER for grid values
  void setValueUNew(tarch::la::Vector<DIMENSIONS, int> subcellIndex, int unknown, double value)
  #ifdef PATCH_INLINE_GETTERS_AND_SETTERS
  {
    assertion(isLeaf() || isVirtual());
    int index = linearize(unknown, subcellIndex);
    assertion3(index >= 0, index, subcellIndex, unknown);
    assertion4(index < static_cast<int>(_uNew->size()), index, subcellIndex, unknown, static_cast<int>(_uNew->size()));
    #ifdef PATCH_RANGE_CHECK
    _uNew->at(index).setU(value);
    #else
    (*_uNew)[index].setU(value);
    #endif
  }
  #else
  ;
  #endif

  void setValueUOld(tarch::la::Vector<DIMENSIONS, int> subcellIndex, int unknown, double value)
  #ifdef PATCH_INLINE_GETTERS_AND_SETTERS
  {
    assertion(isLeaf() || isVirtual());
    int index = linearizeWithGhostlayer(unknown, subcellIndex);
    assertion3(index >= 0, index, subcellIndex, unknown);
    assertion5(index < _auxArrayIndex - _uOldWithGhostlayerArrayIndex, index, subcellIndex, unknown, _auxArrayIndex - _uOldWithGhostlayerArrayIndex, toString());
    #ifdef PATCH_RANGE_CHECK
    _uNew->at(_uOldWithGhostlayerArrayIndex + index).setU(value);
    #else
    (*_uNew)[_uOldWithGhostlayerArrayIndex + index].setU(value);
    #endif
  }
  #else
  ;
  #endif

  double getValueUNew(tarch::la::Vector<DIMENSIONS, int> subcellIndex, int unknown) const
  #ifdef PATCH_INLINE_GETTERS_AND_SETTERS
  {
    assertion1(isLeaf() || isVirtual(), toString());
    int index = linearize(unknown, subcellIndex);
    assertion3(index >= 0, index, subcellIndex, unknown);
    assertion5(index < static_cast<int>(_uNew->size()), index, subcellIndex, unknown, static_cast<int>(_uNew->size()), toString());
    #ifdef PATCH_RANGE_CHECK
    return _uNew->at(index).getU();
    #else
    return (*_uNew)[index].getU();
    #endif
  }
  #else
  ;
  #endif


  double getValueUNew(tarch::la::Vector<DIMENSIONS, double> subcellPosition, int unknown) const;

  double getValueUOld(tarch::la::Vector<DIMENSIONS, int> subcellIndex, int unknown) const
  #ifdef PATCH_INLINE_GETTERS_AND_SETTERS
  {
    assertion1(isLeaf() || isVirtual(), toString());
    int index = linearizeWithGhostlayer(unknown, subcellIndex);
    assertion4(index >= 0, index, subcellIndex, unknown, toString());
//    assertion5(index < static_cast<int>(_uOldWithGhostlayer->size()), index, subcellIndex, unknown, static_cast<int>(_uOldWithGhostlayer->size()), toString());
    assertion5(index < _auxArrayIndex - _uOldWithGhostlayerArrayIndex, index, subcellIndex, unknown, _auxArrayIndex - _uOldWithGhostlayerArrayIndex, toString());
    #ifdef PATCH_RANGE_CHECK
    return _uNew->at(_uOldWithGhostlayerArrayIndex + index).getU();
    #else
    return (*_uNew)[_uOldWithGhostlayerArrayIndex + index].getU();
    #endif
  }
  #else
  ;
  #endif

  /**
   * Returns the linear index of the given subcell, which can be used in the
   * getter and setter methods to the uNew values accepting a linear index.
   */
  int getLinearIndexUNew(tarch::la::Vector<DIMENSIONS, int> subcellIndex) const
  #ifdef PATCH_INLINE_GETTERS_AND_SETTERS
  {
    return linearize(0, subcellIndex);
  }
  #else
  ;
  #endif

  /**
   * Returns the linear index of the given subcell, which can be used in the
   * getter and setter methods to the uOld values accepting a linear index.
   */
  int getLinearIndexUOld(tarch::la::Vector<DIMENSIONS, int> subcellIndex) const
  #ifdef PATCH_INLINE_GETTERS_AND_SETTERS
  {
    return linearizeWithGhostlayer(0, subcellIndex);
  }
  #else
  ;
  #endif

  double getValueUNew(int linearIndex, int unknown) const
  #ifdef PATCH_INLINE_GETTERS_AND_SETTERS
  {
    int index = linearIndex + uNewStrideCache[0] * unknown;
    #ifdef PATCH_RANGE_CHECK
    return _uNew->at(index).getU();
    #else
    return (*_uNew)[index].getU();
    #endif
  }
  #else
  ;
  #endif

  void setValueUNew(int linearIndex, int unknown, double value)
  #ifdef PATCH_INLINE_GETTERS_AND_SETTERS
  {
    int index = linearIndex + uNewStrideCache[0] * unknown;
    #ifdef PATCH_RANGE_CHECK
    _uNew->at(index).setU(value);
    #else
    (*_uNew)[index].setU(value);
    #endif
  }
  #else
  ;
  #endif

  double getValueUOld(int linearIndex, int unknown) const
  #ifdef PATCH_INLINE_GETTERS_AND_SETTERS
  {
    int index = linearIndex + uOldStrideCache[0] * unknown;
    #ifdef PATCH_RANGE_CHECK
    return _uNew->at(_uOldWithGhostlayerArrayIndex + index).getU();
    #else
    return (*_uNew)[_uOldWithGhostlayerArrayIndex + index].getU();
    #endif
  }
  #else
  ;
  #endif

  void setValueUOld(int linearIndex, int unknown, double value)
  #ifdef PATCH_INLINE_GETTERS_AND_SETTERS
  {
    int index = linearIndex + uOldStrideCache[0] * unknown;
    #ifdef PATCH_RANGE_CHECK
    _uNew->at(_uOldWithGhostlayerArrayIndex + index).setU(value);
    #else
    (*_uNew)[_uOldWithGhostlayerArrayIndex + index].setU(value);
    #endif
  }
  #else
  ;
  #endif

  double getValueAux(tarch::la::Vector<DIMENSIONS, int> subcellIndex, int auxField) const;

  tarch::la::Vector<DIMENSIONS, double> getSubcellCenter(tarch::la::Vector<DIMENSIONS, int> subcellIndex) const;

  tarch::la::Vector<DIMENSIONS, double> getSubcellPosition(tarch::la::Vector<DIMENSIONS, int> subcellIndex) const;

  tarch::la::Vector<DIMENSIONS, double> getSubcellSize() const;

  /**
   * Copies the data from uNew to uOld, i.e. this actually performs the
   * transition to the next timestep by setting uOld(t) to uOld(t+1).
   * The ghost layer stays unchanged.
   */
  void copyUNewToUOld();

  /**
   * Sets the specified region of the patch.
   */
  void clearRegion(
    tarch::la::Vector<DIMENSIONS, int> offset,
    tarch::la::Vector<DIMENSIONS, int> size,
    bool clearUOld
  );

  /**
   * Returns the uNew double array.
   */
  double* getUNewArray() const;

  /**
   * Returns the uOld double array.
   */
  double* getUOldWithGhostlayerArray() const;

  /**
   * Returns the aux array as a real double array.
   */
  double* getAuxArray() const;

  /**
   * Returns the index for the uNew array.
   */
  int getUNewIndex() const;

  /**
   * Returns the index for the uOld array.
   */
//  int getUOldIndex() const;

  /**
   * Returns the index for the aux array.
   */
//  int getAuxIndex() const;

  /**
   * Returns the index for the CellDescription. Not always valid,
   * only for testing.
   */
  int getCellDescriptionIndex() const;

  /**
   * Sets the mesh width that was demanded by the application for
   * this patch.
   */
  void setDemandedMeshWidth(double demandedMeshWidth);

  /**
   * Returns the mesh width that was demanded by the application for
   * this patch.
   */
  double getDemandedMeshWidth() const;

  std::string toStringUNew() const;

  std::string toStringUOldWithGhostLayer() const;

  std::string toString() const;

  /**
   * This method returns, wether this patch is valid, i.e. if it holds real grid
   * data. Invalid patches are patches that do not exist in the grid, i.e. lie
   * outside of the domain.
   */
  bool isValid() const;

  /**
   * This method returns, wether this patch is a leaf patch, i.e. if it holds
   * reals grid data or if it is a refined patch that is overlapped by finer
   * patches. Refined patches hold correct time values but no valid grid
   * data. The time that is held by a refined patch resembles the time span that
   * is covered by all subpatches. I.e. currentTime is the maximum of all the
   * currentTimes of all subpatches and currentTime+timestepSize is the minimum
   * of all currentTime+timestepSize of all subpatches.
   */
  bool isLeaf() const;

  /**
   * This method returs, wether this patch is a virtual patch, i.e. if it holds
   * temporary grid data that has been restricted from finer grids.
   */
  bool isVirtual() const;

  /**
   * Changes this patch to a virtual patch. If the patch was a
   * refined patch before, the uNew and uOld arrays are created.
   */
  void switchToVirtual();

  /**
   * Changes this patch to a non-virtual patch and deletes all grid data.
   * Thus, the patch becomes a refined patch.
   */
  void switchToNonVirtual();

  /**
   * Switches a virtual patch to a leaf patch.
   */
  void switchToLeaf();

  /**
   * Returns whether this patch should restrict to virtual patches.
   */
  bool shouldRestrict();

  /**
   * Checks wether uNew or uOld contains NaN.
   */
  bool containsNaN() const;

  /**
   * Checks wether uNew or uOld contain non-positive numbers in the
   * given unknown.
   */
  bool containsNonPositiveNumberInUnknown(int unknown) const;

  /**
   * Increases the age of this patch in grid iterations by one.
   */
  void increaseAgeByOneGridIteration();

  /**
   * Returns the age of this patch in terms of grid iterations.
   */
  int getAge() const;

  /**
   * Resets the bounds for the overlap of neighboring ghostlayers.
   */
  void resetNeighboringGhostlayerBounds();

  /**
   * Updates the lower bound for overlapping ghostlayers in the given dimension.
   * I.e. this is the bound for the overlap of the patch located "above" in
   * the given domain.
   */
  void updateLowerNeighboringGhostlayerBound(int dimension, double bound);

  /**
   * Updates the upper bound for overlapping ghostlayers in the given dimension.
   * I.e. this is the bound for the overlap of the patch located "below" in
   * the given domain.
   */
  void updateUpperNeighboringGhostlayerBound(int dimension, double bound);

  /**
   * Returns the lower bounds for all dimensions that define how far this patch
   * is overlapped by neighboring ghostlayers.
   */
  tarch::la::Vector<DIMENSIONS, double> getLowerNeighboringGhostlayerBounds() const;

  /**
   * Returns the upper bounds for all dimensions that define how far this patch
   * is overlapped by neighboring ghostlayers.
   */
  tarch::la::Vector<DIMENSIONS, double> getUpperNeighboringGhostlayerBounds() const;

  #ifdef Parallel
  /**
   * Sets whether this patch is a remote patchm, i.e. if it actually resides on another
   * MPI rank.
   */
  void setIsRemote(bool isRemote);

  /**
   * Returns whether this patch is a remote patch, i.e. if actually resides on another
   * MPI-rank.
   */
  bool isRemote() const;
  #endif
};

std::ostream& operator<<(std::ostream& out, const peanoclaw::Patch& patch);

#endif /* PEANO_APPLICATIONS_PEANOCLAW_PATCH_H_ */
