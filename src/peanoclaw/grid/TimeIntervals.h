/*
 * TimeIntervals.h
 *
 *  Created on: Dec 5, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_SUBGRID_TIMEINTERVALS_H_
#define PEANOCLAW_SUBGRID_TIMEINTERVALS_H_

#include "peanoclaw/Vertex.h"
#include "peanoclaw/records/CellDescription.h"

#include "peano/grid/aspects/VertexStateAnalysis.h"

#include "tarch/logging/Log.h"

namespace peanoclaw {
  class Patch;

  namespace grid {
    class TimeIntervals;
  }
}

class peanoclaw::grid::TimeIntervals {

  private:
    typedef peanoclaw::records::CellDescription CellDescription;

    /**
     * Logging device.
     */
    static tarch::logging::Log _log;

    CellDescription*      _cellDescription;

  public:
    /**
     * Default constructor. Creates an invalid object.
     */
    TimeIntervals();

    TimeIntervals(
      CellDescription* cellDescription
    );

    /**
     * Determines whether the given patch should be advanced in time.
     *
     * TODO unterweg debug
     * Here we may not block timestepping if a coarse vertex has
     * an Erase_Triggered but only if it is Erasing. In serial both
     * works but in parallel a triggered erase may be postboned due
     * to the parallel grid topology.
     */
    bool isAllowedToAdvanceInTime(
      double                                   maximumTimestepDueToGlobalTimestep,
      peanoclaw::Vertex * const                fineGridVertices,
      const peano::grid::VertexEnumerator&     fineGridVerticesEnumerator,
      peanoclaw::Vertex * const                coarseGridVertices,
      const peano::grid::VertexEnumerator&     coarseGridVerticesEnumerator
    );

    /**
     * Returns wether this patch currently is allowed to advance
     * in time, depending on the neighbor patches.
     */
    bool isBlockedByNeighbors() const;

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
    void setCurrentTime(double currentTime) {
      _cellDescription->setTime(currentTime);
    }

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
     * Returns a string representation of the current object.
     */
    std::string toString() const;
};

#endif /* PEANOCLAW_SUBGRID_TIMEINTERVALS_H_ */
