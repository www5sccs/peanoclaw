/*
 * SubgridAccessor.h
 *
 *  Created on: Apr 30, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_GRID_SUBGRIDACCESSOR_H_
#define PEANOCLAW_GRID_SUBGRIDACCESSOR_H_

#include "peanoclaw/Patch.h"
#include "peanoclaw/records/Data.h"

#include "peano/utils/Dimensions.h"
#include "tarch/la/Vector.h"

namespace peanoclaw {
  namespace grid {
    template<int NumberOfUnknowns>
    class SubgridAccessor;
  }

  class Patch;
}

/**
 * Accessor for the cells of a subgrid. The access can be limited to
 * a certain part of the subgrid.
 *
 * Currently, there are two ways to access data with this accessor:
 *
 * while(a.moveToNextCell()) {
 *   while(a.moveToNextUnknown()) {
 *     a.setUnknownUOld(a.getUnknownUNew());
 *   }
 * }
 *
 * or
 *
 * while(a.moveToNextCell()) {
 *   a.setUnknownsUOld(a.getUnknownsUNew());
 * }
 *
 * These two ways must never be mixed for one object.
 *
 */
template<int NumberOfUnknowns>
class peanoclaw::grid::SubgridAccessor {

  private:
    Patch& _subgrid;
    double*  _data;
    tarch::la::Vector<DIMENSIONS, int> _offset;
    tarch::la::Vector<DIMENSIONS, int> _size;

    int _unknown;
    tarch::la::Vector<DIMENSIONS, int> _position;
    int _indexUNew;
    int _indexUOld;

    tarch::la::Vector<DIMENSIONS, int> _subdivisionFactor;
    tarch::la::Vector<DIMENSIONS, int> _offsetPlusSize;
    int _ghostlayerWidth;
    int _uNewUnknownStride;
    int _uOldUnknownStride;

    bool _isValid;

  public:
    /**
     * Create a new accessor for the given subgrid and
     * restrict the access to the area defined by
     * offset and size. This defines a rectangular
     * part of the subgrid.
     */
    SubgridAccessor(
      Patch& subgrid,
      const tarch::la::Vector<DIMENSIONS, int>& offset,
      const tarch::la::Vector<DIMENSIONS, int>& size
    );

    /**
     * Restarts the iterator.
     */
    void restart();

    /**
     * Restarts the iterator on a different part of the subgrid.
     */
    void restart(
      const tarch::la::Vector<DIMENSIONS, int>& offset,
      const tarch::la::Vector<DIMENSIONS, int>& size
    );

    /**
     * Returns the current unknown value for the current cell at the current timestamp.
     * This access is only valid if the accessor only iterates over inner cells of the
     * subgrid.
     */
    double getUnknownUNew() const;
    tarch::la::Vector<NumberOfUnknowns, double> getUnknownsUNew() const;

    /**
     * Returns the unknown value for the current cell at the previous timestamp.
     */
    double getUnknownUOld() const;
    tarch::la::Vector<NumberOfUnknowns, double> getUnknownsUOld() const;

    /**
     * Sets the given value in the current unknown entry.
     */
    void setUnknownUNew(double value);
    void setUnknownsUNew(const tarch::la::Vector<NumberOfUnknowns, double>& unknowns);

    /**
     * Sets the given value in the current unknown entry.
     */
    void setUnknownUOld(double value);
    void setUnknownsUOld(const tarch::la::Vector<NumberOfUnknowns, double>& unknowns);

    /**
     * Returns the index of the current unknown.
     */
    int getUnknownIndex() const;

    /**
     * Returns the index of the current cell with respect to the subgrid.
     */
    tarch::la::Vector<DIMENSIONS, int> getCellIndex() const;

    /**
     * Returns the lower left corner of the current cell.
     */
    tarch::la::Vector<DIMENSIONS, double> getCellPosition() const;

    /**
     * Moves the accessor to the next cell and returns whether
     * this was successful or not. If this method returns false
     * this means that no further cell is available.
     */
    bool moveToNextCell();

    /**
     * Moves the accessor to the next cell and returns whether
     * this was successful or not. If this method returns false
     * this means that no further unknown is available for the
     * current cell.
     */
    bool moveToNextUnknown();
};

#include "peanoclaw/grid/SubgridAccessor.cpph"

#endif /* PEANOCLAW_GRID_SUBGRIDACCESSOR_H_ */
