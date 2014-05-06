/*
 * SubgridAccessor.h
 *
 *  Created on: Apr 30, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_GRID_SUBGRIDACCESSOR_H_
#define PEANOCLAW_GRID_SUBGRIDACCESSOR_H_

#include "tarch/la/Vector.h"

namespace peanoclaw {
  namespace grid {
    class SubgridAccessor;
  }

  class Patch;
}

/**
 * Accessor for the cells of a subgrid. The access can be limited to
 * a certain part of the subgrid.
 */
class peanoclaw::grid::SubgridAccessor {

  private:
    Patch& _subgrid;
    tarch::la::Vector<DIMENSIONS, double> _offset;
    tarch::la::Vector<DIMENSIONS, double> _size;

  public:
    /**
     * Create a new accessor for the given subgrid and
     * restrict the access to the area defined by
     * offset and size. This defines a rectangular
     * part of the subgrid.
     */
    SubgridAccessor(
      Patch& subgrid,
      const tarch::la::Vector<DIMENSIONS, double>& offset,
      const tarch::la::Vector<DIMENSIONS, double>& size
    );

    /**
     * Returns the unknown value for the current cell.
     */
    double getUnknown(int unknown) const;

    /**
     * Returns the cell center of the current cell.
     */
    tarch::la::Vector<DIMENSIONS, double> getCellCenter() const;

    /**
     * Moves the accessor to the next cell and returns whether
     * this was successful or not. If this method returns false
     * this means that no further cell is available.
     */
    bool moveToNextCell();
};


#endif /* PEANOCLAW_GRID_SUBGRIDACCESSOR_H_ */
