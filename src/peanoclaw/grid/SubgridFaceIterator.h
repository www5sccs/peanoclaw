/*
 * FluxIterator.h
 *
 *  Created on: Dec 4, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_GRID_SUBGRIDFLUXITERATOR_H_
#define PEANOCLAW_GRID_SUBGRIDFLUXITERATOR_H_

#include "peanoclaw/geometry/HyperplaneRegion.h"
#include "peanoclaw/grid/Linearization.h"
#include "peanoclaw/records/CellDescription.h"
#include "peanoclaw/records/Data.h"

#include <vector>

namespace peanoclaw {
  namespace grid {
    template<int NumberOfUnknowns>
    class SubgridFaceIterator;
  }
}

template<int NumberOfUnknowns>
class peanoclaw::grid::SubgridFaceIterator {
  private:
    typedef peanoclaw::records::CellDescription CellDescription;
    typedef peanoclaw::records::Data Data;

    int _dimension;
    int _direction;
    peanoclaw::geometry::HyperplaneRegion _region;
    CellDescription& _cellDescription;
    Linearization& _linearization;
    std::vector<Data>& _u;
    int _fluxIndex;
    int _uNewIndex;
    int _uOldIndex;
    int _ghostCellIndex;
    int _ghostCellDistance;
    int _ghostLayerWidth;

    tarch::la::Vector<DIMENSIONS_MINUS_ONE,int> _position;
    tarch::la::Vector<DIMENSIONS_MINUS_ONE,int> _offsetPlusSize;

    /**
     * Normal pointing outwards of the interface.
     */
    tarch::la::Vector<DIMENSIONS,int> _normal;

    void restart();

    /**
     * Changes the current ghost cell distance by the
     * given increment. A positive increment moves to a
     * ghost cell further away from the interface while
     * a negative increment shortens the distance between
     * the interface cell and the ghost cell.
     *
     * @param increment The increment in number of cells.
     */
    void changeGhostCellDistance(int increment);

  public:
    SubgridFaceIterator(
      int dimension,
      int direction,
      CellDescription& cellDescription,
      Linearization& linearization,
      std::vector<Data>& u,
      const peanoclaw::geometry::HyperplaneRegion& region
    );

    /**
     * Moves the accessor to the next interface cell and returns
     * whether this was successful or not. If this method returns
     * false this means that no further cell is available.
     */
    bool moveToNextInterfaceCell();

    /**
     * Moves the accessor to the next ghost cell and returns whether
     * this was successful or not. If this method returns false
     * this means that no further unknown is available for the
     * current cell.
     */
    bool moveToNextGhostCell();

    /**
     * Skips the remaining ghost cells of the current
     * interface cell. After this method was called,
     * moveToNextInterfaceCell() can be called.s
     */
    void skipRemainingGhostCells();

    /**
     * Returns the index of subcell to which the iterator currently
     * refers.
     */
    tarch::la::Vector<DIMENSIONS_MINUS_ONE,int> getSubcellIndex() const;

    /**
     * Returns the fluxes for the current interface cell.
     */
    tarch::la::Vector<NumberOfUnknowns,double> getFluxes() const;

    /**
     * Set the fluxes for the current interface cell.
     */
    void setFluxes(const tarch::la::Vector<NumberOfUnknowns,double>& fluxes);

    /**
     * Returns the uOld values for the current inner cell adjacent to the
     * current interface cell.
     */
    tarch::la::Vector<NumberOfUnknowns,double> getUnknownsUOld() const;

    /**
     * Returns the uNew values for the current inner cell adjacent to the
     * current interface cell.
     */
    tarch::la::Vector<NumberOfUnknowns,double> getUnknownsUNew() const;

    /**
     * Returns the ghost values for the current inner cell adjacent to the
     * current interface cell.
     */
    tarch::la::Vector<NumberOfUnknowns,double> getGhostUnknowns() const;

    /**
     * Sets the uNew values for the current innter cell adjacent to the
     * current interface cell.
     */
    void setUnknownsUNew(const tarch::la::Vector<NumberOfUnknowns,double>& unknowns);
};

#include "peanoclaw/grid/SubgridFaceIterator.cpph"

#endif /* PEANOCLAW_GRID_SUBGRIDFLUXITERATOR_H_ */
