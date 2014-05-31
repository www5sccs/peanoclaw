/*
 * SubgridAccessor.h
 *
 *  Created on: Apr 30, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_GRID_SUBGRIDACCESSOR_H_
#define PEANOCLAW_GRID_SUBGRIDACCESSOR_H_

#include "peanoclaw/grid/Linearization.h"
#include "peanoclaw/records/CellDescription.h"
#include "peanoclaw/records/Data.h"

#include "peano/utils/Dimensions.h"
#include "tarch/la/Vector.h"

#include <vector>

namespace peanoclaw {

  class Patch;

  namespace grid {
    class SubgridAccessor;

    template<int NumberOfUnknowns>
    class SubgridIterator;
  }
}

/**
 * Iterator for the cells of a subgrid. The access can be limited to
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
class peanoclaw::grid::SubgridIterator {

  private:
    typedef peanoclaw::records::CellDescription CellDescription;
    typedef peanoclaw::records::Data Data;

    std::vector<Data>& _u;
    tarch::la::Vector<DIMENSIONS, int> _offset;
    tarch::la::Vector<DIMENSIONS, int> _size;

    tarch::la::Vector<DIMENSIONS,int> _position;
    int _unknown;
    int _indexUNew;
    int _indexUOld;

    SubgridAccessor& _accessor;
    CellDescription& _cellDescription;
    tarch::la::Vector<DIMENSIONS, int> _offsetPlusSize;

    Linearization     _linearization;
    int _uNewAllUnknownsStride;
    int _uOldAllUnknownsStride;

    bool _isValid;

  public:
    /**
     * Create a new accessor for the given subgrid and
     * restrict the access to the area defined by
     * offset and size. This defines a rectangular
     * part of the subgrid.
     */
    SubgridIterator(
      SubgridAccessor& accessor,
      CellDescription& cellDescription,
      Linearization& linearization,
      std::vector<Data>& u,
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

/**
 * Encapsulation of the SubgridIterator for operations independent
 * of the conrete knowledge of the number of unknowns.
 */
class peanoclaw::grid::SubgridAccessor {

  private:
    template<int NumberOfUnknowns> friend class SubgridIterator;
    friend class peanoclaw::Patch;

    typedef peanoclaw::records::Data Data;
    typedef peanoclaw::records::CellDescription CellDescription;

    CellDescription _cellDescription;
    bool               _isLeaf;
    bool               _isVirtual;
    tarch::la::Vector<DIMENSIONS,int> _subdivisionFactor;
    int _ghostlayerWidth;
    std::vector<Data>* _u;
    int _uOldWithGhostlayerArrayIndex;
    int _parameterWithoutGhostlayerArrayIndex;
    int _parameterWithGhostlayerArrayIndex;
    Linearization      _linearization;

  public:
    SubgridAccessor()
      : _u(0) {
    }

    SubgridAccessor(
        bool isLeaf,
        bool isVirtual,
        CellDescription& cellDescription,
        std::vector<Data>* u
    ) : _cellDescription(cellDescription),
        _isLeaf(isLeaf),
        _isVirtual(isVirtual),
        _u(u),
        _uOldWithGhostlayerArrayIndex(-1),
        _parameterWithoutGhostlayerArrayIndex(-1),
        _parameterWithGhostlayerArrayIndex(-1),
        _linearization(
            cellDescription.getSubdivisionFactor(),
            cellDescription.getUnknownsPerSubcell(),
            cellDescription.getNumberOfParametersWithoutGhostlayerPerSubcell(),
            cellDescription.getNumberOfParametersWithGhostlayerPerSubcell(),
            cellDescription.getGhostlayerWidth()
    ) {

      int volumeNew = tarch::la::volume(cellDescription.getSubdivisionFactor());
      int volumeOld = tarch::la::volume(cellDescription.getSubdivisionFactor() + 2*cellDescription.getGhostlayerWidth());

      _uOldWithGhostlayerArrayIndex = volumeNew * cellDescription.getUnknownsPerSubcell();
      _parameterWithoutGhostlayerArrayIndex = _uOldWithGhostlayerArrayIndex + volumeOld * cellDescription.getUnknownsPerSubcell();
      _parameterWithGhostlayerArrayIndex = _parameterWithoutGhostlayerArrayIndex + volumeNew * cellDescription.getNumberOfParametersWithoutGhostlayerPerSubcell();
    }

    template<int NumberOfUnknowns>
    peanoclaw::grid::SubgridIterator<NumberOfUnknowns> getSubgridIterator(
      const tarch::la::Vector<DIMENSIONS, int> offset,
      const tarch::la::Vector<DIMENSIONS, int> size
    );

    /**
     * Returns the value for the given cell index
     */
    double getValueUNew(tarch::la::Vector<DIMENSIONS, int> cellIndex, int unknown) const;

    void setValueUNew(tarch::la::Vector<DIMENSIONS, int> subcellIndex, int unknown, double value);

    void setValueUOld(tarch::la::Vector<DIMENSIONS, int> subcellIndex, int unknown, double value);

    double getValueUOld(tarch::la::Vector<DIMENSIONS, int> subcellIndex, int unknown) const;

    /**
     * Returns the linear index of the given subcell, which can be used in the
     * getter and setter methods to the uNew values accepting a linear index.
     */
    int getLinearIndexUNew(tarch::la::Vector<DIMENSIONS, int> subcellIndex) const;

    /**
     * Returns the linear index of the given subcell, which can be used in the
     * getter and setter methods to the uOld values accepting a linear index.
     */
    int getLinearIndexUOld(tarch::la::Vector<DIMENSIONS, int> subcellIndex) const;

    double getValueUNew(int linearIndex, int unknown) const;

    void setValueUNew(int linearIndex, int unknown, double value);

    /**
     * Sets the given value in the appropriate cell in the uNew array. If necessary
     * the array is resized to contain the specified cell.
     */
    void setValueUNewAndResize(int linearIndex, int unknown, double value);

    double getValueUOld(int linearIndex, int unknown) const;

    void setValueUOld(int linearIndex, int unknown, double value);

    /**
     * Sets the given value in the appropriate cell in the uOld array. If necessary
     * the array is resized to contain the specified cell.
     */
    void setValueUOldAndResize(int linearIndex, int unknown, double value);

    double getParameterWithoutGhostlayer(const tarch::la::Vector<DIMENSIONS, int>& subcellIndex, int parameter) const;
    void setParameterWithoutGhostlayer(const tarch::la::Vector<DIMENSIONS, int>& subcellIndex, int parameter, double value);

    double getParameterWithGhostlayer(const tarch::la::Vector<DIMENSIONS, int>& subcellIndex, int parameter) const;
    void setParameterWithGhostlayer(const tarch::la::Vector<DIMENSIONS, int>& subcellIndex, int parameter, double value);

    /**
     * Sets the specified region of the subgrid to zero.
     */
    void clearRegion(
      tarch::la::Vector<DIMENSIONS, int> offset,
      tarch::la::Vector<DIMENSIONS, int> size,
      bool clearUOld
    );
};

#include "peanoclaw/grid/SubgridAccessor.cpph"

#endif /* PEANOCLAW_GRID_SUBGRIDACCESSOR_H_ */
