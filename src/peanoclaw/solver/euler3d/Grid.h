#ifndef PEANOCLAW_SOLVER_EULER3D_GRID_H
#define PEANOCLAW_SOLVER_EULER3D_GRID_H

#include "Cell.hpp"
#include "CellAccessor.hpp"

#include "CartesianGrid/Subgrid.hpp"

#include "CartesianGrid/Basic/BackBoundaryGrid.hpp"
#include "CartesianGrid/Basic/BackGhostGrid.hpp"
#include "CartesianGrid/Basic/BottomBoundaryGrid.hpp"
#include "CartesianGrid/Basic/BottomGhostGrid.hpp"
#include "CartesianGrid/Basic/FrontBoundaryGrid.hpp"
#include "CartesianGrid/Basic/FrontGhostGrid.hpp"
#include "CartesianGrid/Basic/Grid.hpp"
#include "CartesianGrid/Basic/InnerGrid.hpp"
#include "CartesianGrid/Basic/LeftBoundaryGrid.hpp"
#include "CartesianGrid/Basic/LeftGhostGrid.hpp"
#include "CartesianGrid/Basic/RightBoundaryGrid.hpp"
#include "CartesianGrid/Basic/RightGhostGrid.hpp"
#include "CartesianGrid/Basic/TopBoundaryGrid.hpp"
#include "CartesianGrid/Basic/TopGhostGrid.hpp"

#include "NumericsMethods/TwoStepMethods/Grid.hpp"

#include "Cell.h"

#include "records/CellUnknown.h"
#include "records/EulerCellDescription.h"

#include <vector>

typedef CartesianGrid::Basic::LeftBoundaryGrid<CellAccessor>
  LeftBoundaryGrid;
typedef CartesianGrid::Basic::RightBoundaryGrid<CellAccessor>
  RightBoundaryGrid;
typedef CartesianGrid::Basic::BottomBoundaryGrid<CellAccessor>
  BottomBoundaryGrid;
typedef CartesianGrid::Basic::TopBoundaryGrid<CellAccessor>
  TopBoundaryGrid;
typedef CartesianGrid::Basic::BackBoundaryGrid<CellAccessor>
  BackBoundaryGrid;
typedef CartesianGrid::Basic::FrontBoundaryGrid<CellAccessor>
  FrontBoundaryGrid;

class Grid
  : public NumericsMethods::TwoStepMethods::Grid<Cell, double>,
    public CartesianGrid::Basic::Grid {
private:
  typedef NumericsMethods::TwoStepMethods::Grid<Cell, double> Base1;
  typedef CartesianGrid::Basic::Grid                          Base2;

  typedef euler::DataHeap             DataHeap;
  typedef euler::records::CellUnknown PersistentCell;
  typedef std::vector<PersistentCell> PersistentCells;

public:
  typedef euler::records::EulerCellDescription Descriptor;

  typedef CartesianGrid::SubgridIterator<CellAccessor> Iterator;
  typedef typename Iterator::Factory                   IteratorFactory;

  typedef CartesianGrid::Basic::InnerGrid<CellAccessor>     InnerGrid;
  typedef CartesianGrid::Basic::LeftGhostGrid<CellAccessor> LeftGhostGrid;
  typedef CartesianGrid::Basic::RightGhostGrid<CellAccessor>
    RightGhostGrid;
  typedef CartesianGrid::Basic::BottomGhostGrid<CellAccessor>
    BottomGhostGrid;
  typedef CartesianGrid::Basic::TopGhostGrid<CellAccessor>
    TopGhostGrid;
  typedef CartesianGrid::Basic::BackGhostGrid<CellAccessor>
    BackGhostGrid;
  typedef CartesianGrid::Basic::FrontGhostGrid<CellAccessor>
    FrontGhostGrid;

public:
  Grid(Descriptor& descriptor)
    : iteratorFactory(
        std::bind(&Grid::createCellAccessor, this,
                  std::placeholders::_1,
                  std::placeholders::_2,
                  std::placeholders::_3,
                  std::placeholders::_4,
                  std::placeholders::_5)),
      innerGrid(this, iteratorFactory),
      leftGhostGrid(this, iteratorFactory),
      rightGhostGrid(this, iteratorFactory),
      bottomGhostGrid(this, iteratorFactory),
      topGhostGrid(this, iteratorFactory),
      backGhostGrid(this, iteratorFactory),
      frontGhostGrid(this, iteratorFactory),
      _descriptor(descriptor),
      _cells(DataHeap::getInstance().getData(
               _descriptor.getUnknowns(!_descriptor.getNewerTimeStep()))),
      _newCells(DataHeap::getInstance().getData(
                  _descriptor.getUnknowns(_descriptor.getNewerTimeStep()))) {}

  Grid(Grid const& other) = delete;

  Grid&
  operator=(Grid const& other) = delete;

  ~Grid() {}

  int
  xSize() const {
    return _descriptor.getCells(0) + 2 * xGhostSize();
  }

  int
  ySize() const {
    return _descriptor.getCells(1) + 2 * yGhostSize();
  }

  int
  zSize() const {
    return _descriptor.getCells(2) + 2 * zGhostSize();
  }

  int
  xGhostSize() const {
    return _descriptor.getHaloCells(0);
  }

  int
  yGhostSize() const {
    return _descriptor.getHaloCells(1);
  }

  int
  zGhostSize() const {
    return _descriptor.getHaloCells(2);
  }

  double
  previousTimeStamp() const {
    return _descriptor.getTimeStamp(_descriptor.getNewerTimeStep());
  }

  double
  currentTimeStamp() const {
    return _descriptor.getTimeStamp(!_descriptor.getNewerTimeStep());
  }

  void
  newTimeStamp(double const& value) {
    _descriptor.setTimeStamp(_descriptor.getNewerTimeStep(), value);
  }

  double
  timeStepSize() const {
    return _descriptor.getTimeStepSize();
  }

  void
  timeStepSize(double const& value) {
    _descriptor.setTimeStepSize(value);
  }

  void
  flip() {
    _descriptor.setNewerTimeStep(!_descriptor.getNewerTimeStep());
  }

  void
  resetTimeStep() {
    _descriptor.setNewerTimeStep(0);
  }

  Cell
  leftCell(int const& index) const {
    return Cell(_cells[leftIndex(index)]);
  }

  Cell
  rightCell(int const& index) const {
    return Cell(_cells[rightIndex(index)]);
  }

  Cell
  bottomCell(int const& index) const {
    return Cell(_cells[bottomIndex(index)]);
  }

  Cell
  topCell(int const& index) const {
    return Cell(_cells[topIndex(index)]);
  }

  Cell
  backCell(int const& index) const {
    return Cell(_cells[backIndex(index)]);
  }

  Cell
  frontCell(int const& index) const {
    return Cell(_cells[frontIndex(index)]);
  }

  Cell
  currentCell(int const& index) const {
    return Cell(_cells[currentIndex(index)]);
  }

  Cell
  leftNewCell(int const& index) const {
    return Cell(_newCells[leftIndex(index)]);
  }

  Cell
  rightNewCell(int const& index) const {
    return Cell(_newCells[rightIndex(index)]);
  }

  Cell
  bottomNewCell(int const& index) const {
    return Cell(_newCells[bottomIndex(index)]);
  }

  Cell
  topNewCell(int const& index) const {
    return Cell(_newCells[topIndex(index)]);
  }

  Cell
  backNewCell(int const& index) const {
    return Cell(_newCells[backIndex(index)]);
  }

  Cell
  frontNewCell(int const& index) const {
    return Cell(_newCells[frontIndex(index)]);
  }

  Cell
  currentNewCell(int const& index) const {
    return Cell(_newCells[currentIndex(index)]);
  }

public:
  CellAccessor
  createCellAccessor(CartesianGrid::Subgrid const* subgrid,
                     int const&                    x,
                     int const&                    y,
                     int const&                    z,
                     int const&                    index) {
    return CellAccessor(subgrid, x, y, z, index, this);
  }

  IteratorFactory iteratorFactory;

  InnerGrid       innerGrid;
  LeftGhostGrid   leftGhostGrid;
  RightGhostGrid  rightGhostGrid;
  BottomGhostGrid bottomGhostGrid;
  TopGhostGrid    topGhostGrid;
  BackGhostGrid   backGhostGrid;
  FrontGhostGrid  frontGhostGrid;

private:
  Descriptor&      _descriptor;
  PersistentCells& _cells;
  PersistentCells& _newCells;
};

#endif //PEANOCLAW_SOLVER_EULER3D_GRID_H
