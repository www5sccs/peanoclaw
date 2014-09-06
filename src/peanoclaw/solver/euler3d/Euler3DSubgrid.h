#ifndef PEANOCLAW_SOLVER_EULER3D_EULER3DSUBGRID_H
#define PEANOCLAW_SOLVER_EULER3D_EULER3DSUBGRID_H

#include "solver/Grid.hpp"
#include "NumericsMethods/TwoStepMethods/Grid.hpp"
#include "CartesianGrid/Basic/Grid.hpp"
#include "peanoclaw/solver/euler3d/Cell.h"

namespace peanoclaw {
  namespace solver {
    namespace euler3d {
      class Euler3DSubgrid;
    }
  }
}

class peanoclaw::solver::euler3d::Euler3DSubgrid
    : public NumericsMethods::TwoStepMethods::Grid<peanoclaw::solver::euler3d::Cell, double>,
      public CartesianGrid::Basic::Grid {

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
};

#endif //PEANOCLAW_SOLVER_EULER3D_EULER3DSUBGRID_H
