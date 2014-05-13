#include "peanoclaw/Cell.h"

#include "peanoclaw/Patch.h"


peanoclaw::Cell::Cell():
  Base(), _subgrid(0) {
  _cellData.setCellDescriptionIndex(-2);

  #ifdef Parallel
  setCellIsAForkCandidate(false);
  #endif
}


peanoclaw::Cell::Cell(const Base::DoNotCallStandardConstructor& value):
  Base(value), _subgrid(0) {
  // Please do not insert anything here
}

peanoclaw::Cell::Cell(const Base::PersistentCell& argument):
  Base(argument), _subgrid(0) {
  // @todo Insert your code here
}

void peanoclaw::Cell::setCellDescriptionIndex(int index) {
  _cellData.setCellDescriptionIndex(index);
}

int peanoclaw::Cell::getCellDescriptionIndex() const {
  return _cellData.getCellDescriptionIndex();
}

bool peanoclaw::Cell::holdsSubgrid() const {
  return _cellData.getCellDescriptionIndex() != -1;
}

void peanoclaw::Cell::setSubgrid(peanoclaw::Patch& subgrid) {
  _subgrid = &subgrid;
}

peanoclaw::Patch& peanoclaw::Cell::getSubgrid() const {
  return *_subgrid;
}
