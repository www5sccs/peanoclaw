#include "peanoclaw/Cell.h"


peanoclaw::Cell::Cell():
  Base() {
  _cellData.setCellDescriptionIndex(-2);
}


peanoclaw::Cell::Cell(const Base::DoNotCallStandardConstructor& value):
  Base(value) {
  // Please do not insert anything here
}

peanoclaw::Cell::Cell(const Base::PersistentCell& argument):
  Base(argument) {
  // @todo Insert your code here
}

void peanoclaw::Cell::setCellDescriptionIndex(int index) {
  _cellData.setCellDescriptionIndex(index);
}

int peanoclaw::Cell::getCellDescriptionIndex() const {
  return _cellData.getCellDescriptionIndex();
}
