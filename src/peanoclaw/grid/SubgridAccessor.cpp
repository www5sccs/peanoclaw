/*
 * SubgridAccessor.cpp
 *
 *  Created on: May 9, 2014
 *      Author: kristof
 */
#include "peanoclaw/grid/SubgridAccessor.h"

#include "peano/utils/Loop.h"

void peanoclaw::grid::SubgridAccessor::setValueUNew(tarch::la::Vector<DIMENSIONS, int> subcellIndex, int unknown, double value) {
  assertion(_isLeaf || _isVirtual);
  int index = _linearization.linearize(unknown, subcellIndex);
  assertion3(index >= 0, index, subcellIndex, unknown);
  assertion4(index < static_cast<int>(_u->size()), index, subcellIndex, unknown, static_cast<int>(_u->size()));
#ifdef PATCH_DISABLE_RANGE_CHECK
  (*_u)[index].setU(value);
#else
  _u->at(index).setU(value);
#endif
}

void peanoclaw::grid::SubgridAccessor::setValueUOld(tarch::la::Vector<DIMENSIONS, int> subcellIndex, int unknown, double value) {
  assertion(_isLeaf || _isVirtual);
  int index = _linearization.linearizeWithGhostlayer(unknown, subcellIndex);
  assertion3(index >= 0, index, subcellIndex, unknown);
  assertion4(index < _parameterWithoutGhostlayerArrayIndex - _uOldWithGhostlayerArrayIndex, index, subcellIndex, unknown, _parameterWithoutGhostlayerArrayIndex - _uOldWithGhostlayerArrayIndex);
  #ifdef PATCH_RANGE_CHECK
  _u->at(_uOldWithGhostlayerArrayIndex + index).setU(value);
  #else
  (*_u)[_uOldWithGhostlayerArrayIndex + index].setU(value);
  #endif
}

double peanoclaw::grid::SubgridAccessor::getValueUNew(tarch::la::Vector<DIMENSIONS, int> subcellIndex, int unknown) const {
  assertion(_isLeaf || _isVirtual);
  int index = _linearization.linearize(unknown, subcellIndex);
  assertion3(index >= 0, index, subcellIndex, unknown);
  assertion3(index < static_cast<int>(_u->size()), index, subcellIndex, unknown);
#ifdef PATCH_DISABLE_RANGE_CHECK
  return (*_u)[index].getU();
#else
  return _u->at(index).getU();
#endif
}

double peanoclaw::grid::SubgridAccessor::getValueUOld(tarch::la::Vector<DIMENSIONS, int> subcellIndex, int unknown) const {
  assertion(_isLeaf || _isVirtual);
  int index = _linearization.linearizeWithGhostlayer(unknown, subcellIndex);
  assertion3(index >= 0, index, subcellIndex, unknown);
  assertion4(index < _parameterWithoutGhostlayerArrayIndex - _uOldWithGhostlayerArrayIndex, index, subcellIndex, unknown, _parameterWithoutGhostlayerArrayIndex - _uOldWithGhostlayerArrayIndex);
#ifdef PATCH_DISABLE_RANGE_CHECK
  return (*_u)[_uOldWithGhostlayerArrayIndex + index].getU();
#else
  return _u->at(_uOldWithGhostlayerArrayIndex + index).getU();
#endif
}

int peanoclaw::grid::SubgridAccessor::getLinearIndexUNew(tarch::la::Vector<DIMENSIONS, int> subcellIndex) const {
  return _linearization.linearize(0, subcellIndex);
}

int peanoclaw::grid::SubgridAccessor::getLinearIndexUOld(tarch::la::Vector<DIMENSIONS, int> subcellIndex) const {
  return _linearization.linearizeWithGhostlayer(0, subcellIndex);
}

double peanoclaw::grid::SubgridAccessor::getValueUNew(int linearIndex, int unknown) const {
  int index = linearIndex + _linearization.getQStrideUNew() * unknown;
#ifdef PATCH_DISABLE_RANGE_CHECK
  return (*_u)[index].getU();
#else
  return _u->at(index).getU();
#endif
}

void peanoclaw::grid::SubgridAccessor::setValueUNew(int linearIndex, int unknown, double value) {
  int index = linearIndex + _linearization.getQStrideUNew() * unknown;
#ifdef PATCH_DISABLE_RANGE_CHECK
  (*_u)[index].setU(value);
#else
  _u->at(index).setU(value);
#endif
}

void peanoclaw::grid::SubgridAccessor::setValueUNewAndResize(int linearIndex, int unknown, double value) {
  size_t index = linearIndex + _linearization.getQStrideUNew() * unknown;
  if(index + 1 > _u->size()) {
    _u->resize(index + 1);
  }
  _u->at(index) = value;
}

double peanoclaw::grid::SubgridAccessor::getValueUOld(int linearIndex, int unknown) const {
  int index = linearIndex + _linearization.getQStrideUOld() * unknown;
#ifdef PATCH_DISABLE_RANGE_CHECK
  return (*_u)[_uOldWithGhostlayerArrayIndex + index].getU();
#else
  return _u->at(_uOldWithGhostlayerArrayIndex + index).getU();
#endif
}

void peanoclaw::grid::SubgridAccessor::setValueUOld(int linearIndex, int unknown, double value) {
  int index = linearIndex + _linearization.getQStrideUOld() * unknown;
#ifdef PATCH_DISABLE_RANGE_CHECK
  (*_u)[_uOldWithGhostlayerArrayIndex + index].setU(value);
#else
  _u->at(_uOldWithGhostlayerArrayIndex + index).setU(value);
#endif
}

void peanoclaw::grid::SubgridAccessor::setValueUOldAndResize(int linearIndex, int unknown, double value) {
  size_t index = linearIndex + _linearization.getQStrideUOld() * unknown;
  if(_uOldWithGhostlayerArrayIndex + index + 1 > _u->size()) {
    _u->resize(_uOldWithGhostlayerArrayIndex + index + 1);
  }
  _u->at(_uOldWithGhostlayerArrayIndex + index) = value;
}

double peanoclaw::grid::SubgridAccessor::getParameterWithoutGhostlayer(
  const tarch::la::Vector<DIMENSIONS, int>& subcellIndex,
  int parameter
) const {
  assertion(_isLeaf || _isVirtual);
  int index = _linearization.linearizeParameterWithoutGhostlayer(parameter, subcellIndex);

  //TODO unterweg debug
//  std::cout << "index=" << index << " stride=" << _linearization.getQStrideParameterWithoutGhostlayer()
//      << ", " << _linearization.getCellStrideParameterWithoutGhostlayer(0) << ","
//      << _linearization.getCellStrideParameterWithoutGhostlayer(1) << std::endl;

  assertion3(index >= 0, index, subcellIndex, parameter);
  assertion5(_parameterWithoutGhostlayerArrayIndex+index < static_cast<int>(_u->size()), _parameterWithoutGhostlayerArrayIndex, index, subcellIndex,
        parameter, static_cast<int>(_u->size()));
  return _u->at(_parameterWithoutGhostlayerArrayIndex + index).getU();
}

void peanoclaw::grid::SubgridAccessor::setParameterWithoutGhostlayer(
  const tarch::la::Vector<DIMENSIONS, int>& subcellIndex,
  int parameter,
  double value
) {
  assertion(_isLeaf || _isVirtual);
  int index = _linearization.linearizeParameterWithGhostlayer(parameter, subcellIndex);
  assertion3(index >= 0, index, subcellIndex, parameter);
  assertion5(_parameterWithoutGhostlayerArrayIndex+index < static_cast<int>(_u->size()), _parameterWithoutGhostlayerArrayIndex, index, subcellIndex,
        parameter, static_cast<int>(_u->size()));
  _u->at(_parameterWithoutGhostlayerArrayIndex + index).setU(value);
}

double peanoclaw::grid::SubgridAccessor::getParameterWithGhostlayer(
  const tarch::la::Vector<DIMENSIONS, int>& subcellIndex,
  int parameter
) const {
  assertion(_isLeaf || _isVirtual);
  int index = _linearization.linearizeParameterWithGhostlayer(parameter, subcellIndex);
  assertion3(index >= 0, index, subcellIndex, parameter);
  assertion5(_parameterWithGhostlayerArrayIndex+index < static_cast<int>(_u->size()), _parameterWithGhostlayerArrayIndex, index, subcellIndex,
        parameter, static_cast<int>(_u->size()));
  return _u->at(_parameterWithGhostlayerArrayIndex + index).getU();
}

void peanoclaw::grid::SubgridAccessor::setParameterWithGhostlayer(
  const tarch::la::Vector<DIMENSIONS, int>& subcellIndex,
  int parameter,
  double value
) {
  assertion(_isLeaf || _isVirtual);
  int index = _linearization.linearizeParameterWithGhostlayer(parameter, subcellIndex);
  assertion3(index >= 0, index, subcellIndex, parameter);
  assertion5(_parameterWithGhostlayerArrayIndex+index < static_cast<int>(_u->size()), _parameterWithGhostlayerArrayIndex, index, subcellIndex,
        parameter, static_cast<int>(_u->size()));
  _u->at(_parameterWithGhostlayerArrayIndex + index).setU(value);
}

void peanoclaw::grid::SubgridAccessor::clearRegion(tarch::la::Vector<DIMENSIONS, int> offset,
    tarch::la::Vector<DIMENSIONS, int> size, bool clearUOld) {
#if defined(Dim2) && false
//  for(int x = 0; x < size(0); x++) {
//    memset(_uOldWithGhostlayer[])
//  }
#else
  int unknownsPerSubcell = _cellDescription.getUnknownsPerSubcell();
  dfor(subcellIndex, size){
  int linearIndex = clearUOld ? getLinearIndexUOld(subcellIndex + offset) : getLinearIndexUNew(subcellIndex + offset);
  for(int unknown = 0; unknown < unknownsPerSubcell; unknown++) {
    if(clearUOld) {
      setValueUOld(linearIndex, unknown, 0.0);
    } else {
      setValueUNew(linearIndex, unknown, 0.0);
    }
  }
}
#endif
}

