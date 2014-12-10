/*
 * SubgridAccessor.cpp
 *
 *  Created on: May 9, 2014
 *      Author: kristof
 */
#include "peanoclaw/grid/SubgridAccessor.h"

#include "peano/utils/Loop.h"

double* peanoclaw::grid::SubgridAccessor::getUNewArray() const {
  assertion(_u != 0);
  return reinterpret_cast<double*>(&(_u->at(0)));
}

double* peanoclaw::grid::SubgridAccessor::getUOldWithGhostLayerArray(int unknown) const {
  int index = _linearization.getUOldWithGhostlayerArrayIndex()
      + tarch::la::volume(_subdivisionFactor + 2*_ghostlayerWidth) * unknown;
  return reinterpret_cast<double*>(&(_u->at(index)));
}

double* peanoclaw::grid::SubgridAccessor::getParameterWithoutGhostLayerArray(int parameter) const {
  int index = _linearization.getParameterWithoutGhostlayerArrayIndex()
      + tarch::la::volume(_subdivisionFactor) * parameter;
  return reinterpret_cast<double*>(&(_u->at(index)));
}

double* peanoclaw::grid::SubgridAccessor::getParameterWithGhostLayerArray(int parameter) const {
  int index = _linearization.getParameterWithGhostlayerArrayIndex()
      + tarch::la::volume(_subdivisionFactor + 2*_ghostlayerWidth) * parameter;
  return reinterpret_cast<double*>(&(_u->at(index)));
}

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
  assertion4(index < _linearization.getParameterWithoutGhostlayerArrayIndex() - _linearization.getUOldWithGhostlayerArrayIndex(), index, subcellIndex, unknown, _linearization.getParameterWithoutGhostlayerArrayIndex() - _linearization.getUOldWithGhostlayerArrayIndex());
  #ifdef PATCH_RANGE_CHECK
  _u->at(_linearization.getUOldWithGhostlayerArrayIndex() + index).setU(value);
  #else
  (*_u)[_linearization.getUOldWithGhostlayerArrayIndex() + index].setU(value);
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
  assertion4(index < _linearization.getParameterWithoutGhostlayerArrayIndex() - _linearization.getUOldWithGhostlayerArrayIndex(), index, subcellIndex, unknown, _linearization.getParameterWithoutGhostlayerArrayIndex() - _linearization.getUOldWithGhostlayerArrayIndex());
#ifdef PATCH_DISABLE_RANGE_CHECK
  return (*_u)[_linearization.getUOldWithGhostlayerArrayIndex() + index].getU();
#else
  return _u->at(_linearization.getUOldWithGhostlayerArrayIndex() + index).getU();
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
  return (*_u)[_linearization.getUOldWithGhostlayerArrayIndex() + index].getU();
#else
  return _u->at(_linearization.getUOldWithGhostlayerArrayIndex() + index).getU();
#endif
}

void peanoclaw::grid::SubgridAccessor::setValueUOld(int linearIndex, int unknown, double value) {
  int index = linearIndex + _linearization.getQStrideUOld() * unknown;
#ifdef PATCH_DISABLE_RANGE_CHECK
  (*_u)[_linearization.getUOldWithGhostlayerArrayIndex() + index].setU(value);
#else
  _u->at(_linearization.getUOldWithGhostlayerArrayIndex() + index).setU(value);
#endif
}

void peanoclaw::grid::SubgridAccessor::setValueUOldAndResize(int linearIndex, int unknown, double value) {
  size_t index = linearIndex + _linearization.getQStrideUOld() * unknown;
  if(_linearization.getUOldWithGhostlayerArrayIndex() + index + 1 > _u->size()) {
    _u->resize(_linearization.getUOldWithGhostlayerArrayIndex() + index + 1);
  }
  _u->at(_linearization.getUOldWithGhostlayerArrayIndex() + index) = value;
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
  assertion5(_linearization.getParameterWithoutGhostlayerArrayIndex()+index < static_cast<int>(_u->size()), _linearization.getParameterWithoutGhostlayerArrayIndex(), index, subcellIndex,
        parameter, static_cast<int>(_u->size()));
  return _u->at(_linearization.getParameterWithoutGhostlayerArrayIndex() + index).getU();
}

void peanoclaw::grid::SubgridAccessor::setParameterWithoutGhostlayer(
  const tarch::la::Vector<DIMENSIONS, int>& subcellIndex,
  int parameter,
  double value
) {
  assertion(_isLeaf || _isVirtual);
  int index = _linearization.linearizeParameterWithGhostlayer(parameter, subcellIndex);
  assertion3(index >= 0, index, subcellIndex, parameter);
  assertion5(_linearization.getParameterWithoutGhostlayerArrayIndex()+index < static_cast<int>(_u->size()), _linearization.getParameterWithoutGhostlayerArrayIndex(), index, subcellIndex,
        parameter, static_cast<int>(_u->size()));
  _u->at(_linearization.getParameterWithoutGhostlayerArrayIndex() + index).setU(value);
}

double peanoclaw::grid::SubgridAccessor::getParameterWithGhostlayer(
  const tarch::la::Vector<DIMENSIONS, int>& subcellIndex,
  int parameter
) const {
  assertion(_isLeaf || _isVirtual);
  int index = _linearization.linearizeParameterWithGhostlayer(parameter, subcellIndex);
  assertion3(index >= 0, index, subcellIndex, parameter);
  assertion5(_linearization.getParameterWithGhostlayerArrayIndex()+index < static_cast<int>(_u->size()), _linearization.getParameterWithGhostlayerArrayIndex(), index, subcellIndex,
        parameter, static_cast<int>(_u->size()));
  return _u->at(_linearization.getParameterWithGhostlayerArrayIndex() + index).getU();
}

void peanoclaw::grid::SubgridAccessor::setParameterWithGhostlayer(
  const tarch::la::Vector<DIMENSIONS, int>& subcellIndex,
  int parameter,
  double value
) {
  assertion(_isLeaf || _isVirtual);
  int index = _linearization.linearizeParameterWithGhostlayer(parameter, subcellIndex);
  assertion3(index >= 0, index, subcellIndex, parameter);
  assertion5(_linearization.getParameterWithGhostlayerArrayIndex()+index < static_cast<int>(_u->size()), _linearization.getParameterWithGhostlayerArrayIndex(), index, subcellIndex,
        parameter, static_cast<int>(_u->size()));
  _u->at(_linearization.getParameterWithGhostlayerArrayIndex() + index).setU(value);
}

double peanoclaw::grid::SubgridAccessor::getFlux(
  const tarch::la::Vector<DIMENSIONS_MINUS_ONE,int>& subcellIndex,
  int unknown,
  int dimension,
  int direction
) const {
  return _u->at(_linearization.linearizeFlux(unknown, subcellIndex, dimension, direction)).getU();
}

void peanoclaw::grid::SubgridAccessor::setFlux(
  const tarch::la::Vector<DIMENSIONS_MINUS_ONE,int>& subcellIndex,
  int unknown,
  int dimension,
  int direction,
  double value
) {
  _u->at(_linearization.linearizeFlux(unknown, subcellIndex, dimension, direction)).setU(value);
}


void peanoclaw::grid::SubgridAccessor::clearRegion(
  const peanoclaw::geometry::Region& region,
  bool clearUOld
) {
#if defined(Dim2) && false
//  for(int x = 0; x < size(0); x++) {
//    memset(_uOldWithGhostlayer[])
//  }
#else
  int unknownsPerSubcell = _cellDescription.getUnknownsPerSubcell();
  dfor(subcellIndex, region._size){
  int linearIndex = clearUOld ? getLinearIndexUOld(subcellIndex + region._offset) : getLinearIndexUNew(subcellIndex + region._offset);
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

bool peanoclaw::grid::SubgridAccessor::isInitialized() const {
  return _isInitialized;
}
