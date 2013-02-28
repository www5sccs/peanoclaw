/*
 * Patch.cpp
 *
 *      Author: Kristof Unterweger
 */
#include "Patch.h"

#include <sstream>
#include <iomanip>
#include <limits>

#include "Area.h"
#include "PatchOperations.h"
#include "Cell.h"
#include "peano/heap/Heap.h"
#include "peano/utils/Loop.h"

int peanoclaw::Patch::counter = 0;

tarch::logging::Log peanoclaw::Patch::_log("peanoclaw::Patch");

std::vector<peanoclaw::Patch::Data> peanoclaw::Patch::_nullData;

int peanoclaw::Patch::linearize(
  int unknown,
  const tarch::la::Vector<DIMENSIONS, int>& subcellIndex
) const {
  int index = 0;
//  int stride = 1;
//  for(int d = DIMENSIONS-1; d >= 0; d--) {
  for(int d = 0; d < DIMENSIONS; d++) {
//    assertion3(subcellIndex(d) >= 0 && subcellIndex(d) < _cellDescription->getSubdivisionFactor()(d),
//        subcellIndex(d),
//        _cellDescription->getSubdivisionFactor(),
//        toString());
//    index += subcellIndex(d) * stride;
//    stride *= _cellDescription->getSubdivisionFactor()(d);
    index += subcellIndex(d) * uNewStrideCache[d+1];
  }
//  index += unknown * stride;
  index += unknown * uNewStrideCache[0];

  return index;
}

#ifndef PATCH_INLINE_GETTERS_AND_SETTERS
int peanoclaw::Patch::linearizeWithGhostlayer(
  int unknown,
  const tarch::la::Vector<DIMENSIONS, int>& subcellIndex
) const {
  int index = 0;
  int ghostlayerWidth = _cellDescription->getGhostLayerWidth();
  tarch::la::Vector<DIMENSIONS, double> subdivisionFactor = _cellDescription->getSubdivisionFactor();

  for(int d = 0; d < DIMENSIONS; d++) {
//  for(int d = DIMENSIONS-1; d >= 0; d--) {
//    assertion4(subcellIndex(d) >= - ghostlayerWidth && subcellIndex(d) < subdivisionFactor(d) + ghostlayerWidth,
//      subcellIndex(d),
//      subdivisionFactor,
//      ghostlayerWidth,
//      toString()
//    );
//    index += (subcellIndex(d) + ghostlayerWidth) * stride;
//    stride *= (subdivisionFactor(d) + 2*ghostlayerWidth);
    index += (subcellIndex(d) + ghostlayerWidth) * uOldStrideCache[d+1];
  }
  index += unknown * uOldStrideCache[0];
//  index += unknown * stride;
  return index;
}
#endif

void peanoclaw::Patch::fillCaches() {
  int ghostlayerWidth = _cellDescription->getGhostLayerWidth();
  tarch::la::Vector<DIMENSIONS, double> subdivisionFactor = _cellDescription->getSubdivisionFactor();

  //UOld
  int stride = 1;
  for(int d = DIMENSIONS; d > 0; d--) {
    uOldStrideCache[d] = stride;
    stride *= subdivisionFactor(d-1) + 2*ghostlayerWidth;
  }
  uOldStrideCache[0] = stride;

  //UNew
  stride = 1;
  for(int d = DIMENSIONS; d > 0; d--) {
      uNewStrideCache[d] = stride;
      stride *= subdivisionFactor(d-1);
  }
  uNewStrideCache[0] = stride;

  //Precompute subcell size
  _subcellSize = _cellDescription->getSize() / _cellDescription->getSubdivisionFactor();
}

void peanoclaw::Patch::switchAreaToMinimalFineGridTimeInterval(
    const Area& area,
    double factorForUOld,
    double factorForUNew
) {
  //TODO unterweg debug
//  std::cout << "Switched cells:" << tarch::la::volume(area._size) << std::endl;

  dfor(subcellIndex, area._size) {
    int linearIndexUOld = getLinearIndexUOld(subcellIndex + area._offset);
    int linearIndexUNew = getLinearIndexUNew(subcellIndex + area._offset);

    for(int unknown = 0; unknown < _cellDescription->getUnknownsPerSubcell(); unknown++) {
      double valueUOld = getValueUOld(linearIndexUOld, unknown);
      double valueUNew = getValueUNew(linearIndexUNew, unknown);

      #ifdef Asserts
      if(unknown == 0) {
        assertion9(tarch::la::greaterEquals(valueUOld * (1.0 - factorForUOld) + valueUNew * factorForUOld, 0.0)
        && tarch::la::greaterEquals(valueUOld * (1.0 - factorForUNew) + valueUNew * factorForUNew, 0.0),
        toString(),
        getCurrentTime(),
        getTimestepSize(),
        factorForUOld,
        factorForUNew,
        valueUOld,
        valueUNew,
        valueUOld * (1.0 - factorForUOld) + valueUNew * factorForUOld,
        valueUOld * (1.0 - factorForUNew) + valueUNew * factorForUNew);
      }
      #endif

      //UOld
      setValueUOld(linearIndexUOld, unknown, valueUOld * (1.0 - factorForUOld) + valueUNew * factorForUOld);

      //UNew
      setValueUNew(linearIndexUNew, unknown, valueUOld * (1.0 - factorForUNew) + valueUNew * factorForUNew);
    }
  }
}

peanoclaw::Patch::Patch()
: _cellDescription(0),
  _uNew(0),
  _uOldWithGhostlayer(0),
  _auxArray(0)
{
}

peanoclaw::Patch::Patch(
  const tarch::la::Vector<DIMENSIONS, double>& position,
  const tarch::la::Vector<DIMENSIONS, double>& size,
  const Cell& cell
) : _cellDescription(0)
{
  _cellDescription = &peano::heap::Heap<CellDescription>::getInstance().getData(cell.getCellDescriptionIndex()).at(0);
  _cellDescription->setPosition(position);
  _cellDescription->setSize(size);

  assertion1(tarch::la::allGreater(getSize(), tarch::la::Vector<DIMENSIONS,double>(0.0)), toString());

  //Retrieve patch data
  assertion2((_cellDescription->getUNewIndex() == -1 && _cellDescription->getUOldIndex() == -1) || (_cellDescription->getUNewIndex() != -1 && _cellDescription->getUOldIndex() != -1) || (_cellDescription->getUNewIndex() != -1  && _cellDescription->getUOldIndex() == -1), _cellDescription->getUNewIndex(), _cellDescription->getUOldIndex());
  if(_cellDescription->getUNewIndex() != -1) {
      _uNew = &peano::heap::Heap<Data>::getInstance().getData(_cellDescription->getUNewIndex());
    } else {
      _uNew = 0;
    }

  if(_cellDescription->getUOldIndex() != -1) {
    _uOldWithGhostlayer = &peano::heap::Heap<Data>::getInstance().getData(_cellDescription->getUOldIndex());
  } else {
    _uOldWithGhostlayer = 0;
  }

  if(_cellDescription->getAuxIndex() != -1) {
    _auxArray = &peano::heap::Heap<Data>::getInstance().getData(_cellDescription->getAuxIndex());
  } else {
    _auxArray = 0;
  }

  fillCaches();
}

peanoclaw::Patch::Patch(
    CellDescription& cellDescription
) : _cellDescription(&cellDescription)
   {
  logTraceInWith2Arguments("Patch(CellDescription,int)", cellDescription.getPosition(), cellDescription.getLevel());

  assertion1(tarch::la::allGreater(getSize(), tarch::la::Vector<DIMENSIONS,double>(0.0)), toString());

  //Retrieve patch data
  assertion2((cellDescription.getUNewIndex() == -1 && cellDescription.getUOldIndex() == -1) || (cellDescription.getUNewIndex() != -1 && cellDescription.getUOldIndex() != -1) || (cellDescription.getUNewIndex() != -1  && cellDescription.getUOldIndex() == -1), cellDescription.getUNewIndex(), cellDescription.getUOldIndex());
  if(cellDescription.getUNewIndex() != -1) {
    _uNew = &peano::heap::Heap<Data>::getInstance().getData(cellDescription.getUNewIndex());
  } else {
    _uNew = 0;
  }

  if(cellDescription.getUOldIndex() != -1) {
    _uOldWithGhostlayer = &peano::heap::Heap<Data>::getInstance().getData(cellDescription.getUOldIndex());
  } else {
    _uOldWithGhostlayer = 0;
  }

  if(cellDescription.getAuxIndex() != -1) {
    _auxArray = &peano::heap::Heap<Data>::getInstance().getData(cellDescription.getAuxIndex());
  } else {
    _auxArray = 0;
  }

  fillCaches();

  logTraceOut("Patch(CellDescription,int)");
}

peanoclaw::Patch::~Patch() {
}

const tarch::la::Vector<DIMENSIONS, double> peanoclaw::Patch::getSize() const {
  return _cellDescription->getSize();
}

const tarch::la::Vector<DIMENSIONS, double> peanoclaw::Patch::getPosition() const {
  return _cellDescription->getPosition();
}

int peanoclaw::Patch::getUnknownsPerSubcell() const {
  return _cellDescription->getUnknownsPerSubcell();
}

int peanoclaw::Patch::getAuxiliarFieldsPerSubcell() const {
  return _cellDescription->getAuxiliarFieldsPerSubcell();
}

tarch::la::Vector<DIMENSIONS, int> peanoclaw::Patch::getSubdivisionFactor() const {
  return _cellDescription->getSubdivisionFactor();
}

int peanoclaw::Patch::getGhostLayerWidth() const {
  return _cellDescription->getGhostLayerWidth();
}

double peanoclaw::Patch::getCurrentTime() const {
  return _cellDescription->getTime();
}

void peanoclaw::Patch::setCurrentTime(double currentTime) {
  _cellDescription->setTime(currentTime);
}

double peanoclaw::Patch::getTimeUOld() const {
  assertion(isLeaf() || isVirtual());
  if(isLeaf()) {
    return _cellDescription->getTime();
  } else {
    return _cellDescription->getMinimalNeighborTime();
  }
}

double peanoclaw::Patch::getTimeUNew() const {
  assertion(isLeaf() || isVirtual());
  if(isLeaf()) {
    return _cellDescription->getTime() + _cellDescription->getTimestepSize();
  } else {
    return _cellDescription->getMinimalNeighborTime() + _cellDescription->getMaximalNeighborTimestep();
  }
}

void peanoclaw::Patch::advanceInTime() {
  _cellDescription->setTime(_cellDescription->getTime() + _cellDescription->getTimestepSize());
}

double peanoclaw::Patch::getTimestepSize() const {
  return _cellDescription->getTimestepSize();
}

void peanoclaw::Patch::setTimestepSize(double timestepSize) {
  _cellDescription->setTimestepSize(timestepSize);
}

double peanoclaw::Patch::getEstimatedNextTimestepSize() const {
  return _cellDescription->getEstimatedNextTimestepSize();
}

void peanoclaw::Patch::setEstimatedNextTimestepSize(double estimatedNextTimestepSize) {
  _cellDescription->setEstimatedNextTimestepSize(estimatedNextTimestepSize);
}

double peanoclaw::Patch::getMinimalNeighborTimeConstraint() const {
  return _cellDescription->getMinimalNeighborTimeConstraint();
}

double peanoclaw::Patch::getMinimalLeafNeighborTimeConstraint() const {
  return _cellDescription->getMinimalLeafNeighborTimeConstraint();
}

void peanoclaw::Patch::updateMinimalNeighborTimeConstraint(double neighborTimeConstraint) {
  if(neighborTimeConstraint < _cellDescription->getMinimalNeighborTimeConstraint()) {
    _cellDescription->setMinimalNeighborTimeConstraint(neighborTimeConstraint);
  }
}

void peanoclaw::Patch::updateMinimalLeafNeighborTimeConstraint(double leafNeighborTime) {
  if(leafNeighborTime < _cellDescription->getMinimalLeafNeighborTimeConstraint()) {
    _cellDescription->setMinimalLeafNeighborTimeConstraint(leafNeighborTime);
  }
}

void peanoclaw::Patch::resetMinimalNeighborTimeConstraint() {
  _cellDescription->setMinimalNeighborTimeConstraint(std::numeric_limits<double>::max());
  _cellDescription->setMinimalLeafNeighborTimeConstraint(std::numeric_limits<double>::max());
}

void peanoclaw::Patch::resetMaximalNeighborTimeInterval() {
  _cellDescription->setMinimalNeighborTime(std::numeric_limits<double>::max());
  _cellDescription->setMaximalNeighborTimestep(-std::numeric_limits<double>::max());
}

void peanoclaw::Patch::updateMaximalNeighborTimeInterval(
  double neighborTime, double neighborTimestepSize
) {
  if(neighborTime + neighborTimestepSize > _cellDescription->getMinimalNeighborTime() + _cellDescription->getMaximalNeighborTimestep()) {
    _cellDescription->setMaximalNeighborTimestep(neighborTime + neighborTimestepSize - _cellDescription->getMinimalNeighborTime());
  }

  if(neighborTime < _cellDescription->getMinimalNeighborTime()) {
    _cellDescription->setMaximalNeighborTimestep(_cellDescription->getMaximalNeighborTimestep() + _cellDescription->getMinimalNeighborTime() - neighborTime);
    _cellDescription->setMinimalNeighborTime(neighborTime);
  }
}

//void peanoclaw::Patch::setTimeIntervalToMaximalNeighborTimeInterval() {
//  setCurrentTime(_cellDescription->getMinimalNeighborTime());
//  setTimestepSize(_cellDescription->getMaximalNeighborTimestep());
//}

bool peanoclaw::Patch::isAllowedToAdvanceInTime() const {
  return !tarch::la::greater(_cellDescription->getTime() + _cellDescription->getTimestepSize(), _cellDescription->getMinimalNeighborTimeConstraint());
}

int peanoclaw::Patch::getLevel() const {
  return _cellDescription->getLevel();
}

void peanoclaw::Patch::resetMinimalFineGridTimeInterval() {
  _cellDescription->setMaximumFineGridTime(-1.0);
  _cellDescription->setMinimumFineGridTimestep(std::numeric_limits<double>::max());
}

void peanoclaw::Patch::updateMinimalFineGridTimeInterval(
  double fineGridTime,
  double fineGridTimestepSize
) {
  if(fineGridTime > _cellDescription->getMaximumFineGridTime()) {
    _cellDescription->setMinimumFineGridTimestep(_cellDescription->getMinimumFineGridTimestep() - (fineGridTime - _cellDescription->getMaximumFineGridTime()));
    _cellDescription->setMaximumFineGridTime(fineGridTime);
  }
  _cellDescription->setMinimumFineGridTimestep(
    std::min(
      _cellDescription->getMinimumFineGridTimestep(),
      (fineGridTime + fineGridTimestepSize - _cellDescription->getMaximumFineGridTime())
    )
  );
}

double peanoclaw::Patch::getTimeConstraint() const {
//  if(!isLeaf()) {
//    return _cellDescription->getMaximumFineGridTime() + _cellDescription->getMinimumFineGridTimestep();
//  } else {
    return _cellDescription->getTime() + _cellDescription->getTimestepSize();
//  }
}

void peanoclaw::Patch::setFineGridsSynchronize(bool synchronizeFineGrids) {
  _cellDescription->setSynchronizeFineGrids(synchronizeFineGrids);
}

bool peanoclaw::Patch::shouldFineGridsSynchronize() const {
  return _cellDescription->getSynchronizeFineGrids();
}

void peanoclaw::Patch::setWillCoarsen(bool willCoarsen) {
  _cellDescription->setWillCoarsen(willCoarsen);
}

bool peanoclaw::Patch::willCoarsen() {
  return _cellDescription->getWillCoarsen();
}

void peanoclaw::Patch::switchValuesAndTimeIntervalToMinimalFineGridTimeInterval() {
  //TODO unterweg restricting to interval [0, 1]
  double time = 0.0; //_cellDescription->getTime();
  double timestepSize = 1.0; //_cellDescription->getTimestepSize();

  _cellDescription->setTime(_cellDescription->getMaximumFineGridTime());
  _cellDescription->setTimestepSize(_cellDescription->getMinimumFineGridTimestep());

  //Interpolate patch values from former [time, time+timestepSize] to new [time, time+timestepSize]
//  if(isVirtual() || willCoarsen()) {
//    double factorForUOld = 1.0;
//    double factorForUNew = 1.0;
//    if(!tarch::la::equals(timestepSize, 0.0)) {
//      factorForUOld = (_cellDescription->getTime() - time) / timestepSize;
//      factorForUNew = (_cellDescription->getTime() + _cellDescription->getTimestepSize() - time) / timestepSize;
//    }
//
//    if (willCoarsen()) {
//      Area area;
//      area._offset = tarch::la::Vector<DIMENSIONS, int>(0);
//      area._size = getSubdivisionFactor();
//      switchAreaToMinimalFineGridTimeInterval(area, factorForUOld, factorForUNew);
//    }
//    else if(isVirtual()) {
//      Area areas[DIMENSIONS_TIMES_TWO];
//      int numberOfAreas = peanoclaw::getAreasForRestriction(
//        getLowerNeighboringGhostlayerBounds(),
//        getUpperNeighboringGhostlayerBounds(),
//        getPosition(),
//        getSubcellSize(),
//        getSubcellSize(),
//        getSubdivisionFactor(),
//        areas
//      );
//
//      assertion1(numberOfAreas <= DIMENSIONS_TIMES_TWO, numberOfAreas);
//
//      for (int areaIndex = 0; areaIndex < numberOfAreas; areaIndex++) {
//        if(tarch::la::allGreater(areas[areaIndex]._size, tarch::la::Vector<DIMENSIONS, int>(0))) {
//          switchAreaToMinimalFineGridTimeInterval(areas[areaIndex], factorForUOld, factorForUNew);
//        }
//      }
//    }
//  }
}

void peanoclaw::Patch::setSkipNextGridIteration(bool skipNextGridIteration) {
  _cellDescription->setSkipNextGridIteration(skipNextGridIteration);
}

bool peanoclaw::Patch::shouldSkipNextGridIteration() {
  return _cellDescription->getSkipNextGridIteration();
}

#ifndef PATCH_INLINE_GETTERS_AND_SETTERS
void peanoclaw::Patch::setValueUNew(tarch::la::Vector<DIMENSIONS, int> subcellIndex, int unknown, double value) {
  assertion(isLeaf() || isVirtual());
  int index = linearize(unknown, subcellIndex);
  assertion3(index >= 0, index, subcellIndex, unknown);
  assertion4(index < static_cast<int>(_uNew->size()), index, subcellIndex, unknown, static_cast<int>(_uNew->size()));
  #ifdef PATCH_DISABLE_RANGE_CHECK
  (*_uNew)[index].setU(value);
  #else
  _uNew->at(index).setU(value);
  #endif
}
#endif

#ifndef PATCH_INLINE_GETTERS_AND_SETTERS
void peanoclaw::Patch::setValueUOld(tarch::la::Vector<DIMENSIONS, int> subcellIndex, int unknown, double value) {
  assertion(isLeaf() || isVirtual());
  int index = linearizeWithGhostlayer(unknown, subcellIndex);
  assertion3(index >= 0, index, subcellIndex, unknown);
  assertion5(index < static_cast<int>(_uOldWithGhostlayer->size()), index, subcellIndex, unknown, static_cast<int>(_uOldWithGhostlayer->size()), toString());
  #ifdef PATCH_DISABLE_RANGE_CHECK
  (*_uOldWithGhostlayer)[index].setU(value);
  #else
  _uOldWithGhostlayer->at(index).setU(value);
  #endif
}
#endif

#ifndef PATCH_INLINE_GETTERS_AND_SETTERS
double peanoclaw::Patch::getValueUNew(tarch::la::Vector<DIMENSIONS, int> subcellIndex, int unknown) const {
  assertion1(isLeaf() || isVirtual(), toString());
  int index = linearize(unknown, subcellIndex);
  assertion3(index >= 0, index, subcellIndex, unknown);
  assertion4(index < static_cast<int>(_uNew->size()), index, subcellIndex, unknown, toString());
  #ifdef PATCH_DISABLE_RANGE_CHECK
  return (*_uNew)[index].getU();
  #else
  return _uNew->at(index).getU();
  #endif
}
#endif

double peanoclaw::Patch::getValueUNew(
  tarch::la::Vector<DIMENSIONS, double> subcellPosition,
  int unknown
) const {
  tarch::la::Vector<DIMENSIONS, int> subcellIndex = (subcellPosition - getPosition()) / getSubcellSize();
  return getValueUNew(subcellIndex, unknown);
}

#ifndef PATCH_INLINE_GETTERS_AND_SETTERS
double peanoclaw::Patch::getValueUOld(tarch::la::Vector<DIMENSIONS, int> subcellIndex, int unknown) const {
  assertion1(isLeaf() || isVirtual(), toString());
  int index = linearizeWithGhostlayer(unknown, subcellIndex);
  assertion4(index >= 0, index, subcellIndex, unknown, toString());
  assertion5(index < static_cast<int>(_uOldWithGhostlayer->size()), index, subcellIndex, unknown, static_cast<int>(_uOldWithGhostlayer->size()), toString());
  #ifdef PATCH_DISABLE_RANGE_CHECK
  return (*_uOldWithGhostlayer)[index].getU();
  #else
  return _uOldWithGhostlayer->at(index).getU();
  #endif
}
#endif

#ifndef PATCH_INLINE_GETTERS_AND_SETTERS
int peanoclaw::Patch::getLinearIndexUNew(tarch::la::Vector<DIMENSIONS, int> subcellIndex) const {
  return linearize(0, subcellIndex);
}
#endif

#ifndef PATCH_INLINE_GETTERS_AND_SETTERS
int peanoclaw::Patch::getLinearIndexUOld(tarch::la::Vector<DIMENSIONS, int> subcellIndex) const {
  return linearizeWithGhostlayer(0, subcellIndex);
}
#endif

#ifndef PATCH_INLINE_GETTERS_AND_SETTERS
double peanoclaw::Patch::getValueUNew(int linearIndex, int unknown) const {
  int index = linearIndex + uNewStrideCache[0] * unknown;
  #ifdef PATCH_DISABLE_RANGE_CHECK
  return (*_uNew)[index].getU();
  #else
  return _uNew->at(index).getU();
  #endif
}
#endif

#ifndef PATCH_INLINE_GETTERS_AND_SETTERS
void peanoclaw::Patch::setValueUNew(int linearIndex, int unknown, double value) {
  int index = linearIndex + uNewStrideCache[0] * unknown;
  #ifdef PATCH_DISABLE_RANGE_CHECK
  (*_uNew)[index].setU(value);
  #else
  _uNew->at(index).setU(value);
  #endif
}
#endif

#ifndef PATCH_INLINE_GETTERS_AND_SETTERS
double peanoclaw::Patch::getValueUOld(int linearIndex, int unknown) const {
  int index = linearIndex + uOldStrideCache[0] * unknown;
  #ifdef PATCH_DISABLE_RANGE_CHECK
  return (*_uOldWithGhostlayer)[index].getU();
  #else
  return _uOldWithGhostlayer->at(index).getU();
  #endif
}
#endif

#ifndef PATCH_INLINE_GETTERS_AND_SETTERS
void peanoclaw::Patch::setValueUOld(int linearIndex, int unknown, double value) {
  int index = linearIndex + uOldStrideCache[0] * unknown;
  #ifdef PATCH_DISABLE_RANGE_CHECK
  (*_uOldWithGhostlayer)[index].setU(value);
  #else
  _uOldWithGhostlayer->at(index).setU(value);
  #endif
}
#endif

double peanoclaw::Patch::getValueAux(tarch::la::Vector<DIMENSIONS, int> subcellIndex, int auxField) const {
  assertion1(isLeaf() || isVirtual(), toString());
  int index = linearize(auxField, subcellIndex);
  assertion4(index >= 0, index, subcellIndex, auxField, toString());
  assertion5(index < static_cast<int>(_auxArray->size()), index, subcellIndex, auxField, static_cast<int>(_auxArray->size()), toString());
  return _auxArray->at(index).getU();
}

tarch::la::Vector<DIMENSIONS, double> peanoclaw::Patch::getSubcellCenter(tarch::la::Vector<DIMENSIONS, int> subcellIndex) const {
  tarch::la::Vector<DIMENSIONS, double> subcellSize = getSubcellSize();
  tarch::la::Vector<DIMENSIONS, double> result = tarch::la::multiplyComponents(subcellSize, subcellIndex);
  return _cellDescription->getPosition() + result + subcellSize * 0.5;
}

tarch::la::Vector<DIMENSIONS, double> peanoclaw::Patch::getSubcellPosition(tarch::la::Vector<DIMENSIONS, int> subcellIndex) const {
  tarch::la::Vector<DIMENSIONS, double> result = tarch::la::multiplyComponents(getSubcellSize(), subcellIndex);
  return _cellDescription->getPosition() + result;
}

tarch::la::Vector<DIMENSIONS, double> peanoclaw::Patch::getSubcellSize() const {
  return _subcellSize;
}

void peanoclaw::Patch::copyUNewToUOld() {
  assertion(isLeaf() || isVirtual());
  for(int unknown = 0; unknown < _cellDescription->getUnknownsPerSubcell(); unknown++) {
    dfor(subcellIndex, _cellDescription->getSubdivisionFactor()) {
      setValueUOld(subcellIndex, unknown, getValueUNew(subcellIndex, unknown));
    }
  }
}

void peanoclaw::Patch::clearRegion(
  tarch::la::Vector<DIMENSIONS, int> offset,
  tarch::la::Vector<DIMENSIONS, int> size,
  bool clearUOld
) {
  #if defined(Dim2) && false
//  for(int x = 0; x < size(0); x++) {
//    memset(_uOldWithGhostlayer[])
//  }
  #else
  int unknownsPerSubcell = getUnknownsPerSubcell();
  dfor(subcellIndex, size) {
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

double* peanoclaw::Patch::getUNewArray() const {
  return reinterpret_cast<double*>(&(_uNew->at(0)));
}

double* peanoclaw::Patch::getUOldWithGhostlayerArray() const {
  return reinterpret_cast<double*>(&(_uOldWithGhostlayer->at(0)));
}

double* peanoclaw::Patch::getAuxArray() const {
  return reinterpret_cast<double*>(&(_auxArray->at(0)));
}

int peanoclaw::Patch::getUNewIndex() const {
  return _cellDescription->getUNewIndex();
}

int peanoclaw::Patch::getUOldIndex() const {
  return _cellDescription->getUOldIndex();
}

int peanoclaw::Patch::getAuxIndex() const {
  return _cellDescription->getAuxIndex();
}

int peanoclaw::Patch::getCellDescriptionIndex() const {
  return _cellDescription->getCellDescriptionIndex();
}

void peanoclaw::Patch::setDemandedMeshWidth(double demandedMeshWidth) {
  _cellDescription->setDemandedMeshWidth(demandedMeshWidth);
}

double peanoclaw::Patch::getDemandedMeshWidth() const {
  return _cellDescription->getDemandedMeshWidth();
}

std::string peanoclaw::Patch::toStringUNew() const {
  if(isLeaf() || isVirtual()) {
    //Plot patch
    std::stringstream str;
    for(int y = getSubdivisionFactor()(1)-1; y>= 0; y--) {
      for(int x = 0; x < getSubdivisionFactor()(0); x++) {
        tarch::la::Vector<DIMENSIONS, int> subcellIndex;
        subcellIndex(0) = x;
        subcellIndex(1) = y;
        str << std::setw(2) << getValueUNew(subcellIndex, 0) << " ";
      }
      if(_cellDescription->getUnknownsPerSubcell() > 1) {
        str << "\t";
        for(int x = 0; x < getSubdivisionFactor()(0); x++) {
          tarch::la::Vector<DIMENSIONS, int> subcellIndex;
          subcellIndex(0) = x;
          subcellIndex(1) = y;
          str << std::setw(2) << getValueUNew(subcellIndex, 1) << "," << std::setw(2) << getValueUNew(subcellIndex, 2) << " ";
        }
      }
      str << std::endl;
    }
    str << std::endl;
    return str.str();
  } else {
    return "Is refined patch.\n";
  }
}

std::string peanoclaw::Patch::toStringUOldWithGhostLayer() const {
  if(isLeaf() || isVirtual()) {
    std::stringstream str;
    for(int y = getSubdivisionFactor()(1) + getGhostLayerWidth()-1; y>= -getGhostLayerWidth(); y--) {
      for(int x = -getGhostLayerWidth(); x < getSubdivisionFactor()(0) + getGhostLayerWidth(); x++) {
        tarch::la::Vector<DIMENSIONS, int> subcellIndex;
        subcellIndex(0) = x;
        subcellIndex(1) = y;
        str << std::setw(2) << getValueUOld(subcellIndex, 0) << " ";
      }
      if(_cellDescription->getUnknownsPerSubcell() > 1) {
        str << "\t";
        for(int x = -getGhostLayerWidth(); x < getSubdivisionFactor()(0) + getGhostLayerWidth(); x++) {
          tarch::la::Vector<DIMENSIONS, int> subcellIndex;
          subcellIndex(0) = x;
          subcellIndex(1) = y;
          str << std::setw(2) << getValueUOld(subcellIndex, 1) << "," << std::setw(2) << getValueUOld(subcellIndex, 2) << " ";
        }
      }
      str << std::endl;
    }
    str << std::endl;
    return str.str();
  } else {
    return "Is refined patch.\n";
  }
}

std::string peanoclaw::Patch::toString() const {
  std::stringstream str;
  if(_cellDescription != 0) {
    str << "size=[" << _cellDescription->getSize() << "],position=[" << _cellDescription->getPosition() << "],level=" << getLevel() << ",isLeaf=" << isLeaf() << ",isVirtual=" << isVirtual()
        << ",subdivisionFactor=[" << getSubdivisionFactor() << "],ghostlayerWidth=" << getGhostLayerWidth() << ",unknownsPerSubcell=" << _cellDescription->getUnknownsPerSubcell()
        << ",cellDescriptionIndex=" << getCellDescriptionIndex() << ",uNewIndex=" << getUNewIndex() << ",uOldIndex=" << getUOldIndex()
        << ",elements in uNew=" << ((_uNew==0) ? 0 : (int)_uNew->size()) << ",elements in uOld=" << ((_uOldWithGhostlayer==0) ? 0 : (int)_uOldWithGhostlayer->size())
        << ",age=" << _cellDescription->getAgeInGridIterations() << ",currentTime=" << _cellDescription->getTime() << ",timestepSize=" << _cellDescription->getTimestepSize()
        << ",minimalNeighborTimeConstraint=" << _cellDescription->getMinimalNeighborTimeConstraint() << ",skipNextGridIteration=" << _cellDescription->getSkipNextGridIteration()
        << ",minimalNeighborTime=" << _cellDescription->getMinimalNeighborTime() << ",maximalNeighborTimestepSize=" << _cellDescription->getMaximalNeighborTimestep()
        << ",maximalFineGridTime=" << _cellDescription->getMaximumFineGridTime() << ",minimalFineGridTimestepSize=" << _cellDescription->getMinimumFineGridTimestep()
        << ",estimatedNextTimestepSize=" << _cellDescription->getEstimatedNextTimestepSize() << ",demandedMeshWidth=" << _cellDescription->getDemandedMeshWidth()
        << ",lowerGhostlayerBounds=[" << _cellDescription->getRestrictionLowerBounds() << "],upperGhostlayerBounds=[" << _cellDescription->getRestrictionUpperBounds() << "]";
  } else {
    str << "null";
  }
  return str.str();
}

bool peanoclaw::Patch::isValid() const {
  return (_cellDescription != 0) && tarch::la::allGreater(_cellDescription->getSubdivisionFactor(), tarch::la::Vector<DIMENSIONS, int>(-1));
}

bool peanoclaw::Patch::isLeaf() const {
  return (_cellDescription != 0) && !_cellDescription->getIsVirtual() && (_uNew != 0 && _uOldWithGhostlayer != 0);
}

bool peanoclaw::Patch::isVirtual() const {
  if(_cellDescription == 0) {
    return false;
  }
  assertion8((!_cellDescription->getIsVirtual() || (_uNew != 0 && _uOldWithGhostlayer != 0)),
      getPosition(), getSize(), _cellDescription->getIsVirtual(), _uNew, _uOldWithGhostlayer, _cellDescription->getUNewIndex(), _cellDescription->getUOldIndex(), _cellDescription->getCellDescriptionIndex());
  return _cellDescription->getIsVirtual();
}

void peanoclaw::Patch::switchToVirtual() {

  _cellDescription->setIsVirtual(true);
  //Create uNew if necessary
  if(_cellDescription->getUNewIndex() == -1) {
    int uNewIndex = peano::heap::Heap<Data>::getInstance().createData();
    _cellDescription->setUNewIndex(uNewIndex);
    std::vector<Data>& virtualUNew = peano::heap::Heap<Data>::getInstance().getData(uNewIndex);
    size_t uNewArraySize = tarch::la::volume(_cellDescription->getSubdivisionFactor()) * _cellDescription->getUnknownsPerSubcell();
    virtualUNew.resize(uNewArraySize, 0.0);
    _uNew = &virtualUNew;
    counter += _uNew->capacity();
  }

  //Create uOld if necessary
  if(_cellDescription->getUOldIndex() == -1) {
    int uOldWithGhostlayerIndex = peano::heap::Heap<Data>::getInstance().createData();
    _cellDescription->setUOldIndex(uOldWithGhostlayerIndex);
    std::vector<Data>& virtualUOldWithGhostlayer = peano::heap::Heap<Data>::getInstance().getData(uOldWithGhostlayerIndex);
    size_t uOldWithGhostlayerArraySize = tarch::la::volume(_cellDescription->getSubdivisionFactor() +  2*_cellDescription->getGhostLayerWidth()) * _cellDescription->getUnknownsPerSubcell();
    virtualUOldWithGhostlayer.resize(uOldWithGhostlayerArraySize, 0.0);
    _uOldWithGhostlayer = &virtualUOldWithGhostlayer;

    counter += _uOldWithGhostlayer->capacity();
  }

  //Initialise aux array
  if(_cellDescription->getAuxIndex() == -1) {
    if(_cellDescription->getAuxiliarFieldsPerSubcell() > 0) {
      _cellDescription->setAuxIndex(peano::heap::Heap<Data>::getInstance().createData());
      std::vector<Data>& auxArray = peano::heap::Heap<Data>::getInstance().getData(_cellDescription->getAuxIndex());
      size_t auxArraySize = tarch::la::volume(_cellDescription->getSubdivisionFactor()) * _cellDescription->getAuxiliarFieldsPerSubcell();
      auxArray.resize(auxArraySize, -1.0);
      _auxArray = &auxArray;

      counter += _auxArray->capacity();
    } else {
      _cellDescription->setAuxIndex(-1);
    }
  }

  //TODO unterweg debug
//  std::cout << "Allocated Cells: " << counter << std::endl;
//  std::cout << "Allocated Memory: " << (sizeof(Data) * counter) << std::endl;
}

void peanoclaw::Patch::switchToNonVirtual() {
  assertion(!isLeaf())
  _cellDescription->setIsVirtual(false);
  if(_cellDescription->getUNewIndex() != -1) {

    counter -= _uNew->capacity();

    peano::heap::Heap<Data>::getInstance().deleteData(getUNewIndex());
    _cellDescription->setUNewIndex(-1);
    _uNew = 0;
  }

  if(_cellDescription->getUOldIndex() != -1) {

    counter -= _uOldWithGhostlayer->capacity();

    peano::heap::Heap<Data>::getInstance().deleteData(getUOldIndex());
    _cellDescription->setUOldIndex(-1);
    _uOldWithGhostlayer = 0;
  }

  if(_cellDescription->getAuxIndex() != -1) {

    counter -= _auxArray->capacity();
    peano::heap::Heap<Data>::getInstance().deleteData(getAuxIndex());
    _cellDescription->setAuxIndex(-1);
    _auxArray = 0;
  }
  assertion(!isLeaf() && !isVirtual());
}

void peanoclaw::Patch::switchToLeaf() {
  assertion(isVirtual() || isLeaf());
  _cellDescription->setIsVirtual(false);

  //TODO unterweg restricting to interval [0, 1]
  double time = 0.0; //_cellDescription->getTime();
  double timestepSize = 1.0; //_cellDescription->getTimestepSize();

  double factorForUOld = 1.0;
  double factorForUNew = 1.0;
  if(!tarch::la::equals(timestepSize, 0.0)) {
    factorForUOld = (_cellDescription->getTime() - time) / timestepSize;
    factorForUNew = (_cellDescription->getTime() + _cellDescription->getTimestepSize() - time) / timestepSize;
  }

  Area area;
  area._offset = tarch::la::Vector<DIMENSIONS, int>(0);
  area._size = getSubdivisionFactor();
  switchAreaToMinimalFineGridTimeInterval(area, factorForUOld, factorForUNew);

  #ifdef Asserts
  //Check uNew
  dfor(subcellIndex, getSubdivisionFactor()) {
    assertion2(getValueUNew(subcellIndex, 0) >= 0.0,
        toString(), subcellIndex);
  }
  //Check uOld
  dfor(subcellIndex, getSubdivisionFactor() + 2 * getGhostLayerWidth()) {
    assertion2(getValueUOld(subcellIndex - getGhostLayerWidth(), 0) >= 0.0,
        toString(), subcellIndex);
  }
  #endif

  assertion(isLeaf());
}

bool peanoclaw::Patch::shouldRestrict() {
  return tarch::la::oneGreater(_cellDescription->getRestrictionLowerBounds(), _cellDescription->getPosition())
         || tarch::la::oneGreater(_cellDescription->getPosition() + _cellDescription->getSize(), _cellDescription->getRestrictionUpperBounds());
}

bool peanoclaw::Patch::containsNaN() const {
  //Check uNew
  dfor(subcellIndex, getSubdivisionFactor()) {
    for(int unknown = 0; unknown < getUnknownsPerSubcell(); unknown++) {
      if(getValueUNew(subcellIndex, unknown) != getValueUNew(subcellIndex, unknown)) {
        return true;
      }
    }
  }
  //Check uOld
  dfor(subcellIndex, getSubdivisionFactor() + 2 * getGhostLayerWidth()) {
    for(int unknown = 0; unknown < getUnknownsPerSubcell(); unknown++) {
      if(getValueUOld(subcellIndex - getGhostLayerWidth(), unknown) != getValueUOld(subcellIndex - getGhostLayerWidth(), unknown)) {
        return true;
      }
    }
  }

  return false;
}

bool peanoclaw::Patch::containsNonPositiveNumberInUnknown(int unknown) const {
  //Check uNew
  dfor(subcellIndex, getSubdivisionFactor()) {
    if( !tarch::la::greater(getValueUNew(subcellIndex, unknown), 0.0) ) {
      return true;
    }
  }
  //Check uOld
//  dfor(subcellIndex, getSubdivisionFactor() + 2 * getGhostLayerWidth()) {
//    if( !tarch::la::greater(getValueUOld(subcellIndex - getGhostLayerWidth(), unknown), 0.0) ) {
//      return true;
//    }
//  }

  return false;
}

void peanoclaw::Patch::increaseAgeByOneGridIteration() {
  _cellDescription->setAgeInGridIterations(_cellDescription->getAgeInGridIterations() + 1);
}

int peanoclaw::Patch::getAge() const {
  return _cellDescription->getAgeInGridIterations();
}

void peanoclaw::Patch::resetNeighboringGhostlayerBounds() {
  _cellDescription->setRestrictionLowerBounds(std::numeric_limits<double>::max());
  _cellDescription->setRestrictionUpperBounds(-std::numeric_limits<double>::max());
}

void peanoclaw::Patch::updateLowerNeighboringGhostlayerBound(int dimension, double bound) {
  _cellDescription->setRestrictionLowerBounds(
    dimension,
    std::min(_cellDescription->getRestrictionLowerBounds(dimension), bound)
  );
}

void peanoclaw::Patch::updateUpperNeighboringGhostlayerBound(int dimension, double bound) {
  _cellDescription->setRestrictionUpperBounds(
    dimension,
    std::max(_cellDescription->getRestrictionUpperBounds(dimension), bound)
  );
}

tarch::la::Vector<DIMENSIONS, double> peanoclaw::Patch::getLowerNeighboringGhostlayerBounds() const {
  return _cellDescription->getRestrictionLowerBounds();
}

tarch::la::Vector<DIMENSIONS, double> peanoclaw::Patch::getUpperNeighboringGhostlayerBounds() const {
  return _cellDescription->getRestrictionUpperBounds();
}

std::ostream& operator<<(std::ostream& out, const peanoclaw::Patch& patch){
 return out << patch.toString() << std::endl;
}
