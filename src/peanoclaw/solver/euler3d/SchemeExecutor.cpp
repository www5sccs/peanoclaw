/*
 * SchemeExecutor.cpp
 *
 *  Created on: Sep 22, 2014
 *      Author: kristof
 */
#include "peanoclaw/solver/euler3d/SchemeExecutor.h"

#include "peanoclaw/solver/euler3d/Cell.h"

#include <iomanip>

tarch::logging::Log peanoclaw::solver::euler3d::SchemeExecutor::_log("peanoclaw::solver::euler3d::SchemeExecutor");

#ifdef PEANOCLAW_EULER3D
#include <tbb/blocked_range.h>
#include <tbb/parallel_for_each.h>

peanoclaw::solver::euler3d::SchemeExecutor::SchemeExecutor(
  SchemeExecutor& other,
  tbb::split
) : _subgrid(other._subgrid),
    _accessor(other._accessor),
    _scheme(other._scheme),
    #ifdef PEANOCLAW_EULER3D
    _cellUpdatesPerThread(),
    #endif
    _dt(other._dt),
    _leftCellOffset(other._leftCellOffset),
    _rightCellOffset(other._rightCellOffset),
    _bottomCellOffset(other._bottomCellOffset),
    _topCellOffset(other._topCellOffset),
    _backCellOffset(other._backCellOffset),
    _frontCellOffset(other._frontCellOffset),
    _uOldOffset(other._uOldOffset),
    _numberOfUnknowns(other._numberOfUnknowns),
    _stride(other._stride),
    _uOldArray(other._uOldArray),
    _maxLambda(-1),
    _maxDensity(std::numeric_limits<double>::min()),
    _minDensity(std::numeric_limits<double>::max())
{
}
#endif

peanoclaw::solver::euler3d::SchemeExecutor::SchemeExecutor(
  peanoclaw::Patch& subgrid,
  double const& timeStepSize
) : _subgrid(subgrid),
    _accessor(subgrid.getAccessor()),
    _dt(timeStepSize),
    _leftCellOffset(0),
    _rightCellOffset(0),
    _bottomCellOffset(0),
    _topCellOffset(0),
    _backCellOffset(0),
    _frontCellOffset(0),
    _uOldOffset(0),
    _numberOfUnknowns(subgrid.getUnknownsPerSubcell()),
    _stride(0),
    _uOldArray(subgrid.getAccessor().getUOldWithGhostLayerArray(0)),
    _maxLambda(-1),
    _maxDensity(std::numeric_limits<double>::min()),
    _minDensity(std::numeric_limits<double>::max())
{
  tarch::la::Vector<DIMENSIONS,double> subcellSize = subgrid.getSubcellSize();
  _scheme.courantNumber(
    timeStepSize,
    subcellSize[0],
    subcellSize[1],
    subcellSize[2]
  );

  tarch::la::Vector<DIMENSIONS,int> neighborSubcellIndex;
  assignList(neighborSubcellIndex) = 0, 0, 0;
  _uOldOffset = _accessor.getLinearIndexUOld(neighborSubcellIndex);
  assignList(neighborSubcellIndex) = -1, 0, 0;
  _leftCellOffset = _accessor.getLinearIndexUOld(neighborSubcellIndex) - _uOldOffset;
  assignList(neighborSubcellIndex) = 1, 0, 0;
  _rightCellOffset = _accessor.getLinearIndexUOld(neighborSubcellIndex) - _uOldOffset;
  assignList(neighborSubcellIndex) = 0, -1, 0;
  _bottomCellOffset = _accessor.getLinearIndexUOld(neighborSubcellIndex) - _uOldOffset;
  assignList(neighborSubcellIndex) = 0, 1, 0;
  _topCellOffset = _accessor.getLinearIndexUOld(neighborSubcellIndex) - _uOldOffset;
  assignList(neighborSubcellIndex) = 0, 0, -1;
  _backCellOffset = _accessor.getLinearIndexUOld(neighborSubcellIndex) - _uOldOffset;
  assignList(neighborSubcellIndex) = 0, 0, 1;
  _frontCellOffset = _accessor.getLinearIndexUOld(neighborSubcellIndex) - _uOldOffset;

  _stride[DIMENSIONS-1] = 1;
  for(int d = DIMENSIONS-2; d >= 0; d--) {
    _stride[d] = _stride[d+1] * _subgrid.getSubdivisionFactor()[d+1];
  }
}

void peanoclaw::solver::euler3d::SchemeExecutor::operator()(
    tbb::blocked_range<int> const& range
) {
  tbb::tbb_thread::id threadID = tbb::this_tbb_thread::get_id();
  std::map<tbb::tbb_thread::id, int>::iterator entry = _cellUpdatesPerThread.find(threadID);
  if(entry == _cellUpdatesPerThread.end()) {
    _cellUpdatesPerThread[threadID] = 0;
    entry = _cellUpdatesPerThread.find(threadID);
  }
  entry->second += range.end() - range.begin();

  for (int iterator = range.begin(); iterator != range.end(); iterator++) {
    int linearIndex = iterator;
    tarch::la::Vector<DIMENSIONS,int> subcellIndex;
    for(int d = 0; d < DIMENSIONS; d++) {
      subcellIndex[d] = linearIndex / _stride[d];
      linearIndex = linearIndex % _stride[d];
    }

    this->operator()(subcellIndex);
  }
}

void peanoclaw::solver::euler3d::SchemeExecutor::operator()(
  const tarch::la::Vector<DIMENSIONS, int>& subcellIndex
) {
  int linearIndexUNew = _accessor.getLinearIndexUNew(subcellIndex);
  int linearIndexUOld = _accessor.getLinearIndexUOld(subcellIndex);

  #ifdef Asserts
  tarch::la::Vector<DIMENSIONS,int> subdivisionFactor = _subgrid.getSubdivisionFactor();
  int ghostlayerWidth = _subgrid.getGhostlayerWidth();
  //The number of doubles in the subgrid-array for a single cell, one row of the array in z-direction, and one yz-plane.
  int numberOfUnknowns =  _subgrid.getUnknownsPerSubcell();
  int zRow = (subdivisionFactor[2] + 2*ghostlayerWidth) * _subgrid.getUnknownsPerSubcell();
  int yzPlane = (subdivisionFactor[1] + 2*ghostlayerWidth) * (subdivisionFactor[2] + 2*ghostlayerWidth) * _subgrid.getUnknownsPerSubcell();
  #endif

  tarch::la::Vector<DIMENSIONS,int> unitX;
  assignList(unitX) = 1, 0, 0;
  tarch::la::Vector<DIMENSIONS,int> unitY;
  assignList(unitY) = 0, 1, 0;
  tarch::la::Vector<DIMENSIONS,int> unitZ;
  assignList(unitZ) = 0, 0, 1;

  //Left/Right: x
  //Bottom/Top: y
  //Back/Front: z
  peanoclaw::solver::euler3d::Cell leftCell(
    &_uOldArray[linearIndexUOld + _leftCellOffset],
    subcellIndex-unitX
    //cellsUOld[linearUOldIndex-yzPlane];
  );
  assertion2(
    (linearIndexUOld + _leftCellOffset) >= 0
    && (linearIndexUOld + _leftCellOffset) < tarch::la::volume(_subgrid.getSubdivisionFactor() + 2 * _subgrid.getGhostlayerWidth()) * _subgrid.getUnknownsPerSubcell(),
    linearIndexUOld,
    _leftCellOffset
  );

  peanoclaw::solver::euler3d::Cell bottomCell(
    &_uOldArray[linearIndexUOld + _bottomCellOffset],
    subcellIndex-unitY
    //cellsUOld[linearUOldIndex-zRow];
  );
  assertion2(
    (linearIndexUOld + _bottomCellOffset) >= 0
    && (linearIndexUOld + _bottomCellOffset) < tarch::la::volume(_subgrid.getSubdivisionFactor() + 2 * _subgrid.getGhostlayerWidth()) * _subgrid.getUnknownsPerSubcell(),
    linearIndexUOld,
    _bottomCellOffset
  );

  peanoclaw::solver::euler3d::Cell backCell(
    &_uOldArray[linearIndexUOld + _backCellOffset],
    subcellIndex-unitZ
    //cellsUOld[linearUOldIndex-1];
  );
  assertion2(
    (linearIndexUOld + _backCellOffset) >= 0
    && (linearIndexUOld + _backCellOffset) < tarch::la::volume(_subgrid.getSubdivisionFactor() + 2 * _subgrid.getGhostlayerWidth()) * _subgrid.getUnknownsPerSubcell(),
    linearIndexUOld,
    _backCellOffset
  );

  peanoclaw::solver::euler3d::Cell centerCell(
    &_uOldArray[linearIndexUOld],
    subcellIndex
    //cellsUOld[linearUOldIndex];
  );
  assertion1(
    linearIndexUOld >= 0
    && linearIndexUOld < tarch::la::volume(_subgrid.getSubdivisionFactor() + 2 * _subgrid.getGhostlayerWidth()) * _subgrid.getUnknownsPerSubcell(),
    linearIndexUOld
  );

  peanoclaw::solver::euler3d::Cell frontCell(
    &_uOldArray[linearIndexUOld + _frontCellOffset],
    subcellIndex+unitZ
    //cellsUOld[linearUOldIndex+1];
  );
  assertion2(
    (linearIndexUOld + _frontCellOffset) >= 0
    && (linearIndexUOld + _frontCellOffset) < tarch::la::volume(_subgrid.getSubdivisionFactor() + 2 * _subgrid.getGhostlayerWidth()) * _subgrid.getUnknownsPerSubcell(),
    linearIndexUOld,
    _frontCellOffset
  );

  peanoclaw::solver::euler3d::Cell topCell(
    &_uOldArray[linearIndexUOld + _topCellOffset],
    subcellIndex+unitY
    //cellsUOld[linearUOldIndex+zRow];
  );
  assertion2(
    (linearIndexUOld + _topCellOffset) >= 0
    && (linearIndexUOld + _topCellOffset) < tarch::la::volume(_subgrid.getSubdivisionFactor() + 2 * _subgrid.getGhostlayerWidth()) * _subgrid.getUnknownsPerSubcell(),
    linearIndexUOld,
    _topCellOffset
  );

  peanoclaw::solver::euler3d::Cell rightCell(
    &_uOldArray[linearIndexUOld + _rightCellOffset],
    subcellIndex+unitX
    //cellsUOld[linearUOldIndex+yzPlane];
  );
  assertion5(
    (linearIndexUOld + _rightCellOffset) >= 0
    && (linearIndexUOld + _rightCellOffset) < tarch::la::volume(_subgrid.getSubdivisionFactor() + 2 * _subgrid.getGhostlayerWidth()) * _subgrid.getUnknownsPerSubcell(),
    subcellIndex,
    linearIndexUOld,
    _rightCellOffset,
    _subgrid.getSubdivisionFactor(),
    _subgrid.getUnknownsPerSubcell()
  );

  assertionEquals(leftCell._index, subcellIndex-unitX);
  assertionEquals(bottomCell._index, subcellIndex-unitY);
  assertionEquals(backCell._index, subcellIndex-unitZ);
  assertionEquals(centerCell._index, subcellIndex);
  assertionEquals(frontCell._index, subcellIndex+unitZ);
  assertionEquals(topCell._index, subcellIndex+unitY);
  assertionEquals(rightCell._index, subcellIndex+unitX);

  assertionEquals2(linearIndexUOld-yzPlane, _accessor.getLinearIndexUOld(subcellIndex - unitX), subcellIndex, linearIndexUOld);
  assertionEquals(linearIndexUOld-zRow, _accessor.getLinearIndexUOld(subcellIndex - unitY));
  assertionEquals(linearIndexUOld-numberOfUnknowns, _accessor.getLinearIndexUOld(subcellIndex - unitZ));
  assertionEquals(linearIndexUOld, _accessor.getLinearIndexUOld(subcellIndex));
  assertionEquals(linearIndexUOld+numberOfUnknowns, _accessor.getLinearIndexUOld(subcellIndex + unitZ));
  assertionEquals(linearIndexUOld+zRow, _accessor.getLinearIndexUOld(subcellIndex + unitY));
  assertionEquals(linearIndexUOld+yzPlane, _accessor.getLinearIndexUOld(subcellIndex + unitX));

  peanoclaw::solver::euler3d::Cell newCell(
    &_accessor.getUNewArray()[linearIndexUNew],
    subcellIndex
  );

  double localMaxLambda;
  _scheme.apply(
    leftCell,
    rightCell,
    bottomCell,
    topCell,
    backCell,
    frontCell,
    centerCell,
    newCell,
    localMaxLambda
  );
  _maxLambda = std::max(_maxLambda, localMaxLambda);

  //TODO unterweg debug
  bool plot =
      false;
//          (x > 3 && x < 6) && (y > 3 && y < 6) && (z > 3 && z < 6);
//            x == 1 && y == 2 && z == 2;
//            x < 3 && y == 0 && z == 0;
//      subcellIndex[0] == 7 && subcellIndex[1] == 12 && subcellIndex[2] == 6;

  if(plot) {
    std::cout << subcellIndex << " linearIndexUNew=" << linearIndexUNew << std::endl;
    std::cout << "dt=" << _dt << std::endl;
    std::cout << "left: density=" << std::setprecision(3) << leftCell.density() << ", momentum=" << leftCell.velocity()(0) << "," << leftCell.velocity()(1) << "," << leftCell.velocity()(2) << ", energy=" << leftCell.energy() << std::endl;
    std::cout << "right: density=" << std::setprecision(3) << rightCell.density() << ", momentum=" << rightCell.velocity()(0) << "," << rightCell.velocity()(1) << "," << rightCell.velocity()(2) << ", energy=" << rightCell.energy() << std::endl;
    std::cout << "bottom: density=" << std::setprecision(3) << bottomCell.density() << ", momentum=" << bottomCell.velocity()(0) << "," << bottomCell.velocity()(1) << "," << bottomCell.velocity()(2) << ", energy=" << bottomCell.energy() << std::endl;
    std::cout << "top: density=" << std::setprecision(3) << topCell.density() << ", momentum=" << topCell.velocity()(0) << "," << topCell.velocity()(1) << "," << topCell.velocity()(2) << ", energy=" << topCell.energy() << std::endl;
    std::cout << "back: density=" << std::setprecision(3) << backCell.density() << ", momentum=" << backCell.velocity()(0) << "," << backCell.velocity()(1) << "," << backCell.velocity()(2) << ", energy=" << backCell.energy() << std::endl;
    std::cout << "front: density=" << std::setprecision(3) << frontCell.density() << ", momentum=" << frontCell.velocity()(0) << "," << frontCell.velocity()(1) << "," << frontCell.velocity()(2) << ", energy=" << frontCell.energy() << std::endl;
    std::cout << "center: density=" << std::setprecision(3) << centerCell.density() << ", momentum=" << centerCell.velocity()(0) << "," << centerCell.velocity()(1) << "," << centerCell.velocity()(2) << ", energy=" << centerCell.energy() << std::endl;
    std::cout << "  new cell density=" << newCell.density() << ", momentum=" << newCell.velocity()(0) << "," << newCell.velocity()(1) << "," << newCell.velocity()(2) << ", energy=" << newCell.energy() << std::endl;
  }
}

void peanoclaw::solver::euler3d::SchemeExecutor::join(
  SchemeExecutor const& other
) {
  _maxLambda = std::max(_maxLambda, other._maxLambda);
  _maxDensity     = std::max(_maxDensity, other._maxDensity);
  _minDensity     = std::min(_minDensity, other._minDensity);

  //TODO unterweg debug
//  std::cout << "Merging..." << std::endl;
  for(std::map<tbb::tbb_thread::id, int>::const_iterator i = other._cellUpdatesPerThread.begin(); i != other._cellUpdatesPerThread.end(); i++) {
    std::map<tbb::tbb_thread::id, int>::iterator entry = _cellUpdatesPerThread.find(i->first);
    if(entry == _cellUpdatesPerThread.end()) {
      //TODO unterweg debug
//      std::cout << "\tadding thread " << i->first << ": " << i->second << std::endl;
      _cellUpdatesPerThread[i->first] = i->second;
    } else {
//      throw "";
      //TODO unterweg debug
//      std::cout << "\tmerging thread " << i->first << ": " << i->second << std::endl;
      _cellUpdatesPerThread[i->first] += i->second;
    }
  }
}

double peanoclaw::solver::euler3d::SchemeExecutor::getMaximumLambda() const {
  return _maxLambda;
}

void peanoclaw::solver::euler3d::SchemeExecutor::logStatistics() const {
  logInfo("logStatistics()", "#Threads=" << _cellUpdatesPerThread.size());
  int threadIndex = 0;
  for(std::map<tbb::tbb_thread::id, int>::const_iterator entry = _cellUpdatesPerThread.begin();
      entry != _cellUpdatesPerThread.end();
      entry++) {
    logInfo("logStatistics()", "CellUpdates on thread " << threadIndex << ": " << entry->second);
    threadIndex++;
  }
}


