/*
 * TestVertexEnumerator.cpp
 *
 *  Created on: Jul 5, 2011
 *      Author: unterweg
 */

#include "TestVertexEnumerator.h"

#include "peano/utils/Loop.h"

peano::applications::peanoclaw::tests::TestVertexEnumerator::TestVertexEnumerator(
  const Vector cellSize
) : _cellSize(cellSize)
{
}

peano::applications::peanoclaw::tests::TestVertexEnumerator::~TestVertexEnumerator()
{
}

void peano::applications::peanoclaw::tests::TestVertexEnumerator::setCellSize(
  tarch::la::Vector<DIMENSIONS, double> cellSize
) {
  _cellSize = cellSize;
}

peano::kernel::gridinterface::CellFlags peano::applications::peanoclaw::tests::TestVertexEnumerator::getCellFlags() const {
  return peano::kernel::gridinterface::NotStationary;
}

int peano::applications::peanoclaw::tests::TestVertexEnumerator::operator() (
  int localVertexNumber
) const {
  return localVertexNumber;
}

int peano::applications::peanoclaw::tests::TestVertexEnumerator::operator() (
  const LocalVertexIntegerIndex& localVertexNumber
) const {
  return peano::utils::dLinearised(localVertexNumber, 2);
}

tarch::la::Vector<DIMENSIONS, double> peano::applications::peanoclaw::tests::TestVertexEnumerator::getCellSize() const {
  return _cellSize;
}
