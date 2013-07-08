/*
 * TestVertexEnumerator.cpp
 *
 *  Created on: Jul 5, 2011
 *      Author: unterweg
 */

#include "TestVertexEnumerator.h"

#include "peano/utils/Loop.h"

peanoclaw::tests::TestVertexEnumerator::TestVertexEnumerator(
  const Vector cellSize
) : _cellSize(cellSize)
{
}

peanoclaw::tests::TestVertexEnumerator::~TestVertexEnumerator()
{
}

void peanoclaw::tests::TestVertexEnumerator::setCellSize(
  tarch::la::Vector<DIMENSIONS, double> cellSize
) {
  _cellSize = cellSize;
}

peano::grid::CellFlags peanoclaw::tests::TestVertexEnumerator::getCellFlags() const {
  return peano::grid::NotStationary;
}

int peanoclaw::tests::TestVertexEnumerator::operator() (
  int localVertexNumber
) const {
  return localVertexNumber;
}

int peanoclaw::tests::TestVertexEnumerator::operator() (
  const LocalVertexIntegerIndex& localVertexNumber
) const {
  return peano::utils::dLinearised(localVertexNumber, 2);
}

tarch::la::Vector<DIMENSIONS, double> peanoclaw::tests::TestVertexEnumerator::getCellSize() const {
  return _cellSize;
}
