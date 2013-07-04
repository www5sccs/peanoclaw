/*
 * StatisticsTest.cpp
 *
 *  Created on: Jul 1, 2013
 *      Author: kristof
 */
#include "peanoclaw/tests/StatisticsTest.h"

#include "peanoclaw/statistics/ParallelGridValidator.h"

#include "peano/utils/Dimensions.h"

#include "tarch/la/Vector.h"
#include "tarch/tests/TestCaseFactory.h"
registerTest(peanoclaw::tests::StatisticsTest)


#ifdef UseTestSpecificCompilerSettings
#pragma optimize("",off)
#endif

void peanoclaw::tests::StatisticsTest::run() {
  testMethod(testGetNeighborPositionOnSameLevel);
  testMethod(testGetNeighborPositionOnDifferentLevel);
  testMethod(testGetNeighborPositionOnRectangularDomainWithOffset);
  testMethod(testRootPatch);
}

void peanoclaw::tests::StatisticsTest::setUp() {
  // @todo If you have to configure your global test object, please do this
  //       here. Typically this operation remains empty.
}

void peanoclaw::tests::StatisticsTest::testGetNeighborPositionOnSameLevel() {
  tarch::la::Vector<DIMENSIONS, double> position;
  assignList(position) = 2.0/3.0, 1.0/3.0;
  tarch::la::Vector<DIMENSIONS, double> size(1.0/3.0);
  int level = 2;
  tarch::la::Vector<DIMENSIONS, double> expectedNeighborPosition;
  assignList(expectedNeighborPosition) = 1.0, 0.0;
  tarch::la::Vector<DIMENSIONS, int> discreteNeighborPosition;
  assignList(discreteNeighborPosition) = 1, 0;

  tarch::la::Vector<DIMENSIONS, double> domainOffset(0.0);
  tarch::la::Vector<DIMENSIONS, double> domainSize(1.0);
  peanoclaw::statistics::ParallelGridValidator validator(domainOffset, domainSize);

  tarch::la::Vector<DIMENSIONS, double> neighborPosition
    = validator.getNeighborPositionOnLevel(
      position,
      size,
      level,
      level-1,
      discreteNeighborPosition
    );

  validateWithParams2(tarch::la::equals(neighborPosition, expectedNeighborPosition), neighborPosition, expectedNeighborPosition);
}

void peanoclaw::tests::StatisticsTest::testGetNeighborPositionOnDifferentLevel() {
  tarch::la::Vector<DIMENSIONS, double> position(1.0/3.0);
  tarch::la::Vector<DIMENSIONS, double> size(1.0/3.0);
  int level = 2;
  tarch::la::Vector<DIMENSIONS, double> expectedNeighborPosition;
  assignList(expectedNeighborPosition) = 2.0/3.0, 0.0;
  tarch::la::Vector<DIMENSIONS, int> discreteNeighborPosition;
  assignList(discreteNeighborPosition) = 1, -1;

  tarch::la::Vector<DIMENSIONS, double> domainOffset(0.0);
  tarch::la::Vector<DIMENSIONS, double> domainSize(1.0);
  peanoclaw::statistics::ParallelGridValidator validator(domainOffset, domainSize);

  tarch::la::Vector<DIMENSIONS, double> neighborPosition
    = validator.getNeighborPositionOnLevel(
      position,
      size,
      level,
      level,
      discreteNeighborPosition
    );

  validateWithParams2(tarch::la::equals(neighborPosition, expectedNeighborPosition), neighborPosition, expectedNeighborPosition);
}

void peanoclaw::tests::StatisticsTest::testGetNeighborPositionOnRectangularDomainWithOffset() {
  tarch::la::Vector<DIMENSIONS, double> position;
  assignList(position) = 2.0/3.0 + 1.0, 1.0/3.0 + 2.0;
  tarch::la::Vector<DIMENSIONS, double> size;
  assignList(size) = 2.0/3.0, 1.0/3.0;
  int level = 2;
  tarch::la::Vector<DIMENSIONS, double> expectedNeighborPosition;
  assignList(expectedNeighborPosition) = 4.0/3.0 + 1.0, 2.0;
  tarch::la::Vector<DIMENSIONS, int> discreteNeighborPosition;
  assignList(discreteNeighborPosition) = 1, -1;

  tarch::la::Vector<DIMENSIONS, double> domainOffset;
  assignList(domainOffset) = 1.0, 2.0;
  tarch::la::Vector<DIMENSIONS, double> domainSize;
  assignList(domainSize) = 2.0, 1.0;
  peanoclaw::statistics::ParallelGridValidator validator(domainOffset, domainSize);

  tarch::la::Vector<DIMENSIONS, double> neighborPosition
    = validator.getNeighborPositionOnLevel(
      position,
      size,
      level,
      level,
      discreteNeighborPosition
    );

  validateWithParams2(tarch::la::equals(neighborPosition, expectedNeighborPosition), neighborPosition, expectedNeighborPosition);
}

void peanoclaw::tests::StatisticsTest::testRootPatch() {
  tarch::la::Vector<DIMENSIONS, double> position(0.0);
  tarch::la::Vector<DIMENSIONS, double> size(1.0);
  int level = 1;
  tarch::la::Vector<DIMENSIONS, double> expectedNeighborPosition;
  assignList(expectedNeighborPosition) = -1.0, -1.0;
  tarch::la::Vector<DIMENSIONS, int> discreteNeighborPosition;
  assignList(discreteNeighborPosition) = -1, -1;

  tarch::la::Vector<DIMENSIONS, double> domainOffset(0.0);
  tarch::la::Vector<DIMENSIONS, double> domainSize(1.0);
  peanoclaw::statistics::ParallelGridValidator validator(domainOffset, domainSize);

  tarch::la::Vector<DIMENSIONS, double> neighborPosition
    = validator.getNeighborPositionOnLevel(
      position,
      size,
      level,
      level,
      discreteNeighborPosition
    );

  validateWithParams2(tarch::la::equals(neighborPosition, expectedNeighborPosition), neighborPosition, expectedNeighborPosition);
}

#ifdef UseTestSpecificCompilerSettings
#pragma optimize("",on)
#endif
