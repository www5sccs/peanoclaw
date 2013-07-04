/*
 * StatisticsTest.h
 *
 *  Created on: Jul 1, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_TESTS_STATISTICSTEST_H_
#define PEANOCLAW_TESTS_STATISTICSTEST_H_

#include "tarch/tests/TestCase.h"

namespace peanoclaw {
  namespace tests {
    class StatisticsTest;
  }
}

class peanoclaw::tests::StatisticsTest : public tarch::tests::TestCase {

  private:
    /**
     * Tests the method getNeighborPositionOnLevel(...) from class
     * peanoclaw::statistics::ParallelGridValidator
     */
    void testGetNeighborPositionOnSameLevel();

    void testGetNeighborPositionOnDifferentLevel();

    void testGetNeighborPositionOnRectangularDomainWithOffset();

    void testRootPatch();

  public:
    void run();

    void setUp();
};


#endif /* PEANOCLAW_TESTS_STATISTICSTEST_H_ */
