/*
 * GhostlayerExtrapolationTest.h
 *
 *  Created on: Jul 8, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_TESTS_GHOSTLAYEREXTRAPOLATIONTEST_H_
#define PEANOCLAW_TESTS_GHOSTLAYEREXTRAPOLATIONTEST_H_

#include "tarch/tests/TestCase.h"

namespace peanoclaw {
  namespace tests {
    class GhostlayerExtrapolationTest;
  }
}

class peanoclaw::tests::GhostlayerExtrapolationTest : public tarch::tests::TestCase {

  private:
    /**
     * Testing the interpolation of ghostlayer corners in a 2d-patch. The patch has
     * 3x3 cells and a ghostlayer width of 2. Thus, the values are set as follows:
     *
     *  c1 c3| 3  2  1|c2 c1
     *  c2 c4| 6  5  4|c4 c3
     *   -------------------
     *   1  4| 0  0  0| 6  3
     *   2  5| 0  0  0| 5  2
     *   3  6| 0  0  0| 4  1
     *   -------------------
     *  c3 c4| 4  5  6|c4 c2
     *  c1 c2| 1  2  3|c3 c1
     *
     * The corner values therefore derive as follows:
     * c1 = ((3*1 -2*2) + (3*3 -2*2)) / 2 = (-1 + 5) / 2 =  4/2 = 2
     * c2 = ((2*1 -1*2) + (3*6 -2*5)) / 2 =  (0 + 8) / 2 =  8/2 = 4
     * c3 = ((3*4 -2*5) + (2*3 -1*2)) / 2 =  (2 + 4) / 2 =  6/2 = 3
     * c4 = ((2*4 -1*5) + (2*6 -1*5)) / 2 =  (3 + 7) / 2 = 10/2 = 5
     */
    void testCornerExtrapolation2D();

  public:
    void run();
    void setUp();
};


#endif /* PEANOCLAW_TESTS_GHOSTLAYEREXTRAPOLATIONTEST_H_ */
