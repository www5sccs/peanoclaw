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
     * Testing the extrapolation of ghostlayer corners in a 2d-patch. The patch has
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
    void testEdgeExtrapolation2D();


    /**
     * Tests the extrapolation of edges and corners of a ghostlayer in 3d.
     *
     * The inital subgrid looks as follows:
     *
     * z==-2
       0  0  0  0  0  0  0
       0  0  0  0  0  0  0
       0  0 13 15 17  0  0
       0  0  7  9 11  0  0
       0  0  1  3  5  0  0
       0  0  0  0  0  0  0
       0  0  0  0  0  0  0

      z==-1
       0  0  0  0  0  0  0
       0  0  0  0  0  0  0
       0  0 14 16 18  0  0
       0  0  8 10 12  0  0
       0  0  2  4  6  0  0
       0  0  0  0  0  0  0
       0  0  0  0  0  0  0


      z==0
       0  0  1  3  5  0  0
       0  0  2  4  6  0  0
       5  6  0  0  0  6  5
       3  4  0  0  0  4  3
       1  2  0  0  0  2  1
       0  0  2  4  6  0  0
       0  0  1  3  5  0  0


      z==1
       0  0  7  9 11  0  0
       0  0  8 10 12  0  0
      11 12  0  0  0 12 11
       9 10  0  0  0 10  9
       7  8  0  0  0  8  7
       0  0  8 10 12  0  0
       0  0  7  9 11  0  0


      z==2
       0  0 13 15 17  0  0
       0  0 14 16 18  0  0
      17 18  0  0  0 18 17
      15 16  0  0  0 16 15
      13 14  0  0  0 14 13
       0  0 14 16 18  0  0
       0  0 13 15 17  0  0


      z==3
       0  0  0  0  0  0  0
       0  0  0  0  0  0  0
       0  0 14 16 18  0  0
       0  0  8 10 12  0  0
       0  0  2  4  6  0  0
       0  0  0  0  0  0  0
       0  0  0  0  0  0  0


      z==4
       0  0  0  0  0  0  0
       0  0  0  0  0  0  0
       0  0 13 15 17  0  0
       0  0  7  9 11  0  0
       0  0  1  3  5  0  0
       0  0  0  0  0  0  0
       0  0  0  0  0  0  0

       Hence, the expected results are

      z==-2
       0  0    7    9   11    8    9
       0  0  4.5  6.5  8.5 17/3 20/3
       0  0   13   15   17  6.5    7
       0  0    7    9   11  2.5    3
       0  0    1    3    5 -1.5   -1
       0  0 -7.5 -5.5 -3.5    0    0
       0  0  -11   -9   -7    0    0

      z==-1
       0  0 10.5 12.5 14.5 37/3 40/3
       0  0    8   10   12   10   11
       0  0   14   16   18   10 10.5
       0  0    8   10   12    6  6.5
       0  0    2    4    6    2  2.5
       0  0   -4   -2    0    0    0
       0  0 -7.5 -5.5 -3.5    0    0


      z==0
       0  0  1  3  5 8.5   9
       0  0  2  4  6   8 8.5
       5  6  0  0  0   6   5
       3  4  0  0  0   4   3
       1  2  0  0  0   2   1
       0  0  2  4  6   0   0
       0  0  1  3  5   0   0


      z==1
       0  0  7  9 11 14.5   15
       0  0  8 10 12   14 14.5
      11 12  0  0  0   12   11
       9 10  0  0  0   10    9
       7  8  0  0  0    8    7
       0  0  8 10 12    0    0
       0  0  7  9 11    0    0


      z==2
       0  0 13 15 17  0  0
       0  0 14 16 18  0  0
      17 18  0  0  0 18 17
      15 16  0  0  0 16 15
      13 14  0  0  0 14 13
       0  0 14 16 18  0  0
       0  0 13 15 17  0  0


      z==3
       0  0  0  0  0  0  0
       0  0  0  0  0  0  0
       0  0 14 16 18  0  0
       0  0  8 10 12  0  0
       0  0  2  4  6  0  0
       0  0  0  0  0  0  0
       0  0  0  0  0  0  0


      z==4
       0  0  0  0  0  0  0
       0  0  0  0  0  0  0
       0  0 13 15 17  0  0
       0  0  7  9 11  0  0
       0  0  1  3  5  0  0
       0  0  0  0  0  0  0
       0  0  0  0  0  0  0
     */
    void testEdgeAndCornerExtrapolation3D();

  public:
    void run();
    void setUp();
};


#endif /* PEANOCLAW_TESTS_GHOSTLAYEREXTRAPOLATIONTEST_H_ */
