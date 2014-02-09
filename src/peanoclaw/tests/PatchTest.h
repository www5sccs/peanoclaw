/*
 * PatchTest.h
 *
 *  Created on: Feb 15, 2012
 *      Author: Kristof Unterweger
 */

#ifndef PEANO_APPLICATIONS_PEANOCLAW_TESTS_PATCHTEST_H_
#define PEANO_APPLICATIONS_PEANOCLAW_TESTS_PATCHTEST_H_

#include "tarch/tests/TestCase.h"

#include <vector>

#include "peanoclaw/Patch.h"
#include "peanoclaw/records/Data.h"


namespace peanoclaw {
  namespace tests {
  class PatchTest;
  }
}


/**
 * This is just a default test case that demonstrated how to write unit tests
 * in Peano. Feel free to rename, remove, or duplicate it. It is not required
 * by the project but often useful if you wanna write unit tests.
 */
class peanoclaw::tests::PatchTest: public tarch::tests::TestCase {
  private:
    typedef peanoclaw::records::Data Data;

    /**
     * Test wether the filling of the uNew-Array is working properly.
     * Working on the 3x3x2 array
     *
     * Unknown 0:    Unknown 1:
     * 0 1 2          9 10 11
     * 3 4 5         12 13 14
     * 6 7 8         15 16 17
     */
    void testFillingOfUNewArray();

    /**
     * Test wether the filling of the uOld-Array is working properly.
     * Working on the 5x5 array
     *
     * Unknown 0:             Unknown 1:
     *  0  1  2  3  4         25 26 27 28 29
     *  5  6  7  8  9         30 31 32 33 34
     * 10 11 12 13 14         35 36 37 38 39
     * 15 16 17 18 19         40 41 42 43 44
     * 20 21 22 23 24         45 46 47 48 49
     */
    void testFillingOfUOldArray();

    /**
     * Checks wether a Patch object created via the default
     * constructor is considered to be invalid.
     */
    void testInvalidPatch();

    /**
     * This test checks whether the calculation of the coarse patch time interval
     * works as designed. Therefore, a 3x3 array of patches is set up with the
     * following time intervals
     * [1.0, 3.0] [1.0, 3.0] [1.0, 4.0]
     * [0.0, 3.0] [1.0, 2.7] [0.0, 4.0]
     * [0.0, 3.0] [1.0, 3.0] [2.5, 4.0]
     *
     * Here the first value is the time of the last timestep, the second value
     * the time of the current timestep. The timestep size therefore is the
     * difference of them. The correct time interval of the coarse patch is
     * [2.5, 2.7].
     */
    void testCoarsePatchTimeInterval();

    /**
     * Tests the counting of adjacent subgrids that reside on different ranks.
     *
     * For this test four vertices with the following adjacency information are created:
     *  1   1   1   1
     *   v2      v3
     *  0   0   0   0
     *
     *  0   0   0   0
     *   v0      v1
     *  0   0   0   0
     *
     *  Therefore, there is one neighboring rank and two vertices that are shared between
     *  the local and the neighboring rank.
     */
    void testCountingOfAdjacentParallelSubgrids();

    /**
     * Tests the counting of adjacent subgrids that reside on different ranks
     * where more than one ranks are neighboring.
     *
     * For this test four vertices with the following adjacency information are created:
     *  1   1   1   1
     *   v2      v3
     *  4   0   0   2
     *
     *  4   0   0   2
     *   v0      v1
     *  3   3   3   3
     *
     *  Therefore, there are four neighboring ranks. I.e. the number of shared vertices
     *  can't be counted.
     */
    void testCountingOfAdjacentParallelSubgridsFourNeighboringRanks();

    /**
     * Tests the setting of the overlap of remote ghostlayers which is done in
     * AdjacentSubgrids. The following setup is considered:
     *
     * Ranks:
     * 0   0
     *   v
     * 1   2
     *
     * The top-left subgrid does not exist, only the top-right subgrid is considered.
     * The subcell size is 0.1x0.1 in every case, while the ghostlayer width of
     * rank 1 is 2 and for rank 2 is 1.
     * The expected result in the according fields of the subgrid are:
     *
     *  0|   0  |0
     * --|------|--
     *  0|      |0
     *   |      |
     * --|------|--
     *  2|   1  |0
     */
    void testSettingOverlapOfRemoteGhostlayer();

    /**
     * Tests the number of manifolds for a subgrid as well as the
     * determination of the actual manifolds.
     */
    void testManifolds();

    /**
     * Tests the number of adjacent manifolds for a given manifold
     * of a given dimensionality.
     */
    void testNumberOfAdjacentManifolds();

    /**
     * Tests the determination of adjacent manifolds for a given
     * manifold of a given dimensionality. This operation is done
     * in Area::getIndexOfAdjacentManifold(...)
     *
     *  0|   0  |0
     * --|------|--
     *  1|      |0
     *   |      |
     * --|------|--
     *  1|   1  |0
     */
    void testAdjacentManifolds();

    /**
     * Tests the determination of the overlapped areas by remote
     * ghostlayers.
     *
     * For 2D we test the following scenario, where the first
     * number represents the adjacent rank and the second represents
     * the overlap.
     *
     * 1/2|  2/2 |2/3
     *  --|------|--
     * 1/2|      |0/2
     *    |      |
     *  --|------|--
     * 1/3|  1/2 |0/2
     *
     * The subdivision factor of the subgrid is 6. Hence, we expect
     * the following areas for rank 1:
     *
     * offset, size
     * [0, 0], [3, 3]
     * [0, 3], [2, 3]
     * [3, 0], [3, 2]
     *
     * and for rank 2:
     *
     * offset, size
     * [3, 3], [3, 3]
     * [0, 4], [3, 2]
     *
     *
     * For 3D a similar setup is used, where the upper settings are mapped
     * to the x0/x2 plane. Additionally, all corners with x1==-1 have an overlap
     * of 3, while all corners with x1==1 have an overlap of 2.
     * Hence, we expect the following areas for rank 1:
     *
     * offset,    size
     * [0, 0, 0], [3, 3, 3]
     * [
     */
    void testOverlapOfRemoteGhostlayers();

    /**
     * A second testcase for a problem that occured in 2D:
     *
     *0/18| 0/18 |0/18
     *  --|------|--
     * 1/0|      |2/2
     *    |      |
     *  --|------|--
     * 1/0|  1/0 |2/2
     *
     * The subdivision factor is 18.
     * For rank 2 there is only one area expected:
     *
     * offset,  size
     * [16, 0], [2, 18]
     */
    void testOverlapOfRemoteGhostlayers2();

  public:
    PatchTest();
    virtual ~PatchTest();

    virtual void run();

    virtual void setUp();
};


#endif /* PEANO_APPLICATIONS_PEANOCLAW_TESTS_PATCHTEST_H_ */
