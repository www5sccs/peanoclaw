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

#include "peano/applications/peanoclaw/Patch.h"
#include "peano/applications/peanoclaw/records/Data.h"


namespace peano {
  namespace applications {
    namespace peanoclaw {
      namespace tests {
      class PatchTest;
      }
    }
  }
}


/**
 * This is just a default test case that demonstrated how to write unit tests
 * in Peano. Feel free to rename, remove, or duplicate it. It is not required
 * by the project but often useful if you wanna write unit tests.
 */
class peano::applications::peanoclaw::tests::PatchTest: public tarch::tests::TestCase {
  private:
    typedef peano::applications::peanoclaw::records::Data Data;

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


  public:
    PatchTest();
    virtual ~PatchTest();

    virtual void run();

    virtual void setUp();
};


#endif /* PEANO_APPLICATIONS_PEANOCLAW_TESTS_PATCHTEST_H_ */
