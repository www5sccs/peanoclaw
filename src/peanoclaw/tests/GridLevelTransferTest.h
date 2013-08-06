/*
 * GridLevelTransferTest.h
 *
 *  Created on: Mar 19, 2012
 *      Author: Kristof Unterweger
 */

#ifndef PEANO_APPLICATIONS_PEANOCLAW_GRIDLEVELTRANSFERTEST_H_
#define PEANO_APPLICATIONS_PEANOCLAW_GRIDLEVELTRANSFERTEST_H_

#include "tarch/tests/TestCase.h"

#include "peanoclaw/records/CellDescription.h"
#include "peanoclaw/records/Data.h"

#include "tarch/logging/Log.h"

namespace peanoclaw {
  namespace tests {
  class GridLevelTransferTest;
  }
}

/**
 * This test class tests the functionality to copy data between fine and coarse
 * parts of the grid. One important part is the determination of the indices of
 * adjacent patches on hanging nodes.
 */
class peanoclaw::tests::GridLevelTransferTest: public tarch::tests::TestCase {

  private:
    /**
     * Logging device.
     */
    static tarch::logging::Log _log;

    typedef peanoclaw::records::CellDescription CellDescription;
    typedef peanoclaw::records::Data Data;

    /**
     * Tests the setting of the adjacent patch indices on a single refined cell
     * surrounded by coarser cells.
     *
     * Therefore, we set up an array with the adjacent-patch-indices for the
     * coarse grid vertices.
     *
     * In 2d the numbers of the patches are like this
     *
     *   5  |  6  |  7
     * -----c-----d----
     *   3  | -1  |  4
     * -----a-----b----
     *   0  |  1  |  2
     *
     * Where the -1 shows the refined cell and a-d represent the four coarse
     * grid vertices. The adjacent-patch-indices of these coarse grid vertices
     * therefore look like
     *
     *  - a: [-1,  3,  1,  0]
     *  - b: [ 4, -1,  2,  1]
     *  - c: [ 6,  5, -1,  3]
     *  - d: [ 7,  6,  4, -1]
     *
     *  The test checks whether the indices on the hanging fine grid vertices are
     *  set accordingly.
     */
    void testAdjacentPatchIndicesForSingleRefinedCell();

    /**
     * This test creates a fine grid patch and tests, wether the updating of
     * the minimal neighbor time on the fine grid patch depending on the
     * coarse grid patch is working correctly.
     * Therefore, it resets the minimal neighbor time on the fine grid patch
     * and calls the updateMinimalNeighborTime(double) with a value of 1.0.
     * After this the fine grid patch should show the same minimal neighbor
     * time.
     */
    void testUpdateMinimalNeighborTime();

    /**
     * Test of calculation of the overlap area between a patch at (0.5, 0.5) of
     * size (5, 5) and a patch at (1, 4) of size (1.5, 2).
     * The patches overlap along the x0 axis from 1 to 2.5 and along the x1 axis
     * from 4 to 5. Therefore, the overlapping area is 1.5.
     */
    void testOverlappingAreaWithRealOverlap();

    /**
     * Test of calculation of the overlap area between a patch at (1, 1) of size
     * (1, 1) and a patch at (2, 1.5) of size (1, 1). The patches are touching
     * each other and, therefore, the overlap area is zero.
     */
    void testOverlappingAreaWithTouchingPatches();

    /**
     * Test of calculation of the overlap area between a patch at (-1, -1) of size
     * (3, 3) and a patch at (0, 3) of size (5, 5). The overlap area is zero.
     */
    void testOverlappingAreaWithoutOverlap();

    /**
     * Tests the restriction functionality that transfers the patch data from a fine
     * patch to the overlaying virtual coarse patch.
     *
     * The test creates a GridLevelTransfer object and performs a
     * stepDown(cellDescriptionIndex) which creates a virtual patch of size 2x2 cells.
     * Afterwards it fills a finegrid patch of 3x3 cells which is located at the
     * center of the finegrid patches.
     *
     * The coarse vertices only have one valid patch located to the right of the fine
     * patches. Since, this patch has a size of 2x2 cells and a ghostlayer width of
     * 2 cells, the setup looks like this:
     *
     * @image html RestrictionTest.png
     *
     * Here the boldly framed cells are the patches which actually exist. The integer
     * values in the fine subcells describe the preset values on the fine level, while
     * the numbers in the coarse ghostlayer subcells describe the numbers that are
     * supposed to be computed by the averaging.
     *
     */
    void testRestrictionToVirtualPatch();

  public:
  GridLevelTransferTest();
    virtual ~GridLevelTransferTest();

    virtual void run();

    virtual void setUp();
};

#endif /* PEANO_APPLICATIONS_PEANOCLAW_GRIDLEVELTRANSFERTEST_H_ */
