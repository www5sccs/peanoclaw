/*
 * GhostLayerCompositorTest.h
 *
 *  Created on: Mar 6, 2012
 *      Author: Kristof Unterweger
 */

#ifndef PEANO_APPLICATIONS_PEANOCLAW_TESTS_GHOSTLAYERCOMPOSITORTEST_H_
#define PEANO_APPLICATIONS_PEANOCLAW_TESTS_GHOSTLAYERCOMPOSITORTEST_H_

#include "peano/utils/Globals.h"

#include "tarch/tests/TestCase.h"

#include "tarch/la/Vector.h"
#include "tarch/logging/Log.h"

#include <vector>

namespace peanoclaw {
  class Patch;

  namespace tests {
    class GhostLayerCompositorTest;
    class TestEdgeAdjacentPatchTraversalFunctor;
    class TestFaceAdjacentPatchTraversalFunctor;
  }
}

class peanoclaw::tests::GhostLayerCompositorTest : public tarch::tests::TestCase {
private:
  /**
   * Logging device.
   */
  static tarch::logging::Log _log;

  /**
   * Checks wether the veto mechanism to forbid timestepping for a grid
   * works as expected. Therefore it sets up a GhostLayerCompositor with
   * the following times
   * 0.0|1.0      0.0|1.0
   *
   * 0.5|1.5      0.0|1.0
   *
   * So the only cell which cannot advance in time is the lower left one (index=3).
   *
   */
  void testTimesteppingVeto2D();

  /**
   * Tests the interpolation of subcell-data from a coarse patch to a fine patch.
   *
   * The coarse patch is of size 2x2 with a ghostlayer width of 1 cell, and it is
   * three times larger than the fine one. The coarse patch is located at (2,1)
   * with size (3,3) while the fine one is located at (5,2) with size (1,1)
   * The coarse patch is filled with the values
   *
   * UNew:
   *  1  1
   *  1  1
   *
   * UOld:
   * -1 -1 -1 -1
   * -1  7 11 -1
   * -1  6 10 -1
   * -1 -1 -1 -1
   *
   * The fine patch is of size 2x2 with a ghostlayer width of 2 cells. The four
   * values of the left ghostlayer of the fine patch reside on the points
   * (4.25, 2.25), (4.25, 2.75), (4.75, 2.25), and (4.75, 2.75).
   *
   * Due to the extrapolation scheme along the patch's boundary they depend on
   * the values of the coarse grid residing at
   * (2.75, 1.75): 0, (2.75, 3.25): 1, (4.25, 1.75): 2, and (4.25, 3.25): 3.
   *
   * The interpolated values therefore are:
   *
   * 1*2/3*10 + 1*1/3*11 = 20/3 + 11/3 = 31/3
   * 1*1/3*10 + 1*2/3*11 = 10/3 + 22/3 = 32/3
   * -1/3*2/3*6 + -1/3*1/3*7 + 4/3*2/3*10 + 4/3*1/3*11 = -4/3 - 7/9 + 80/9 + 44/9 = 105/9
   * -1/3*1/3*6 + -1/3*2/3*7 + 4/3*1/3*10 + 4/3*2/3*11 = -2/3 - 14/9 + 40/9 + 88/9 = 108/9
   *
   * To test the interpolation in time the UNew values are set to 1 in the whole
   * coarse patch. The time interval of the coarse patch is [0, 1], while the
   * time of UNew in the fine patch is 1/3. By this, we get the following
   * results:
   *
   * 31/3 * 2/3 + 1 * 1/3 = 62/9 + 1/3 = 65/9
   * 32/3 * 2/3 + 1 * 1/3 = 64/9 + 1/3 = 67/9
   * 105/9 * 2/3 + 1 * 1/3 = 210/27 + 1/3 = 73/9
   * 108/9 * 2/3 + 1 * 1/3 = 216/27 + 1/3 = 25/3
   *
   */
  void testInterpolationFromCoarseToFinePatchLeftGhostLayer2D();

  /**
   * The same test case like
   * testInterpolationFromCoarseToFinePatchLeftGhostLayer2D only with the fine
   * patch on the left of the coarse grid, i.e. filling the right ghostlayer.
   * So, the fine patch is located at 1, 1 while the coarse patch is located at
   * 2, 1.
   *
   * * The coarse patch is filled with the values
   *
   * UNew:
   *    1  1
   *    1  1
   *
   * UOld:
   * 4  8 12 16
   * 3  7 11 15
   * 2  6 10 14
   * 1  5  9 13
   *
   * The fine patch is of size 2x2 with a ghostlayer width of 2 cells. The four
   * values of the right ghostlayer of the fine patch reside on the points
   * (2.25, 1.25), (2.25, 1.75), (2.75, 1.25), and (2.75, 1.75).
   *
   * Due to the extrapolation scheme along the patch's boundary they depend on
   * the values of the coarse grid residing at
   * (2.75, 1.75): 0, (2.75, 3.25): 1, (4.25, 1.75): 2, and (4.25, 3.25): 3.
   *
   * The interpolated values therefore are:
   *
   * 4/3*4/3*6 + 4/3*(-1/3)*7 + -1/3*4/3*10 + -1/3*(-1/3)*11 = 32/3 - 28/9 - 40/9 + 11/9 = 13/3
   * 4/3*1*6 + -1/3*1*10 = 24/3  - 10/3 = 14/3
   * 1*4/3*6 + -1*1/3*7  = 24/3  - 7/3  = 17/3
   * 1*1*6 = 6
   *
   * To test the interpolation in time the UNew values are set to 1 in the whole
   * coarse patch. The time interval of the coarse patch is [0, 1], while the
   * time of UNew in the fine patch is 1/3. By this, we get the following
   * results:
   *
   * 13/3 * 2/3 + 1 * 1/3 = 26/9   + 1/3 = 29/9
   * 14/3 * 2/3 + 1 * 1/3 = 28/9   + 1/3 = 31/9
   * 17/3 * 2/3 + 1 * 1/3 = 34/9   + 1/3 = 37/9
   * 6    * 2/3 + 1 * 1/3 = 4      + 1/3 = 13/3
   */
  void testInterpolationFromCoarseToFinePatchRightGhostLayer2D();

  /**
   * Tests the projection of subcell-data from a coarse patch to a fine patch.
   *
   * Both patches are of size 4x4, while the coarse patch is three times larger
   * than the fine one. The coarse patch is located at (1,2) with size (0.3,0.3)
   * while the fine one is located at (0.7,2.0) with size (0.1,0.1).
   */
  void testProjectionFromCoarseToFinePatchRightGhostLayer2D();

  /**
   * Tests the correction step for the coarse grid values according to the fine
   * grid fluxes.
   *
   * The test sets up two patches, a coarse one and a fine one, where the coarse
   * one consists out of 2x2 cells and the fine one out of 3x3. The fine patch is
   * located at (2/3, 1/3) and has size (1/3, 1/3), while the coarse one is on the
   * right of the fine one at (1, 0) and has size (1, 1).
   *
   * The cells in the fine patch are filled with the values
   *
   * 1 1 54
   * 1 1 36
   * 1 1 18
   *
   * and all velocities are set to (0.5, -0.5) while the coarse cells all have the
   * value 1 and velocity (0.5, 0.5).
   *
   * the resulting values in the coarse patch are therefore
   *
   * 4466/729 1
   * 2558/729 1
   *
   * !! Derivation
   * Flux from lower right fine cell F(f,0) = u(f,0) * velocity(f,0) * h(f) = 18 * 0.5 * 1/9 = 1
   * The transfered volume from this cell s(f,0) = F(f,0) * deltaT(f) = 1 * 1/3 = 1/3
   * Flux to the coarse cell from this fine cell F(c,0) = u(c,0) * velocity(c,0) * h(f) = 1 * 0.5 * 1/9 = 1/18
   * The transfered volume to the coarse cell from this fine cell s(c,0,1) = F(c,0) * deltaT(f) = 1/18 * 1/3 = 1/54
   * Correction for the lower left coarse cell from this fine cell delta(c,0,0) = 1/hc^2 * (s(f,0) - s(c,0,0))
   *    = 4 * (1/3 - 1/54) = 34/27
   * New value for the lower left coarse cell therefore is u'(c,0) = u(c,0) + delta(c,0,0) = 1 + 34/27 = 61/27
   *
   * Flux from middle right fine cell F(f,1) = u(f,1) * velocity(f,1) * h(f)/2 = 36 * 0.5 * 1/18 = 1
   * The transfered volume from this cell s(f,1) = F(f,1) * deltaT(f) = 1 * 1/3 = 1/3
   * The transfered volume to the coarse cell from this fine cell s(c,0,1) = u'(c,0) * velocity(c,0) * h(f)/2 * deltaT(f) = 61/2916
   * Correction for the lower left coarse cell from this fine cell delta(c,0,1) = 1/hc^2 * (s(f,1) - s(c,0,1))
   *    = 4 * (1/3 - 61/2916) = 911/729
   * New value for the lower left coarse cell therefore is u''(c,0) = u'(c,0) + delta(c,0,1) = 2558/729
   *
   * Flux from middle right fine cell to upper left coarse cell F(f,1) = u(f,1) * velocity(f,1) * h(f)/2 = 36 * 0.5 * 1/18 = 1
   * The transfered volume from this cell s(f,1) = F(f,1) * deltaT(f) = 1 * 1/3 = 1/3
   * The transfered volume to the coarse cell from this fine cell s(c,1,1) = u(c,1) * velocity(c,1) * h(f)/2 * deltaT(f) = 1/108
   * Correction for the lower left coarse cell from this fine cell delta(c,1,1) = 1/hc^2 * (s(f,1) - s(c,1,1))
   *    = 4 * (1/3 - 1/108) = 35/27
   * New value for the lower left coarse cell therefore is u'(c,1) = u(c,1) + delta(c,1,1) = 1 + 35/27 = 62/27
   *
   * Flux from upper right fine cell to upper left coarse cell F(f,2) = u(f,2) * velocity(f,2) * h(f) = 3
   * The transfered volume from this cell s(f,2) = F(f,2) * deltaT(f) = 3 * 1/3 = 1
   * The transfered volume to the coarse cell from this fine cell s(c,1,2) = u'(c,1) * velocity(c,1) * h(f) * deltaT(f) = 62/27 * 0.5 * 1/9 * 1/3 = 31/729
   * Correction for the lower left coarse cell from this fine cell delta(c,0,1) = 1/hc^2 * (s(f,1) - s(c,1,1))
   *    = 4 * (1/3 - 31/729) = 2792/729
   * New value for the lower left coarse cell therefore is u''(c,1) = u'(c,1) + delta(c,1,2) = 62/27 + 2792/729 = 4466/729
   */
  void testFluxCorrection();

  /**
   * Tests the interpolation in time, when copying ghostlayer data from
   * one patch to another.
   *
   * The source Patch of size 1,1 with 2x2 cells has the values
   * uNew:
   * 30 40
   * 10 20
   *
   * uOld:
   * 3 4
   * 1 2
   *
   * and spans the time interval [0, 1].
   *
   * The destination patch is located on the right of the source patch
   * and has a ghostlayer width of 1. UNew resides on time 1/3. Thus,
   * we expect the following values in the ghostlayer:
   *
   * 4*2/3 + 40*1/3 = 320/3
   * 2*2/3 + 20*1/3 = 80/3
   */
  void testInterpolationInTime();

  /**
   * Restrict to a patch where the upper
   * and lower bounds of neighboring ghostlayers
   * overlap. Thus, the first returned area holds
   * the complete domain, while the rest is empty.
   */
  void testRestrictionWithOverlappingBounds();

  /**
   * Restrict to a patch of size 1.0, where the upper
   * and lower ghostlayers overlap by a value of 0.2.
   * The patch holds 10 cells in each dimension. Hence,
   * the areas should look like the following:
   *
   * Number  Offset  Size
   * 0       [0, 0]  [2, 10]
   * 1       [8, 0]  [2, 10]
   * 2       [2, 0]  [6, 2]
   * 3       [2, 8]  [6, 2]
   */
  void testPartialRestrictionAreas();

  /**
   * Restrict to a patch of size 1/27 with a subdivision
   * factor of 6, where the upper bound overlap by 0 cells
   * in x0 direction and 1 cell in x1 direction and the
   * lower bounds are still set to double::max().
   * The areas should look like:
   *
   * Number  Offset  Size
   * 0       [0, 0]  [0, 6]
   * 1       [6, 0]  [0, 6]
   * 2       [0, 0]  [6, 1]
   * 3       [0, 6]  [6, 0]
   */
  void testPartialRestrictionAreasWithInfiniteLowerBounds();

  /**
   * Tests whether the correct patches are traversed in 2d
   * when looking for patches adjacent via faces.
   */
  void testFaceAdjacentPatchTraversal2D();

  /**
   * Tests whether the correct patches are traversed in 2d
   * when looking for patches adjacent via edges (or vertices
   * in 2d).
   */
  void testEdgeAdjacentPatchTraversal2D();

public:
  GhostLayerCompositorTest();

  virtual ~GhostLayerCompositorTest();

  virtual void run();

  void virtual setUp();
};

/**
 * Helper class for testing FaceAdjacentPatchTraversal.
 */
class peanoclaw::tests::TestFaceAdjacentPatchTraversalFunctor {

  public:
    std::vector<std::vector<int> > _calls;

    void operator()(
      peanoclaw::Patch& patch1,
      int               index1,
      peanoclaw::Patch& patch2,
      int               index2,
      tarch::la::Vector<DIMENSIONS, int> direction
    );
};

/**
 * Helper class for testing EdgeAdjacentPatchTraversal.
 */
class peanoclaw::tests::TestEdgeAdjacentPatchTraversalFunctor {
  public:
    std::vector<std::vector<int> > _calls;

    void operator()(
      peanoclaw::Patch&                  patch1,
      int                                index1,
      peanoclaw::Patch&                  patch2,
      int                                index2,
      tarch::la::Vector<DIMENSIONS, int> direction
    );
};

#endif /* PEANO_APPLICATIONS_PEANOCLAW_TESTS_GHOSTLAYERCOMPOSITORTEST_H_ */
