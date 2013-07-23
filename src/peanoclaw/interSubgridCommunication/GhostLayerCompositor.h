/*
 * GhostLayerCompositor.h
 *
 *  Created on: Feb 16, 2012
 *      Author: Kristof Unterweger
 */

#ifndef PEANO_APPLICATIONS_PEANOCLAW_GHOSTLAYERCOMPOSITOR_H_
#define PEANO_APPLICATIONS_PEANOCLAW_GHOSTLAYERCOMPOSITOR_H_

#include "peanoclaw/interSubgridCommunication/Interpolation.h"
#include "peanoclaw/interSubgridCommunication/FluxCorrection.h"
#include "peanoclaw/pyclaw/PyClaw.h"
#include "peanoclaw/records/CellDescription.h"

#include "peano/utils/Globals.h"

#include "tarch/logging/Log.h"

namespace peanoclaw {

  class Numerics;
  class Patch;

  namespace interSubgridCommunication {
    class GhostLayerCompositor;
  }

  namespace tests {
    class GhostLayerCompositorTest;
  }
} /* namespace peanoclaw */

/**
 * In this class the functionality for setting the ghostlayer
 * data is implemented as it is seen from a vertex. The numbering
 * of the patches array is done like the setting of the adjacent
 * indices on the arrays, thus a cell writes its index in the
 * i-th adjacent vertex in position i. This leads to the
 * following numbering:
 *
 * \code
 * patch1   patch0
 *     vertex
 * patch3   patch2
 * \endcode
 *
 * Besides the mere filling of the ghostlayers, the computation
 * of the "maximum neighbor time interval" and of the
 * "minimal neighbor time" is done in this class.
 *
 * @see peanoclaw::Patch for further details.
 *
 *
 */
class peanoclaw::interSubgridCommunication::GhostLayerCompositor
{

private:
  /**
   * Logging device.
   */
  static tarch::logging::Log _log;

  friend class peanoclaw::tests::GhostLayerCompositorTest;

  Patch* _patches;

  int _level;

  peanoclaw::Numerics& _numerics;

  bool _useDimensionalSplittingOptimization;

  /**
   * Performs a copy of a block from the uNew of one cell to the ghostlayer of uOld
   * in a neighbouring cell.
   */
  void copyGhostLayerDataBlock(
    const tarch::la::Vector<DIMENSIONS, int>&    size,
    const tarch::la::Vector<DIMENSIONS, int>&    sourceOffset,
    const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
    const peanoclaw::Patch& source,
    peanoclaw::Patch&       destination);

  /**
   * Updates the maximum allowed timestep size for a patch dependent on the
   * time interval spanned by a neighboring patch.
   *
   */
  void updateNeighborTime(int updatedPatchIndex, int neighborPatchIndex);

  //Ghost layers, i.e. copy from one patch to another.
//  void fillLeftGhostLayer();
//
//  void fillLowerLeftGhostLayer();
//
//  void fillLowerGhostLayer();
//
//  void fillUpperLeftGhostLayer();
//
//  void fillUpperGhostLayer();
//
//  void fillUpperRightGhostLayer();
//
//  void fillRightGhostLayer();
//
//  void fillLowerRightGhostLayer();

  /**
   * Updates the lower ghostlayer bound for the given pair of patches.
   */
  void updateLowerGhostlayerBound(int updatedPatchIndex, int neighborPatchIndex, int dimension);

  /**
   * Updates the upper ghostlayer bound for the given pair of patches.
   */
  void updateUpperGhostlayerBound(int updatedPatchIndex, int neighborPatchIndex, int dimension);

  /**
   * Estimates, whether ghostlayer data should be transfered
   * from the source to the destination patch.
   */
  bool shouldTransferGhostlayerData(Patch& sourcePatch, Patch& destinationPatch);

public:

  /**
   * Constructor for a GhostLayerCompositor. GhostLayerCompositors are vertex-centered.
   * Therefore, the given patches are the $2^d$ patches adjacent to the current vertex
   * and the given level is the level on which the vertex is residing.
   */
  GhostLayerCompositor(
    peanoclaw::Patch patches[TWO_POWER_D],
    int level,
    peanoclaw::Numerics& numerics,
    bool useDimensionalSplittingOptimization
  );

  ~GhostLayerCompositor();

  /**
   * Fills the ghostlayers of the given patches as far as possible. Also the
   * maximum timestep size for the patches is adjusted.
   */
  void fillGhostLayers(int destinationPatchIndex = -1);

  /**
   * Updates the maximum timesteps for the given patches depending on the time
   * interval spanned by the other patches and the maximum neighbor time interval.
   */
  void updateNeighborTimes();

  /**
   * Updates the ghostlayer bounds, i.e. how far a patch is overlapped by
   * neighboring ghostlayers.
   */
  void updateGhostlayerBounds();

  /**
   * Applies the flux correction on all adjacent coarse patches.
   */
  void applyFluxCorrection();

  /**
   * Functor for filling ghostlayer faces between two patches.
   */
  class FillGhostlayerFaceFunctor {
    private:
      GhostLayerCompositor& _ghostlayerCompositor;
      int                   _destinationPatchIndex;

    public:
      FillGhostlayerFaceFunctor(
        GhostLayerCompositor& ghostlayerCompositor,
        int                   destinationPatchIndex
      );

      void operator() (
        peanoclaw::Patch&                  patch1,
        int                                index1,
        peanoclaw::Patch&                  patch2,
        int                                index2,
        tarch::la::Vector<DIMENSIONS, int> direction
      );
  };

  /**
   * Functor for filling ghostlayer edges between two patches.
   */
  class FillGhostlayerEdgeFunctor {
    private:
      GhostLayerCompositor& _ghostlayerCompositor;
      int                   _destinationPatchIndex;

    public:
      FillGhostlayerEdgeFunctor(
        GhostLayerCompositor& ghostlayerCompositor,
        int                   destinationPatchIndex
      );

      void operator() (
        peanoclaw::Patch&                  patch1,
        int                                index1,
        peanoclaw::Patch&                  patch2,
        int                                index2,
        tarch::la::Vector<DIMENSIONS, int> direction
      );
  };

  /**
   * Functor for updating the neighbor time constraint in
   * adjacent patches.
   */
  class UpdateNeighborTimeFunctor {
    private:
      GhostLayerCompositor& _ghostlayerCompositor;

    public:
      UpdateNeighborTimeFunctor(
        GhostLayerCompositor& ghostlayerCompositor
      );

      void operator() (
        peanoclaw::Patch&                  patch1,
        int                                index1,
        peanoclaw::Patch&                  patch2,
        int                                index2,
        tarch::la::Vector<DIMENSIONS, int> direction
      );
  };

  /**
   * Functor for correcting the flux on an adjacent
   * coarse patch.
   */
  class FluxCorrectionFunctor {
    private:
      Numerics& _numerics;

    public:
      FluxCorrectionFunctor(
        Numerics& numerics
      );

      void operator() (
        peanoclaw::Patch&                  patch1,
        int                                index1,
        peanoclaw::Patch&                  patch2,
        int                                index2,
        tarch::la::Vector<DIMENSIONS, int> direction
      );
  };

  class UpdateGhostlayerBoundsFaceFunctor {
    private:
    GhostLayerCompositor& _ghostlayerCompositor;

    public:
    UpdateGhostlayerBoundsFaceFunctor(
      GhostLayerCompositor& ghostlayerCompositor
    );

    void operator() (
      peanoclaw::Patch&                  patch1,
      int                                index1,
      peanoclaw::Patch&                  patch2,
      int                                index2,
      tarch::la::Vector<DIMENSIONS, int> direction
    );
  };

  class UpdateGhostlayerBoundsEdgeFunctor {
    private:
    GhostLayerCompositor& _ghostlayerCompositor;

    public:
    UpdateGhostlayerBoundsEdgeFunctor(
      GhostLayerCompositor& ghostlayerCompositor
    );

    void operator() (
      peanoclaw::Patch&                  patch1,
      int                                index1,
      peanoclaw::Patch&                  patch2,
      int                                index2,
      peanoclaw::Patch&                  faceNeighbor1,
      int                                indexFaceNeighbor1,
      peanoclaw::Patch&                  faceNeighbor2,
      int                                indexFaceNeighbor2,
      tarch::la::Vector<DIMENSIONS, int> direction
    );
  };
};

#endif /* PEANO_APPLICATIONS_PEANOCLAW_GHOSTLAYERCOMPOSITOR_H_ */
