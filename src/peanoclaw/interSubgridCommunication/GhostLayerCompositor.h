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
#include "peanoclaw/statistics/SubgridStatistics.h"

#include "peano/utils/Globals.h"

#include "tarch/logging/Log.h"

namespace peanoclaw {

  class Numerics;
  class Patch;

  namespace interSubgridCommunication {
    class GhostLayerCompositor;

    class FillGhostlayerFaceFunctor;
    class FillGhostlayerEdgeFunctor;
    class FillGhostlayerCornerFunctor;
    class UpdateNeighborTimeFunctor;
    class FluxCorrectionFunctor;
    class UpdateGhostlayerBoundsFaceFunctor;
    class UpdateGhostlayerBoundsEdgeFunctor;
    class UpdateGhostlayerBoundsCornerFunctor;
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
  friend class peanoclaw::interSubgridCommunication::FillGhostlayerFaceFunctor;
  friend class peanoclaw::interSubgridCommunication::FillGhostlayerEdgeFunctor;
  friend class peanoclaw::interSubgridCommunication::FillGhostlayerCornerFunctor;
  friend class peanoclaw::interSubgridCommunication::UpdateGhostlayerBoundsFaceFunctor;
  friend class peanoclaw::interSubgridCommunication::UpdateGhostlayerBoundsEdgeFunctor;
  friend class peanoclaw::interSubgridCommunication::UpdateGhostlayerBoundsCornerFunctor;

  Patch* _patches;

  int _level;

  peanoclaw::Numerics& _numerics;

  bool _useDimensionalSplittingExtrapolation;

  /**
   * Updates the maximum allowed timestep size for a patch dependent on the
   * time interval spanned by a neighboring patch.
   *
   */
  void updateNeighborTime(int updatedPatchIndex, int neighborPatchIndex);

  /**
   * Updates the appropriate ghostlayer bound for the given pair of patches.
   * These patches are assumed to share a face (or an edge in 2D).
   */
  void updateGhostlayerBound(int updatedPatchIndex, int neighborPatchIndex, int dimension);

  /**
   * Updates the appropriate ghostlayer bound for the given pair of patches.
   * These patches are assumed to share an edge (or a vertex in 2D) or a vertex (in 3D),
   * therefore, more than one ghostlayer bound is affected by this updating, but only the
   * required bounds are updated.
   *
   * The method checks for each dimension whether the already set bounds cover the given
   * intersection. If they do, no further action is required. If they don't one of the
   * bounds is set.
   */
  void updateGhostlayerBound(int updatedPatchIndex, int neighborPatchIndex, tarch::la::Vector<DIMENSIONS, int> direction);

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

  /**
   * Fills the manifolds of the given dimensionality.
   */
  double fillGhostlayerManifolds(int destinationSubgridIndex, bool fillFromNeighbor, int dimensionality);

  /**
   * Updates the neighbor time for the neighbors, corresponding to ghe given manifolds of the ghostlayers.
   */
  void updateNeighborTimeForManifolds(int destinationSubgridIndex, int dimensionality);

  /**
   * Fills the manifolds of the given dimensionality either from the neighbors or by extrapolating
   * the already set pars of the subgrids.
   */
  void fillOrExtrapolateGhostlayerAndUpdateNeighborTime(int destinationSubgridIndex);

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
    bool useDimensionalSplittingExtrapolation
  );

  ~GhostLayerCompositor();

  /**
   * Fills the ghostlayers of the given patches as far as possible. Also the
   * maximum timestep size for the patches is adjusted.
   */
  void fillGhostLayersAndUpdateNeighborTimes(int destinationSubgridIndex = -1);

  /**
   * Updates the maximum timesteps for the given patches depending on the time
   * interval spanned by the other patches and the maximum neighbor time interval.
   */
//  void updateNeighborTimes();

  /**
   * Updates the ghostlayer bounds, i.e. how far a patch is overlapped by
   * neighboring ghostlayers.
   */
  void updateGhostlayerBounds();

  /**
   * Applies the flux correction on all adjacent coarse patches.
   */
  void applyFluxCorrection(int sourceSubgridIndex);
};

#endif /* PEANO_APPLICATIONS_PEANOCLAW_GHOSTLAYERCOMPOSITOR_H_ */
