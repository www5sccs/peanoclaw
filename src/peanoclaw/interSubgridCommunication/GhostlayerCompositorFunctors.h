/*
 * GhostlayerCompositorFunctors.h
 *
 *  Created on: Sep 5, 2013
 *      Author: unterweg
 */

#ifndef PEANOCLAW_INTERSUBGRIDCOMMUNICATION_GHOSTLAYERCOMPOSITORFUNCTORS_H_
#define PEANOCLAW_INTERSUBGRIDCOMMUNICATION_GHOSTLAYERCOMPOSITORFUNCTORS_H_

#include "peano/utils/Globals.h"
#include "tarch/la/Vector.h"

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

    class ExtrapolateGhostlayerEdgeFunctor;
    class ExtrapolateGhostlayerCornerFunctor;
  }
}

/**
 * Functor for filling ghostlayer faces between two patches.
 */
class peanoclaw::interSubgridCommunication::FillGhostlayerFaceFunctor {
  private:
    GhostLayerCompositor& _ghostlayerCompositor;
    int                   _destinationPatchIndex;

  public:
    FillGhostlayerFaceFunctor(
      GhostLayerCompositor& ghostlayerCompositor,
      int                   destinationPatchIndex
    );

    void operator() (
      peanoclaw::Patch&                         patch1,
      int                                       index1,
      peanoclaw::Patch&                         patch2,
      int                                       index2,
      const tarch::la::Vector<DIMENSIONS, int>& direction
    );
};

/**
 * Functor for filling ghostlayer edges between two patches.
 */
class peanoclaw::interSubgridCommunication::FillGhostlayerEdgeFunctor {
  private:
    GhostLayerCompositor& _ghostlayerCompositor;
    int                   _destinationPatchIndex;

  public:
    FillGhostlayerEdgeFunctor(
      GhostLayerCompositor& ghostlayerCompositor,
      int                   destinationPatchIndex
    );

    void operator() (
      peanoclaw::Patch&                         patch1,
      int                                       index1,
      peanoclaw::Patch&                         patch2,
      int                                       index2,
      const tarch::la::Vector<DIMENSIONS, int>& direction
    );
};

/**
 * Functor for filling a ghostlayer between subgrids
 * adjacent over a vertex (not used in 2D).
 */
class peanoclaw::interSubgridCommunication::FillGhostlayerCornerFunctor {
  private:
  GhostLayerCompositor& _ghostlayerCompositor;
  int                   _destinationPatchIndex;

  public:
  FillGhostlayerCornerFunctor(
    GhostLayerCompositor& ghostlayerCompositor,
    int                   destinationPatchIndex
  );

  void operator() (
    peanoclaw::Patch&                         patch1,
    int                                       index1,
    peanoclaw::Patch&                         patch2,
    int                                       index2,
    const tarch::la::Vector<DIMENSIONS, int>& direction
  );
};

/**
 * Functor for updating the neighbor time constraint in
 * adjacent patches.
 */
class peanoclaw::interSubgridCommunication::UpdateNeighborTimeFunctor {
  private:
    GhostLayerCompositor& _ghostlayerCompositor;

  public:
    UpdateNeighborTimeFunctor(
      GhostLayerCompositor& ghostlayerCompositor
    );

    void operator() (
      peanoclaw::Patch&                         patch1,
      int                                       index1,
      peanoclaw::Patch&                         patch2,
      int                                       index2,
      const tarch::la::Vector<DIMENSIONS, int>& direction
    );
};

/**
 * Functor for correcting the flux on an adjacent
 * coarse patch.
 */
class peanoclaw::interSubgridCommunication::FluxCorrectionFunctor {
  private:
    Numerics& _numerics;

  public:
    FluxCorrectionFunctor(
      Numerics& numerics
    );

    void operator() (
      peanoclaw::Patch&                         patch1,
      int                                       index1,
      peanoclaw::Patch&                         patch2,
      int                                       index2,
      const tarch::la::Vector<DIMENSIONS, int>& direction
    );
};

/**
 * Functor for filling a ghostlayer between subgrids
 * adjacent over a face (or an edge in 2D).
 */
class peanoclaw::interSubgridCommunication::UpdateGhostlayerBoundsFaceFunctor {
  private:
  GhostLayerCompositor& _ghostlayerCompositor;

  public:
  UpdateGhostlayerBoundsFaceFunctor(
    GhostLayerCompositor& ghostlayerCompositor
  );

  void operator() (
    peanoclaw::Patch&                         patch1,
    int                                       index1,
    peanoclaw::Patch&                         patch2,
    int                                       index2,
    const tarch::la::Vector<DIMENSIONS, int>& direction
  );
};

/**
 * Functor for filling a ghostlayer between subgrids
 * adjacent over an edge (or vertex in 2D).
 */
class peanoclaw::interSubgridCommunication::UpdateGhostlayerBoundsEdgeFunctor {
  private:
  GhostLayerCompositor& _ghostlayerCompositor;

  public:
  UpdateGhostlayerBoundsEdgeFunctor(
    GhostLayerCompositor& ghostlayerCompositor
  );

  void operator() (
    peanoclaw::Patch&                         patch1,
    int                                       index1,
    peanoclaw::Patch&                         patch2,
    int                                       index2,
    const tarch::la::Vector<DIMENSIONS, int>& direction
  );
};

/**
 * Functor for filling a ghostlayer between subgrids
 * adjacent over an edge (or vertex in 2D).
 */
class peanoclaw::interSubgridCommunication::UpdateGhostlayerBoundsCornerFunctor {
  private:
    GhostLayerCompositor& _ghostlayerCompositor;

  public:
    UpdateGhostlayerBoundsCornerFunctor(
      GhostLayerCompositor& ghostlayerCompositor
    );

    void operator() (
      peanoclaw::Patch&                         patch1,
      int                                       index1,
      peanoclaw::Patch&                         patch2,
      int                                       index2,
      const tarch::la::Vector<DIMENSIONS, int>& direction
    );
};

#endif /* PEANOCLAW_INTERSUBGRIDCOMMUNICATION_GHOSTLAYERCOMPOSITORFUNCTORS_H_ */
