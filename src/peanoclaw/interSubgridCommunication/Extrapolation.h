/*
 * Extrapolation.h
 *
 *  Created on: Jul 8, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_INTERSUBGRIDCOMMUNICATION_EXTRAPOLATION_H_
#define PEANOCLAW_INTERSUBGRIDCOMMUNICATION_EXTRAPOLATION_H_

#include "peanoclaw/Patch.h"

namespace peanoclaw {
  namespace interSubgridCommunication {
    class CornerExtrapolation;
    class Extrapolation;
  }
}

/**
 * Functor for traversing the corners of a patch and interpolating the ghostlayer values.
 */
class peanoclaw::interSubgridCommunication::CornerExtrapolation {
  private:
    double _maximumGradient;

  public:
    void operator()(
      peanoclaw::Patch& patch,
      const peanoclaw::Area& area,
      const tarch::la::Vector<DIMENSIONS,int> cornerIndex
    );

    double getMaximumGradient() const;
};

/**
 * This class contains functionality to extrapolate parts of a
 * ghostlayer to other parts. This is used to fill edges or
 * corners of a ghostlayer to avoid dependency of some adjacent
 * patches.
 */
class peanoclaw::interSubgridCommunication::Extrapolation {

  private:
    peanoclaw::Patch& _patch;

    /**
     * Extrapolates the values from the ghostlayer faces to the edges.
     * In 2D this operation does nothing.
     */
    double extrapolateEdges();

    /**
     * Extrapolates the values from the ghostlayer faces/edges to the corners.
     */
    double extrapolateCorners();

  public:
    Extrapolation(Patch& patch);

    /**
     * Extrapolates the ghostlayer from the faces to the edges or corners depending
     * on the dimensionality.
     *
     * Returns an estimate of the maximum gradient used for extrapolation.
     */
    double extrapolateGhostlayer();
};


#endif /* PEANOCLAW_INTERSUBGRIDCOMMUNICATION_EXTRAPOLATION_H_ */
