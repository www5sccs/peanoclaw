/*
 * DefaultFluxCorrection.h
 *
 *  Created on: Mar 7, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_INTERSUBGRIDCOMMUNICATION_DEFAULTFLUXCORRECTION_H_
#define PEANOCLAW_INTERSUBGRIDCOMMUNICATION_DEFAULTFLUXCORRECTION_H_

#include "peanoclaw/interSubgridCommunication/FluxCorrection.h"

#include "peano/utils/Dimensions.h"
#include "tarch/la/Vector.h"

namespace peanoclaw {
  namespace interSubgridCommunication {
    class DefaultFluxCorrection;
  }

  class Patch;
}

class peanoclaw::interSubgridCommunication::DefaultFluxCorrection
    : public peanoclaw::interSubgridCommunication::FluxCorrection {

  private:
    /**
     * Returns the area of the region where the two given
     * patches overlap.
     * This overload projects the patches along the given projection axis
     * and just calculates the overlap in this projection.
     *
     * In 2d this refers to a projection to one-dimensional intervals and
     * the intersection between these intervals.
     */
    double calculateOverlappingArea(
      tarch::la::Vector<DIMENSIONS, double> position1,
      tarch::la::Vector<DIMENSIONS, double> size1,
      tarch::la::Vector<DIMENSIONS, double> position2,
      tarch::la::Vector<DIMENSIONS, double> size2,
      int projectionAxis
    ) const;

  public:

    virtual ~DefaultFluxCorrection();

    /**
     * Applying the default flux correction on the coarse patch.
     */
    void applyCorrection(
      const Patch& finePatch,
      Patch& coarsePatch,
      int dimension,
      int direction
    ) const;
};

#endif /* PEANOCLAW_INTERSUBGRIDCOMMUNICATION_DEFAULTFLUXCORRECTION_H_ */
