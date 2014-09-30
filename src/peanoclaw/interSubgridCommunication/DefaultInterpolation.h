/*
 * DefaultInterpolation.h
 *
 *  Created on: Mar 7, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_INTERSUBGRIDCOMMUNICATION_DEFAULTINTERPOLATION_H_
#define PEANOCLAW_INTERSUBGRIDCOMMUNICATION_DEFAULTINTERPOLATION_H_

#include "peanoclaw/interSubgridCommunication/Interpolation.h"
#include "peanoclaw/Patch.h"

#include "peano/utils/Dimensions.h"
#include "tarch/la/Vector.h"

namespace peanoclaw {
  namespace interSubgridCommunication {
    class DefaultInterpolation;

    template<int NumberOfUnknowns>
    class DefaultInterpolationTemplate;
  }
}

/**
 * Default implementation for interpolation grid values from coarse
 * to fine subgrids.
 */
template<int NumberOfUnknowns>
class peanoclaw::interSubgridCommunication::DefaultInterpolationTemplate {
  private:
    /**
     * Logging device for the trace macros.
     */
    static tarch::logging::Log  _log;

    int _signLookupTable[TWO_POWER_D_TIMES_D];

  public:
    DefaultInterpolationTemplate();

    /**
     * @see peanoclaw::interSubgridCommunication::Interpolation
     *
     * TODO unterweg: useTimeUNewOrTimeUOld is only used because the
     * restriction now is done to the minimalNeighborTimeInterval. However,
     * during grid refinement the interpolation has to take the actual currentTime
     * and timestepSize of the subgrid into account and not timeUNew and timeUOld
     * from the TimeIntervals class. This could be circumvented if the restriction
     * is carried out in the ascend event and can be done to the minimalFineGrid
     * interval which becomes the new currentTime and timestepSize of the virtual
     * subgrid anyway.
     */
    void interpolateSolution (
      const tarch::la::Vector<DIMENSIONS, int>&    destinationSize,
      const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
      peanoclaw::Patch& source,
      peanoclaw::Patch& destination,
      bool interpolateToUOld,
      bool interpolateToCurrentTime,
      bool useTimeUNewOrTimeUOld
    );
};

/**
 * Implements a d-linear interpolation of grid values.
 */
class peanoclaw::interSubgridCommunication::DefaultInterpolation : public peanoclaw::interSubgridCommunication::Interpolation {
  public:
    void interpolateSolution (
      const tarch::la::Vector<DIMENSIONS, int>&    destinationSize,
      const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
      peanoclaw::Patch& source,
      peanoclaw::Patch& destination,
      bool interpolateToUOld,
      bool interpolateToCurrentTime,
      bool useTimeUNewOrTimeUOld
    ) {
      switch(source.getUnknownsPerSubcell()) {
        case 1:
          {
            peanoclaw::interSubgridCommunication::DefaultInterpolationTemplate<1> transfer1;
            transfer1.interpolateSolution(destinationSize, destinationOffset, source, destination, interpolateToUOld, interpolateToCurrentTime, useTimeUNewOrTimeUOld);
          }
          break;
        case 2:
          {
            peanoclaw::interSubgridCommunication::DefaultInterpolationTemplate<2> transfer2;
            transfer2.interpolateSolution(destinationSize, destinationOffset, source, destination, interpolateToUOld, interpolateToCurrentTime, useTimeUNewOrTimeUOld);
          }
          break;
        case 3:
          {
            peanoclaw::interSubgridCommunication::DefaultInterpolationTemplate<3> transfer3;
            transfer3.interpolateSolution(destinationSize, destinationOffset, source, destination, interpolateToUOld, interpolateToCurrentTime, useTimeUNewOrTimeUOld);
          }
          break;
        case 4:
          {
            peanoclaw::interSubgridCommunication::DefaultInterpolationTemplate<4> transfer4;
            transfer4.interpolateSolution(destinationSize, destinationOffset, source, destination, interpolateToUOld, interpolateToCurrentTime, useTimeUNewOrTimeUOld);
          }
          break;
        case 5:
          {
            peanoclaw::interSubgridCommunication::DefaultInterpolationTemplate<5> transfer5;
            transfer5.interpolateSolution(destinationSize, destinationOffset, source, destination, interpolateToUOld, interpolateToCurrentTime, useTimeUNewOrTimeUOld);
          }
          break;
        case 6:
          {
            peanoclaw::interSubgridCommunication::DefaultInterpolationTemplate<6> transfer6;
            transfer6.interpolateSolution(destinationSize, destinationOffset, source, destination, interpolateToUOld, interpolateToCurrentTime, useTimeUNewOrTimeUOld);
          }
          break;
        case 7:
          {
            peanoclaw::interSubgridCommunication::DefaultInterpolationTemplate<7> transfer7;
            transfer7.interpolateSolution(destinationSize, destinationOffset, source, destination, interpolateToUOld, interpolateToCurrentTime, useTimeUNewOrTimeUOld);
          }
          break;
        case 8:
          {
            peanoclaw::interSubgridCommunication::DefaultInterpolationTemplate<8> transfer8;
            transfer8.interpolateSolution(destinationSize, destinationOffset, source, destination, interpolateToUOld, interpolateToCurrentTime, useTimeUNewOrTimeUOld);
          }
          break;
        case 9:
          {
            peanoclaw::interSubgridCommunication::DefaultInterpolationTemplate<9> transfer9;
            transfer9.interpolateSolution(destinationSize, destinationOffset, source, destination, interpolateToUOld, interpolateToCurrentTime, useTimeUNewOrTimeUOld);
          }
          break;
        case 10:
          {
            peanoclaw::interSubgridCommunication::DefaultInterpolationTemplate<10> transfer10;
            transfer10.interpolateSolution(destinationSize, destinationOffset, source, destination, interpolateToUOld, interpolateToCurrentTime, useTimeUNewOrTimeUOld);
          }
          break;
        default:
          assertionFail("Number of unknowns " << source.getUnknownsPerSubcell() << " not supported!");
      }
    }
};

#include "peanoclaw/interSubgridCommunication/DefaultInterpolation.cpph"

#endif /* PEANOCLAW_INTERSUBGRIDCOMMUNICATION_DEFAULTINTERPOLATION_H_ */
