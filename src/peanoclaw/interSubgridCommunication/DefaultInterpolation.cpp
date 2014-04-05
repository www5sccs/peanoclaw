/*
 * DefaultInterpolation.cpp
 *
 *  Created on: Mar 7, 2013
 *      Author: kristof
 */

#include "peanoclaw/interSubgridCommunication/DefaultInterpolation.h"

#include "peanoclaw/Patch.h"
#include "peano/utils/Loop.h"

#ifdef Parallel
#include "tarch/parallel/Node.h"
#endif

tarch::logging::Log peanoclaw::interSubgridCommunication::DefaultInterpolation::_log( "peanoclaw::interSubgridCommunication::DefaultInterpolation" ); 

void peanoclaw::interSubgridCommunication::DefaultInterpolation::interpolate(
  const tarch::la::Vector<DIMENSIONS, int>&    destinationSize,
  const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
  const peanoclaw::Patch& source,
  peanoclaw::Patch&        destination,
  bool interpolateToUOld,
  bool interpolateToCurrentTime,
  bool useTimeUNewOrTimeUOld
) {
  logTraceInWith4Arguments("", destinationSize, destinationOffset, source.toString(), destination.toString());
  assertionEquals(source.getUnknownsPerSubcell(), destination.getUnknownsPerSubcell());

  //TODO unterweg debug
//  std::cout << "Interpolated cells:" << tarch::la::volume(destinationSize) << std::endl;

  //Factor for interpolation in time
  double timeFactor;
  if(tarch::la::equals(source.getTimeIntervals().getTimestepSize(), 0.0)) {
    timeFactor = 1.0;
  } else {
    if(useTimeUNewOrTimeUOld) {
      if(interpolateToCurrentTime) {
        timeFactor = (destination.getTimeIntervals().getTimeUOld() - source.getTimeIntervals().getTimeUOld()) / (source.getTimeIntervals().getTimeUNew() - source.getTimeIntervals().getTimeUOld());
      } else {
        timeFactor = (destination.getTimeIntervals().getTimeUNew() - source.getTimeIntervals().getTimeUOld()) / (source.getTimeIntervals().getTimeUNew() - source.getTimeIntervals().getTimeUOld());
      }
    } else {
      if(interpolateToCurrentTime) {
        timeFactor = (destination.getTimeIntervals().getCurrentTime() - source.getTimeIntervals().getCurrentTime()) / (source.getTimeIntervals().getCurrentTime() + source.getTimeIntervals().getTimestepSize() - source.getTimeIntervals().getCurrentTime());
      } else {
        timeFactor = (destination.getTimeIntervals().getCurrentTime() + destination.getTimeIntervals().getTimestepSize() - source.getTimeIntervals().getCurrentTime()) / (source.getTimeIntervals().getCurrentTime() + source.getTimeIntervals().getTimestepSize() - source.getTimeIntervals().getCurrentTime());
      }
    }
  }

  //TODO This should be guaranteed by the timestepping criterion but may be violated at the moment
  // when coarsening the grid. Fixed?
  timeFactor = std::max(0.0, std::min(1.0, timeFactor));

  assertion(!tarch::la::smaller(timeFactor, 0.0) && !tarch::la::greater(timeFactor, 1.0));

  //Clear ghostlayer
  destination.clearRegion(destinationOffset, destinationSize, interpolateToUOld);

  //Prefetch data
  tarch::la::Vector<DIMENSIONS, double> sourcePosition = source.getPosition();
  tarch::la::Vector<DIMENSIONS, int> sourceSubdivisionFactor = source.getSubdivisionFactor();
  tarch::la::Vector<DIMENSIONS, double> sourceSubcellSize = source.getSubcellSize();
  tarch::la::Vector<DIMENSIONS, double> inverseSourceSubcellSize = tarch::la::invertEntries(sourceSubcellSize);
  int unknownsPerSubcell = destination.getUnknownsPerSubcell();

  #ifdef Asserts
  for(int d = 0; d < DIMENSIONS; d++) {
    assertionMsg(
        tarch::la::smallerEquals(source.getPosition()(d) - source.getGhostlayerWidth() * destination.getSubcellSize()(d),
            destination.getPosition()(d) + destination.getSubcellSize()(d) * destinationOffset(d))
        && tarch::la::greaterEquals(source.getPosition()(d) + source.getSize()(d) + source.getGhostlayerWidth() * destination.getSubcellSize()(d),
            destination.getPosition()(d) + destination.getSubcellSize()(d) * (destinationOffset(d) + destinationSize(d))),
        "The kernel of the destination block must be totally included in the source patch."
        << "\nd=" << d
        << "\nSource: " << source.toString() << "\nDestination: " << destination.toString()
        << "\ndestinationOffset: " << destinationOffset << "\ndestinationSize: " << destinationSize);
  }
  #endif

  //Interpolate/Extrapolate
  dfor(subcellIndex, destinationSize) {
    //Map to global space
    tarch::la::Vector<DIMENSIONS, int> subcellIndexInDestinationPatch = subcellIndex + destinationOffset;
    tarch::la::Vector<DIMENSIONS, double> subcellPositionInSourcePatch = destination.getSubcellCenter(subcellIndexInDestinationPatch);
    tarch::la::Vector<DIMENSIONS, double> destinationSubcellCenter = destination.getSubcellCenter(subcellIndexInDestinationPatch);

    //Map to local space of source patch
    subcellPositionInSourcePatch -= (sourcePosition + sourceSubcellSize * 0.5);
    tarch::la::Vector<DIMENSIONS, int> subcellIndexInSourcePatchUOld;
    for(int d = 0; d < DIMENSIONS; d++) {
      subcellIndexInSourcePatchUOld(d) = std::floor(subcellPositionInSourcePatch(d) / sourceSubcellSize(d));
    }
    for(int d = 0; d < DIMENSIONS; d++) {
      //Correct source patch subcell index, if on the boundary -> switch to extrapolation
      subcellIndexInSourcePatchUOld(d) = std::max(0, std::min(sourceSubdivisionFactor(d)-2,subcellIndexInSourcePatchUOld(d)));
    }

    logDebug("", "Mapping source-point " << subcellPositionInSourcePatch
        << " in source-cell " << subcellIndexInSourcePatchUOld
        << " to destination-cell " << (subcellIndex + destinationOffset));

    dfor2(offset)
      tarch::la::Vector<DIMENSIONS, int> neighborIndexInSourcePatch = subcellIndexInSourcePatchUOld + offset;
      tarch::la::Vector<DIMENSIONS, double> neighborPositionInSourcePatch = source.getSubcellCenter(neighborIndexInSourcePatch);

      //Calculate factor for spatial interpolation. sign is used to account for extrapolation
      //if destination cell center is outside the source cell.
      double spatialFactor = 1.0;
      for(int d = 0; d < DIMENSIONS; d++) {
        double sign = 1 - 2*offset(d);
        spatialFactor *= 1.0 -
                         (
                           sign *
                           (
                             destinationSubcellCenter(d) - neighborPositionInSourcePatch(d)
                           )
                         )
                         * inverseSourceSubcellSize(d);
      }

      int linearSourceUOldIndex = source.getLinearIndexUOld(neighborIndexInSourcePatch);
      int linearSourceUNewIndex = source.getLinearIndexUNew(neighborIndexInSourcePatch);
      int linearDestinationIndex;
      if(interpolateToUOld){
        linearDestinationIndex = destination.getLinearIndexUOld(subcellIndexInDestinationPatch);
      } else {
        linearDestinationIndex = destination.getLinearIndexUNew(subcellIndexInDestinationPatch);
      }

      for(int unknown = 0; unknown < unknownsPerSubcell; unknown++) {
        //Get value from source patch either from uNew or from the ghostlayer in uOld.
        double sourceUOld = source.getValueUOld(linearSourceUOldIndex, unknown);
        double sourceUNew = source.getValueUNew(linearSourceUNewIndex, unknown);

        logDebug("interpolate(...)", "\tspatialFactorUNew=" << spatialFactor
            << ", spatialFactorUNew=" << spatialFactor
            << " due to destination position " << destination.getSubcellCenter(subcellIndexInDestinationPatch)
            << " and source position " << source.getSubcellCenter(neighborIndexInSourcePatch)
            << " and source.subcellsize=" << sourceSubcellSize << " and offset=" << offset);
        logDebug("interpolate(...)", "\ttimeFactor=" << timeFactor
            << " due to destination time " << destination.getTimeIntervals().getCurrentTime()
            << " and destination timestep size " << destination.getTimeIntervals().getTimestepSize()
            << " and source time " << source.getTimeIntervals().getCurrentTime()
            << " and source timestep size " << source.getTimeIntervals().getTimestepSize());

        if(interpolateToUOld) {
          logDebug("", "\tAdding UOld value " << sourceUOld << " and UNew value " << sourceUNew
              << " with spatialFactor " << spatialFactor << " and timeFactor "  << timeFactor
              << " to value " << destination.getValueUOld(subcellIndexInDestinationPatch, unknown));

          destination.setValueUOld(linearDestinationIndex, unknown,
              destination.getValueUOld(linearDestinationIndex, unknown)
              + spatialFactor * sourceUOld * (1.0-timeFactor) + spatialFactor * sourceUNew * timeFactor);
        } else {
          logDebug("interpolate(...)", "\tAdding UNew value " << sourceUNew << " and UOld value " << sourceUOld
              << " with spatialFactor " << spatialFactor << " and timeFactor " << timeFactor
              << " to value " << destination.getValueUNew(subcellIndexInDestinationPatch, unknown));

          destination.setValueUNew(linearDestinationIndex, unknown,
            destination.getValueUNew(linearDestinationIndex, unknown)
            + spatialFactor * sourceUOld * (1.0-timeFactor) + spatialFactor * sourceUNew * timeFactor);
        }
      }
    enddforx
    logDebug("interpolate(...)", "For subcell " << (subcellIndex + destinationOffset) << " interpolated value is " << destination.getValueUOld(subcellIndex + destinationOffset, 0));
  }

  #ifdef Asserts
  if(destination.containsNaN()) {
    std::cout << "Invalid interpolation "
        #ifdef Parallel
        << "on rank " << tarch::parallel::Node::getInstance().getRank() << " "
        #endif
        << "from patch " << std::endl << source.toString() << std::endl << source.toStringUNew() << std::endl << source.toStringUOldWithGhostLayer()
              << std::endl << "to patch" << std::endl << destination.toString() << std::endl << destination.toStringUNew() << std::endl << destination.toStringUOldWithGhostLayer() << std::endl;
    assertion(false);
  }

  #ifdef AssertForPositiveValues
  dfor(subcellIndex, destinationSize) {
    tarch::la::Vector<DIMENSIONS, int> subcellIndexInDestinationPatch = subcellIndex + destinationOffset;
    double checkedValue
      = interpolateToUOld ? destination.getValueUOld(subcellIndexInDestinationPatch, 0): destination.getValueUNew(subcellIndexInDestinationPatch, 0);
    if(checkedValue<= 0.0) {
      assertionFail("Invalid interpolation "
        #ifdef Parallel
        << "on rank " << tarch::parallel::Node::getInstance().getRank() << " "
        #endif
        << "from patch " << std::endl << source.toString() << std::endl << source.toStringUNew() << std::endl << source.toStringUOldWithGhostLayer()
        << std::endl << "to patch" << std::endl << destination.toString() << std::endl << destination.toStringUNew() << std::endl << destination.toStringUOldWithGhostLayer()
        << std::endl << "value=" << destination.getValueUOld(subcellIndexInDestinationPatch, 0)
      );
      throw "";
    }
  }
  #endif
  #endif

  logTraceOut("");
}
