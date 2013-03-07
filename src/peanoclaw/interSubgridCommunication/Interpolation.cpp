#include "peanoclaw/interSubgridCommunication/Interpolation.h"

#include "peanoclaw/Patch.h"
#include "peanoclaw/pyclaw/PyClaw.h"
#include "peano/utils/Loop.h"

void peanoclaw::interSubgridCommunication::Interpolation::interpolateDLinear(
  const tarch::la::Vector<DIMENSIONS, int>&    destinationSize,
  const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
  const peanoclaw::Patch& source,
  peanoclaw::Patch&       destination,
  bool interpolateToUOld,
  bool interpolateToCurrentTime
) {
  logTraceInWith4Arguments("", destinationSize, destinationOffset, source.toString(), destination.toString());
  assertionEquals(source.getUnknownsPerSubcell(), destination.getUnknownsPerSubcell());

  //TODO unterweg debug
//  std::cout << "Interpolated cells:" << tarch::la::volume(destinationSize) << std::endl;

  //Factor for interpolation in time
  double timeFactor;
  if(tarch::la::equals(source.getTimestepSize(), 0.0)) {
    timeFactor = 1.0;
  } else {
    if(interpolateToCurrentTime) {
      timeFactor = (destination.getTimeUOld() - source.getTimeUOld()) / (source.getTimeUNew() - source.getTimeUOld());
    } else {
      timeFactor = (destination.getTimeUNew() - source.getTimeUOld()) / (source.getTimeUNew() - source.getTimeUOld());
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
        tarch::la::smallerEquals(source.getPosition()(d) - source.getGhostLayerWidth() * destination.getSubcellSize()(d),
            destination.getPosition()(d) + destination.getSubcellSize()(d) * destinationOffset(d))
        && tarch::la::greaterEquals(source.getPosition()(d) + source.getSize()(d) + source.getGhostLayerWidth() * destination.getSubcellSize()(d),
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

        logDebug("", "\tspatialFactorUNew=" << spatialFactor
            << ", spatialFactorUNew=" << spatialFactor
            << " due to destination position " << destination.getSubcellCenter(subcellIndexInDestinationPatch)
            << " and source position " << source.getSubcellCenter(neighborIndexInSourcePatch)
            << " and source.subcellsize=" << sourceSubcellSize << " and offset=" << offset);
        logDebug("", "\ttimeFactor=" << timeFactor
            << " due to destination time " << destination.getCurrentTime()
            << " and destination timestep size " << destination.getTimestepSize()
            << " and source time " << source.getCurrentTime()
            << " and source timestep size " << source.getTimestepSize());

        if(interpolateToUOld) {
          logDebug("", "\tAdding UOld value " << sourceUOld << " and UNew value " << sourceUNew
              << " with spatialFactor " << spatialFactor << " and timeFactor "  << timeFactor
              << " to value " << destination.getValueUOld(subcellIndexInDestinationPatch, unknown));

          destination.setValueUOld(linearDestinationIndex, unknown,
              destination.getValueUOld(linearDestinationIndex, unknown)
              + spatialFactor * sourceUOld * (1.0-timeFactor) + spatialFactor * sourceUNew * timeFactor);
        } else {
          logDebug("", "\tAdding UNew value " << sourceUNew << " and UOld value " << sourceUOld
              << " with spatialFactor " << spatialFactor << " and timeFactor " << timeFactor
              << " to value " << destination.getValueUNew(subcellIndexInDestinationPatch, unknown));

          destination.setValueUNew(linearDestinationIndex, unknown,
            destination.getValueUNew(linearDestinationIndex, unknown)
            + spatialFactor * sourceUOld * (1.0-timeFactor) + spatialFactor * sourceUNew * timeFactor);
        }
      }
    enddforx
    logDebug("interpolateGhostLayerDataBlock", "For subcell " << (subcellIndex + destinationOffset) << " interpolated value is " << destination.getValueUOld(subcellIndex + destinationOffset, 0));
  }

  //TODO unterweg debug
  #ifdef Asserts
  if(destination.containsNaN()) {
    std::cout << "Invalid interpolation from patch " << std::endl << source.toString() << std::endl << source.toStringUNew() << std::endl << source.toStringUOldWithGhostLayer()
              << std::endl << "to patch" << std::endl << destination.toString() << std::endl << destination.toStringUNew() << std::endl << destination.toStringUOldWithGhostLayer() << std::endl;
    throw "";
  }

  dfor(subcellIndex, destinationSize) {
    tarch::la::Vector<DIMENSIONS, int> subcellIndexInDestinationPatch = subcellIndex + destinationOffset;
    double checkedValue
      = interpolateToUOld ? destination.getValueUOld(subcellIndexInDestinationPatch, 0): destination.getValueUNew(subcellIndexInDestinationPatch, 0);
    if(checkedValue<= 0.0) {
      std::cout << "Invalid interpolation from patch " << std::endl << source.toString() << std::endl << source.toStringUNew() << std::endl << source.toStringUOldWithGhostLayer()
          << std::endl << "to patch" << std::endl << destination.toString() << std::endl << destination.toStringUNew() << std::endl << destination.toStringUOldWithGhostLayer()
          << std::endl << "value=" << destination.getValueUOld(subcellIndexInDestinationPatch, 0) << std::endl;
      throw "";
    }
  }

  //Find max and min values to check correctness of the interpolation
  double* min = new double[source.getUnknownsPerSubcell()];
  double* max = new double[source.getUnknownsPerSubcell()];
  for (int d = 0; d < DIMENSIONS; d++) {
    for (int unknown = 0; unknown < source.getUnknownsPerSubcell(); unknown++) {
      min[unknown] = std::numeric_limits<double>::max();
      max[unknown] = -std::numeric_limits<double>::max();
    }
  }
  dfor(subcellIndex, source.getSubdivisionFactor() - 1) {
    for (int unknown = 0; unknown < source.getUnknownsPerSubcell(); unknown++) {
      double difference = 0.0;
      double minUValue = std::numeric_limits<double>::max();
      double maxUValue = -std::numeric_limits<double>::max();
      for(int d = 0; d < DIMENSIONS; d++) {
        tarch::la::Vector<DIMENSIONS, int> offset(0.0);
        offset(d) = 1;

        minUValue = std::min(minUValue, std::min(source.getValueUOld(subcellIndex, unknown), source.getValueUOld(subcellIndex + offset, unknown)));
        maxUValue = std::max(maxUValue, std::max(source.getValueUOld(subcellIndex, unknown), source.getValueUOld(subcellIndex + offset, unknown)));

        double differenceUOld = std::abs(source.getValueUOld(subcellIndex, unknown) - source.getValueUOld(subcellIndex + offset, unknown));
        double differenceUNew = std::abs(source.getValueUNew(subcellIndex, unknown) - source.getValueUNew(subcellIndex + offset, unknown));
        difference += differenceUOld * (1.0 - timeFactor) + differenceUNew * timeFactor;
      }

      double localMin = minUValue - difference / 2.0;
      double localMax = maxUValue + difference / 2.0;

      min[unknown] = std::min(localMin, min[unknown]);
      max[unknown] = std::max(localMax, max[unknown]);
    }
  }

  delete[] min;
  delete[] max;
  #endif

  logTraceOut("");
}

void peanoclaw::interSubgridCommunication::Interpolation::interpolateDLinearVersion2(
  const tarch::la::Vector<DIMENSIONS, int>&    destinationSize,
  const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
  const peanoclaw::Patch& source,
  peanoclaw::Patch&       destination,
  bool interpolateToUOld,
  bool interpolateToCurrentTime
) {
//  logTraceInWith4Arguments("", destinationSize, destinationOffset, source.toString(), destination.toString());
//  assertionEquals(source.getUnknownsPerSubcell(), destination.getUnknownsPerSubcell());
//
//  //TODO unterweg debug
////  std::cout << "Interpolated cells:" << tarch::la::volume(destinationSize) << std::endl;
//
//  //Factor for interpolation in time
//  double timeFactor;
//  if(tarch::la::equals(source.getTimestepSize(), 0.0)) {
//    timeFactor = 1.0;
//  } else {
//    if(interpolateToCurrentTime) {
//      timeFactor = (destination.getTimeUOld() - source.getTimeUOld()) / (source.getTimeUNew() - source.getTimeUOld());
//    } else {
//      timeFactor = (destination.getTimeUNew() - source.getTimeUOld()) / (source.getTimeUNew() - source.getTimeUOld());
//    }
//  }
//
//  //TODO This should be guaranteed by the timestepping criterion but may be violated at the moment
//  // when coarsening the grid. Fixed?
//  timeFactor = std::max(0.0, std::min(1.0, timeFactor));
//
//  assertion(!tarch::la::smaller(timeFactor, 0.0) && !tarch::la::greater(timeFactor, 1.0));
//
//  double epsilon = 1e-12;
//  Area destinationArea;
//  destinationArea._offset = destinationOffset;
//  destinationArea._size = destinationSize;
//
//  Area sourceArea = destinationArea.mapToPatch(source, destination, epsilon);
//
//  dfor(sourceSubcellIndexInArea, sourceArea._size) {
//    tarch::la::Vector<DIMENSIONS, int> sourceSubcellIndex = sourceSubcellIndexInArea + sourceArea._offset;
//    tarch::la::Vector<DIMENSIONS, double> sourceSubcellPosition = tarch::la::multiplyComponents(sourceSubcellIndex, source.getSubcellSize());
//    sourceSubcellPosition += source.getPosition();
//
//    assertion4(tarch::la::allGreaterEquals(sourceSubcellIndex, tarch::la::Vector<DIMENSIONS, int>(0))
//              && tarch::la::allGreater(source.getSubdivisionFactor(), sourceSubcellIndex), destinationArea, sourceArea, destination, source);
//
//    //Get area for single source cell
//    Area subcellArea = destinationArea.mapCellToPatch(source, destination, sourceSubcellIndex, sourceSubcellPosition, epsilon);
//
//    dfor(destinationSubcellIndexInArea, subcellArea._size) {
//      tarch::la::Vector<DIMENSIONS, int> destinationSubcellIndex = destinationSubcellIndexInArea + subcellArea._offset;
//
//      assertion6(tarch::la::allGreaterEquals(destinationSubcellIndex, tarch::la::Vector<DIMENSIONS, int>(0))
//                && tarch::la::allGreater(destination.getSubdivisionFactor(), destinationSubcellIndex),
//                destinationSubcellIndex, subcellArea, destinationArea, destinationArea, destination, source);
//
//    }
//  }
//
//  logTraceOut("");
}

peanoclaw::interSubgridCommunication::Interpolation::Interpolation(
  const peanoclaw::pyclaw::PyClaw& pyClaw
) : _pyClaw(pyClaw) {
}

void peanoclaw::interSubgridCommunication::Interpolation::interpolate (
    const tarch::la::Vector<DIMENSIONS, int>&    destinationSize,
    const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
    const peanoclaw::Patch& source,
    peanoclaw::Patch&        destination,
    bool interpolateToUOld,
    bool interpolateToCurrentTime
) {

  if(_pyClaw.providesInterpolation()) {
    _pyClaw.interpolate(
      source,
      destination
    );
  } else {
    //Use default interpolation
    interpolateDLinear(
      destinationSize,
      destinationOffset,
      source,
      destination,
      interpolateToUOld,
      interpolateToCurrentTime
    );
  }
}
