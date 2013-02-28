/*
 * PatchOperations.cpp
 *
 *  Created on: Jun 6, 2012
 *      Author: unterweg
 */

#include "PatchOperations.h"

#include "Patch.h"
#include "Area.h"
#include "peano/utils/Loop.h"

//double peanoclaw::calculateOverlappingArea(
//  const tarch::la::Vector<DIMENSIONS, double>& position1,
//  const tarch::la::Vector<DIMENSIONS, double>& size1,
//  const tarch::la::Vector<DIMENSIONS, double>& position2,
//  const tarch::la::Vector<DIMENSIONS, double>& size2
//) {
//  double area = 1.0;
//
//  for(int d = 0; d < DIMENSIONS; d++) {
//    double overlappingInterval =
//        std::min(position1(d)+size1(d), position2(d)+size2(d))
//          - std::max(position1(d), position2(d));
//    area *= overlappingInterval;
//
//    area = std::max(area, 0.0);
//  }
//
//  return area;
//}

void peanoclaw::interpolate(
  const tarch::la::Vector<DIMENSIONS, int>&    destinationSize,
  const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
  const peanoclaw::Patch& source,
  peanoclaw::Patch&       destination,
  bool interpolateToUOld,
  bool interpolateToCurrentTime
) {
  static tarch::logging::Log _log("peanoclaw::interpolate");
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
  tarch::la::Vector<DIMENSIONS, double> inverseSourceSubcellSize = tarch::la::Vector<DIMENSIONS, double>(1.0) / sourceSubcellSize;
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

void peanoclaw::interpolateVersion2(
  const tarch::la::Vector<DIMENSIONS, int>&    destinationSize,
  const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
  const peanoclaw::Patch& source,
  peanoclaw::Patch&       destination,
  bool interpolateToUOld,
  bool interpolateToCurrentTime
) {
//  static tarch::logging::Log _log("peanoclaw::interpolate");
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



void peanoclaw::restrict(
  const peanoclaw::Patch& source,
  peanoclaw::Patch&       destination,
  bool restrictOnlyOverlappedAreas
) {
  static tarch::logging::Log _log("peanoclaw");
  if(restrictOnlyOverlappedAreas) {
    Area areas[DIMENSIONS_TIMES_TWO];
    int numberOfAreasToProcess = getAreasForRestriction(
      destination.getLowerNeighboringGhostlayerBounds(),
      destination.getUpperNeighboringGhostlayerBounds(),
      source.getPosition(),
      source.getSize(),
      source.getSubcellSize(),
      source.getSubdivisionFactor(),
      areas
    );

    logDebug("restrict", "Restriction from patch " << source << std::endl << " to patch " << destination);
    for( int i = 0; i < numberOfAreasToProcess; i++ ) {

      logDebug("restrict", "Restricting area [" << areas[i]._offset << "], [" << areas[i]._size << "]");
      if(tarch::la::allGreater(areas[i]._size, tarch::la::Vector<DIMENSIONS, int>(0))) {
        restrictArea(source, destination, areas[i]);
      }
    }
  } else {
    //Restrict complete Patch
    Area area;
    area._offset = tarch::la::Vector<DIMENSIONS, int>(0);
    area._size = source.getSubdivisionFactor();
    restrictArea(source, destination, area);
  }
}

int peanoclaw::getAreasForRestriction (
  const tarch::la::Vector<DIMENSIONS, double>& lowerNeighboringGhostlayerBounds,
  const tarch::la::Vector<DIMENSIONS, double>& upperNeighboringGhostlayerBounds,
  const tarch::la::Vector<DIMENSIONS, double>& sourcePosition,
  const tarch::la::Vector<DIMENSIONS, double>& sourceSize,
  const tarch::la::Vector<DIMENSIONS, double>& sourceSubcellSize,
  const tarch::la::Vector<DIMENSIONS, int>& sourceSubdivisionFactor,
  Area areas[DIMENSIONS_TIMES_TWO]
) {

  //Check if bounds overlap
  if(tarch::la::oneGreater(upperNeighboringGhostlayerBounds + sourceSubcellSize, lowerNeighboringGhostlayerBounds)
    || tarch::la::oneGreaterEquals(upperNeighboringGhostlayerBounds, sourcePosition + sourceSize)
    || !tarch::la::allGreater(lowerNeighboringGhostlayerBounds, sourcePosition)) {
    //If whole patch is overlapped -> One area holds whole patch, others are empty
    areas[0]._offset = tarch::la::Vector<DIMENSIONS, int>(0);
    areas[0]._size = sourceSubdivisionFactor;
    areas[1]._offset = tarch::la::Vector<DIMENSIONS, int>(0);
    areas[1]._size = tarch::la::Vector<DIMENSIONS, int>(0);
    for(int d = 1; d < DIMENSIONS; d++) {
      areas[2*d]._offset = tarch::la::Vector<DIMENSIONS, int>(0);
      areas[2*d]._size = tarch::la::Vector<DIMENSIONS, int>(0);
      areas[2*d + 1]._offset = tarch::la::Vector<DIMENSIONS, int>(0);
      areas[2*d + 1]._size = tarch::la::Vector<DIMENSIONS, int>(0);
    }
    return 1;
  } else {
    double epsilon = 1e-12;
    tarch::la::Vector<DIMENSIONS, double> doubleLowerBoundsInSourcePatch = (lowerNeighboringGhostlayerBounds - sourcePosition) / sourceSubcellSize + epsilon;
    tarch::la::Vector<DIMENSIONS, double> doubleUpperBoundsInSourcePatch = (upperNeighboringGhostlayerBounds - sourcePosition) / sourceSubcellSize - epsilon;
    tarch::la::Vector<DIMENSIONS, int> lowerBoundsInSourcePatch;
    tarch::la::Vector<DIMENSIONS, int> upperBoundsInSourcePatch;

    for(int d = 0; d < DIMENSIONS; d++) {
      //TODO unterweg: The constants +/-1e10 are choosen sufficiently large -> need a good heuristic for this...
      doubleLowerBoundsInSourcePatch(d) = std::min((double)std::numeric_limits<int>::max(), doubleLowerBoundsInSourcePatch(d));
      doubleUpperBoundsInSourcePatch(d) = std::max((double)std::numeric_limits<int>::min(), doubleUpperBoundsInSourcePatch(d));
      lowerBoundsInSourcePatch(d) = std::max(std::min((int)std::floor(doubleLowerBoundsInSourcePatch(d)), sourceSubdivisionFactor(d)), -1);
      upperBoundsInSourcePatch(d) = std::max(std::min((int)std::floor(doubleUpperBoundsInSourcePatch(d)), sourceSubdivisionFactor(d)), -1);
    }

    for(int d = 0; d < DIMENSIONS; d++) {
      areas[2*d]._size(d) = upperBoundsInSourcePatch(d) + 1;
      areas[2*d]._offset(d) = 0;
      areas[2*d+1]._size(d) = sourceSubdivisionFactor(d) - lowerBoundsInSourcePatch(d);
      areas[2*d+1]._offset(d) = lowerBoundsInSourcePatch(d);

      //Restricted on lower dimensions
      for(int i = 0; i < d; i++) {
        areas[2*d]._size(i) = -upperBoundsInSourcePatch(i) - 1 + lowerBoundsInSourcePatch(i);
        areas[2*d]._offset(i) = upperBoundsInSourcePatch(i) + 1;
        areas[2*d+1]._size(i) = areas[2*d]._size(i);
        areas[2*d+1]._offset(i) = areas[2*d]._offset(i);
      }
      //Spread over whole patch in higher dimensions
      for(int i = d + 1; i < DIMENSIONS; i++) {
        areas[2*d]._size(i) = sourceSubdivisionFactor(i);
        areas[2*d]._offset(i) = 0;
        areas[2*d+1]._size(i) = sourceSubdivisionFactor(i);
        areas[2*d+1]._offset(i) = 0;
      }
    }

    return DIMENSIONS_TIMES_TWO;
  }
}

void peanoclaw::restrictArea (
  const peanoclaw::Patch& source,
  peanoclaw::Patch&       destination,
  const Area&                                  area
) {
  static tarch::logging::Log _log("peanoclaw");
  logTraceInWith2Arguments("restrict", source.toString(), destination.toString());
  assertionEquals(source.getUnknownsPerSubcell(), destination.getUnknownsPerSubcell());
  assertion(destination.isVirtual());

  double epsilon = 1e-12;

  //TODO unterweg restricting to interval [0, 1]
  double destinationTimeUOld = 0.0;// destination.getTimeUOld();
  double destinationTimeUNew = 1.0;// destination.getTimeUNew();

  const tarch::la::Vector<DIMENSIONS, double> sourceSubcellSize = source.getSize() / source.getSubdivisionFactor();
  const tarch::la::Vector<DIMENSIONS, double> destinationSubcellSize = destination.getSize() / destination.getSubdivisionFactor();
  const double destinationSubcellArea = tarch::la::volume(destinationSubcellSize);
  const tarch::la::Vector<DIMENSIONS, double> sourcePosition = source.getPosition();
  const tarch::la::Vector<DIMENSIONS, double> destinationPosition = destination.getPosition();
  int unknownsPerSubcell = source.getUnknownsPerSubcell();

  //Time factor
  double timeFactorUOld = 1.0;
  double timeFactorUNew = 1.0;
  if(tarch::la::greater(source.getTimestepSize(), 0.0)) {
    timeFactorUOld = (destinationTimeUOld - source.getTimeUOld()) / (source.getTimeUNew() - source.getTimeUOld());
    timeFactorUNew = (destinationTimeUNew - source.getTimeUOld()) / (source.getTimeUNew() - source.getTimeUOld());
  }

  assertion4(timeFactorUOld == timeFactorUOld, destinationTimeUOld, destinationTimeUNew, source.getCurrentTime(), source.getTimestepSize());
  assertion4(timeFactorUNew == timeFactorUNew, destinationTimeUOld, destinationTimeUNew, source.getCurrentTime(), source.getTimestepSize());

  //Destination area
  Area destinationArea = area.mapToPatch(source, destination, epsilon);

  //Loop through area in destination
  dfor(destinationSubcellIndexInArea, destinationArea._size) {
    tarch::la::Vector<DIMENSIONS, int> destinationSubcellIndex = destinationSubcellIndexInArea + destinationArea._offset;
    tarch::la::Vector<DIMENSIONS, double> destinationSubcellPosition = tarch::la::multiplyComponents(destinationSubcellIndex, destinationSubcellSize);
    destinationSubcellPosition += destinationPosition;

    assertion4(tarch::la::allGreaterEquals(destinationSubcellIndex, tarch::la::Vector<DIMENSIONS, int>(0))
              && tarch::la::allGreater(destination.getSubdivisionFactor(), destinationSubcellIndex), area, destinationArea, destination, source);

    //Get area for single destination cell
    Area subcellArea = area.mapCellToPatch(sourcePosition, sourceSubcellSize, destinationSubcellSize, destinationSubcellIndex, destinationSubcellPosition, epsilon);

    //TODO unterweg debug
//    std::cout << "Restricted cells:" << tarch::la::volume(subcellArea._size) << std::endl;

    //Loop through area in source
    dfor(sourceSubcellIndexInArea, subcellArea._size) {
      tarch::la::Vector<DIMENSIONS, int> sourceSubcellIndex = sourceSubcellIndexInArea + subcellArea._offset;
      tarch::la::Vector<DIMENSIONS, double> sourceSubcellPosition = tarch::la::multiplyComponents(sourceSubcellIndex, sourceSubcellSize);
      sourceSubcellPosition += sourcePosition;

      assertion6(tarch::la::allGreaterEquals(sourceSubcellIndex, tarch::la::Vector<DIMENSIONS, int>(0))
                && tarch::la::allGreater(source.getSubdivisionFactor(), sourceSubcellIndex),
                destinationSubcellIndex, subcellArea, area, destinationArea, destination, source);

      double overlapArea = calculateOverlappingArea(
        destinationSubcellPosition,
        destinationSubcellSize,
        sourceSubcellPosition,
        sourceSubcellSize);

      const int linearIndexSourceUOld = source.getLinearIndexUOld(sourceSubcellIndex);
      const int linearIndexSourceUNew = source.getLinearIndexUNew(sourceSubcellIndex);
      const int linearIndexDestinationUOld = destination.getLinearIndexUOld(destinationSubcellIndex);
      const int linearIndexDestinationUNew = destination.getLinearIndexUNew(destinationSubcellIndex);

      for(int unknown = 0; unknown < unknownsPerSubcell; unknown++) {
        logDebug("restrictToVirtualPatch(...)", "Copying values " <<
            (source.getValueUOld(sourceSubcellIndex, unknown) * (1.0 - timeFactorUOld)
            + source.getValueUNew(sourceSubcellIndex, unknown) * timeFactorUOld)
            << " and " << (source.getValueUOld(sourceSubcellIndex, unknown) * (1.0 - timeFactorUNew)
                + source.getValueUNew(sourceSubcellIndex, unknown) * timeFactorUNew)
            << " from cell " << (sourceSubcellIndex)
            << " to cell " << destinationSubcellIndex
            << " with destinationSubcellArea=" << destinationSubcellArea
            << ", overlapArea=" << overlapArea
            << ", fraction=" << (overlapArea / destinationSubcellArea)
            << ", timeFactorUOld=" << timeFactorUOld
            << ", timeFactorUNew=" << timeFactorUNew
        );

        double sourceValueUOld = source.getValueUOld(linearIndexSourceUOld, unknown);
        double sourceValueUNew = source.getValueUNew(linearIndexSourceUNew, unknown);

        double deltaUOld = (sourceValueUOld * (1.0 - timeFactorUOld) + sourceValueUNew * timeFactorUOld)
                         * overlapArea / destinationSubcellArea;
        double deltaUNew = (sourceValueUOld * (1.0 - timeFactorUNew) + sourceValueUNew * timeFactorUNew)
                         * overlapArea / destinationSubcellArea;

        destination.setValueUOld(
          linearIndexDestinationUOld,
          unknown,
          destination.getValueUOld(linearIndexDestinationUOld, unknown) + deltaUOld
        );
        destination.setValueUNew(
          linearIndexDestinationUNew,
          unknown,
          destination.getValueUNew(linearIndexDestinationUNew, unknown) + deltaUNew
        );
      }
    }
  }
}
