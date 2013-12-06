/*
 * DefaultRestriction.cpp
 *
 *  Created on: Mar 7, 2013
 *      Author: kristof
 */

#include "peanoclaw/interSubgridCommunication/DefaultRestriction.h"

#include "peanoclaw/Area.h"
#include "peanoclaw/Patch.h"

#include "peano/utils/Loop.h"

tarch::logging::Log peanoclaw::interSubgridCommunication::DefaultRestriction::_log( "peanoclaw::interSubgridCommunication::DefaultRestriction" ); 

void peanoclaw::interSubgridCommunication::DefaultRestriction::restrictArea (
  const peanoclaw::Patch& source,
  peanoclaw::Patch&       destination,
  const Area&             area
) {
  logTraceInWith2Arguments("restrictArea", source.toString(), destination.toString());
  assertionEquals(source.getUnknownsPerSubcell(), destination.getUnknownsPerSubcell());
  assertion(destination.isVirtual());

  double epsilon = 1e-12;

  //TODO unterweg restricting to interval [0, 1]
  double destinationTimeUOld = 0.0;// destination.getTimeUOld();
  double destinationTimeUNew = 1.0;// destination.getTimeUNew();

  const tarch::la::Vector<DIMENSIONS, double> sourceSubcellSize = tarch::la::multiplyComponents(source.getSize(), tarch::la::invertEntries(source.getSubdivisionFactor().convertScalar<double>()));
  const tarch::la::Vector<DIMENSIONS, double> destinationSubcellSize = tarch::la::multiplyComponents(destination.getSize(), tarch::la::invertEntries(destination.getSubdivisionFactor().convertScalar<double>()));
  const double destinationSubcellArea = tarch::la::volume(destinationSubcellSize);
  const tarch::la::Vector<DIMENSIONS, double> sourcePosition = source.getPosition();
  const tarch::la::Vector<DIMENSIONS, double> destinationPosition = destination.getPosition();
  int unknownsPerSubcell = source.getUnknownsPerSubcell();

  //Time factor
  double timeFactorUOld = 1.0;
  double timeFactorUNew = 1.0;
  if(tarch::la::greater(source.getTimeIntervals().getTimestepSize(), 0.0)) {
    timeFactorUOld = (destinationTimeUOld - source.getTimeIntervals().getTimeUOld()) / (source.getTimeIntervals().getTimeUNew() - source.getTimeIntervals().getTimeUOld());
    timeFactorUNew = (destinationTimeUNew - source.getTimeIntervals().getTimeUOld()) / (source.getTimeIntervals().getTimeUNew() - source.getTimeIntervals().getTimeUOld());
  }

  assertion4(timeFactorUOld == timeFactorUOld, destinationTimeUOld, destinationTimeUNew, source.getTimeIntervals().getCurrentTime(), source.getTimeIntervals().getTimestepSize());
  assertion4(timeFactorUNew == timeFactorUNew, destinationTimeUOld, destinationTimeUNew, source.getTimeIntervals().getCurrentTime(), source.getTimeIntervals().getTimestepSize());

  //Destination area
  Area destinationArea = area.mapToPatch(source, destination, epsilon);

  //Loop through area in destination
  dfor(destinationSubcellIndexInArea, destinationArea._size) {
    tarch::la::Vector<DIMENSIONS, int> destinationSubcellIndex = destinationSubcellIndexInArea + destinationArea._offset;
    tarch::la::Vector<DIMENSIONS, double> destinationSubcellPosition = tarch::la::multiplyComponents(destinationSubcellIndex.convertScalar<double>(), destinationSubcellSize);
    destinationSubcellPosition += destinationPosition;

    assertion4(tarch::la::allGreaterEquals(destinationSubcellIndex, tarch::la::Vector<DIMENSIONS, int>(0))
              && tarch::la::allGreater(destination.getSubdivisionFactor(), destinationSubcellIndex), area, destinationArea, destination, source);

    //Get area for single destination cell
    Area subcellArea = area.mapCellToPatch(sourcePosition, sourceSubcellSize, destinationSubcellSize, destinationSubcellIndex, destinationSubcellPosition, epsilon);
    assertion7(tarch::la::allGreaterEquals(subcellArea._size, 0), subcellArea, sourcePosition, sourceSubcellSize, destinationSubcellSize, destinationSubcellPosition, source, destination);

    //TODO unterweg debug
//    std::cout << "Restricted cells:" << tarch::la::volume(subcellArea._size) << std::endl;

    //Loop through area in source
    dfor(sourceSubcellIndexInArea, subcellArea._size) {
      tarch::la::Vector<DIMENSIONS, int> sourceSubcellIndex = sourceSubcellIndexInArea + subcellArea._offset;
      tarch::la::Vector<DIMENSIONS, double> sourceSubcellPosition = tarch::la::multiplyComponents(sourceSubcellIndex.convertScalar<double>(), sourceSubcellSize);
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

  logTraceOut("restrictArea");
}

void peanoclaw::interSubgridCommunication::DefaultRestriction::restrict(
  const peanoclaw::Patch& source,
  peanoclaw::Patch&       destination,
  bool restrictOnlyOverlappedAreas
) {
  if(restrictOnlyOverlappedAreas) {
    Area areas[DIMENSIONS_TIMES_TWO];
    int numberOfAreasToProcess = Area::getAreasOverlappedByNeighboringGhostlayers(
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

      assertion1(tarch::la::allGreaterEquals(areas[i]._size, 0), areas[i]);

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



