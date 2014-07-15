/*
 * DefaultFluxCorrection.cpp
 *
 *  Created on: Mar 7, 2013
 *      Author: kristof
 */

#include "peanoclaw/interSubgridCommunication/DefaultFluxCorrection.h"

#include "peanoclaw/Patch.h"

#include "peano/utils/Loop.h"

tarch::logging::Log peanoclaw::interSubgridCommunication::DefaultFluxCorrection::_log( "peanoclaw::interSubgridCommunication::DefaultFluxCorrection" ); 

double peanoclaw::interSubgridCommunication::DefaultFluxCorrection::calculateOverlappingArea(
    tarch::la::Vector<DIMENSIONS, double> position1,
    tarch::la::Vector<DIMENSIONS, double> size1,
    tarch::la::Vector<DIMENSIONS, double> position2,
    tarch::la::Vector<DIMENSIONS, double> size2,
    int projectionAxis
) const {
  double area = 1.0;

  for(int d = 0; d < DIMENSIONS; d++) {
    if(d != projectionAxis) {
      double overlappingInterval =
          std::min(position1(d)+size1(d), position2(d)+size2(d))
      - std::max(position1(d), position2(d));
      area *= overlappingInterval;

      if(area < 0.0) {
        area = 0.0;
      }
    }
  }

  return area;
}

peanoclaw::interSubgridCommunication::DefaultFluxCorrection::~DefaultFluxCorrection() {
}

void peanoclaw::interSubgridCommunication::DefaultFluxCorrection::applyCorrection(
    const Patch& sourceSubgrid,
    Patch& destinationSubgrid,
    int dimension,
    int direction
) const {
  logTraceInWith4Arguments("applyCoarseGridCorrection", finePatch.toString(), coarsePatch.toString(), dimension, direction);

  peanoclaw::grid::SubgridAccessor sourceAccessor = sourceSubgrid.getAccessor();
  peanoclaw::grid::SubgridAccessor destinationAccessor = destinationSubgrid.getAccessor();

  //Create description of the fine patch's face to be traversed
  tarch::la::Vector<DIMENSIONS, int> face = sourceSubgrid.getSubdivisionFactor();
  face(dimension) = 1;
  tarch::la::Vector<DIMENSIONS, int> offset(0);
  if(direction == 1) {
    offset(dimension) = sourceSubgrid.getSubdivisionFactor()(dimension) - 1;
  }

  //Create search area that needs to be considered around the neighboring cell in the coarse patch
  tarch::la::Vector<DIMENSIONS, int> searchArea = tarch::la::Vector<DIMENSIONS, int>(2);
  searchArea(dimension) = 1;

  logDebug("applyFluxCorrection", "face=" << face << ", offset=" << offset << ", searchArea=" << searchArea);

  tarch::la::Vector<DIMENSIONS, double> sourceSubcellSize = sourceSubgrid.getSubcellSize();
  tarch::la::Vector<DIMENSIONS, double> destinationSubcellSize = destinationSubgrid.getSubcellSize();

  const peanoclaw::grid::TimeIntervals& sourceTimeIntervals = sourceSubgrid.getTimeIntervals();
  const peanoclaw::grid::TimeIntervals& destinationTimeIntervals = destinationSubgrid.getTimeIntervals();
  double timestepOverlap
    = std::max(0.0,   std::min(sourceTimeIntervals.getCurrentTime() + sourceTimeIntervals.getTimestepSize(), destinationTimeIntervals.getCurrentTime() + destinationTimeIntervals.getTimestepSize())
                  - std::max(sourceTimeIntervals.getCurrentTime(), destinationTimeIntervals.getCurrentTime()));

  double refinementFactor = 1.0;
  for( int d = 0; d < DIMENSIONS; d++ ) {
    if( d != dimension ) {
      refinementFactor *= destinationSubcellSize(d) / sourceSubcellSize(d);
    }
  }

  dfor(subcellIndexInFace, face) {
    tarch::la::Vector<DIMENSIONS, int> subcellIndexInSourcePatch = subcellIndexInFace + offset;
    tarch::la::Vector<DIMENSIONS, double> subcellPositionInFinePatch = sourceSubgrid.getSubcellPosition(subcellIndexInSourcePatch);
    tarch::la::Vector<DIMENSIONS, double> neighboringSubcellPositionInCoarsePatch = subcellPositionInFinePatch;
    neighboringSubcellPositionInCoarsePatch(dimension) += sourceSubcellSize(dimension) * direction;
    tarch::la::Vector<DIMENSIONS, int> neighboringSubcellIndexInCoarsePatch =
        (tarch::la::multiplyComponents((neighboringSubcellPositionInCoarsePatch.convertScalar<double>() -destinationSubgrid.getPosition()), tarch::la::invertEntries(destinationSubcellSize))).convertScalar<int>();

    logDebug("applyFluxCorrection", "Correcting from cell " << subcellIndexInSourcePatch);

    dfor(neighborOffset, searchArea) {
      tarch::la::Vector<DIMENSIONS, int> adjacentSubcellIndexInDestinationPatch = neighboringSubcellIndexInCoarsePatch + neighborOffset;

      logDebug("applyFluxCorrection", "Correcting cell " << adjacentSubcellIndexInDestinationPatch);

      if(
          !tarch::la::oneGreater(tarch::la::Vector<DIMENSIONS, int>(0), adjacentSubcellIndexInDestinationPatch)
      && !tarch::la::oneGreaterEquals(adjacentSubcellIndexInDestinationPatch, destinationSubgrid.getSubdivisionFactor())
      ) {
        //Get interface area
        double interfaceArea = calculateOverlappingArea(
            sourceSubgrid.getSubcellPosition(subcellIndexInSourcePatch),
            sourceSubcellSize,
            destinationSubgrid.getSubcellPosition(adjacentSubcellIndexInDestinationPatch),
            destinationSubcellSize,
            dimension
        );

        for(int unknown = 0; unknown < sourceSubgrid.getUnknownsPerSubcell(); unknown++) {
          //TODO This depends on the application, since we assume that variable 0 holds the height of the shallow water equation
          //or pressure for the euler equation and variables 1, 2, ... hold the impulse in x, y, ... direction
          //Estimate the flux through the interface from fine to coarse grid, once from the point of view of the fine subcell and
          //once from the of the coarse subcell
          double fineGridFlux = sourceAccessor.getValueUNew(subcellIndexInSourcePatch, unknown) * sourceAccessor.getValueUNew(subcellIndexInSourcePatch, 1 + dimension) * interfaceArea;
          double coarseGridFlux = destinationAccessor.getValueUNew(adjacentSubcellIndexInDestinationPatch, unknown) * destinationAccessor.getValueUNew(adjacentSubcellIndexInDestinationPatch, 1 + dimension) * interfaceArea;

          //Estimate the according transfered volume during the fine patch's timestep
          double transferedVolumeFineGrid = fineGridFlux * timestepOverlap;
          double transferedVolumeCoarseGrid = coarseGridFlux * timestepOverlap;

          double delta = transferedVolumeFineGrid - transferedVolumeCoarseGrid;

          logDebug("applyFluxCorrection", "Correcting neighbor cell " << adjacentSubcellIndexInDestinationPatch << " from cell " << subcellIndexInSourcePatch << " with interfaceArea=" << interfaceArea << std::endl
              << "\tu0=" << sourceSubgrid.getAccessor().getValueUNew(subcellIndexInSourcePatch, 0) << " u1=" << sourceSubgrid.getAccessor().getValueUNew(subcellIndexInSourcePatch, 1) << " u2=" << sourceSubgrid.getAccessor().getValueUNew(subcellIndexInSourcePatch, 2) << std::endl
              << "\tfineGridFlux=" << fineGridFlux << std::endl
              << "\tcoarseGridFlux=" << coarseGridFlux << std::endl
              << "\ttransferedVolumeFineGrid=" << transferedVolumeFineGrid << std::endl
              << "\ttransferedVolumeCoarseGrid=" << transferedVolumeCoarseGrid << std::endl
              << "\tdelta=" << delta << std::endl
              << "\told u=" << coarsePatch.getAccessor().getValueUNew(adjacentSubcellIndexInDestinationPatch, 0) << std::endl
              << "\tnew u=" << coarsePatch.getAccessor().getValueUNew(adjacentSubcellIndexInDestinationPatch, 0) + delta);

          destinationAccessor.setValueUNew(adjacentSubcellIndexInDestinationPatch, unknown,
              destinationAccessor.getValueUNew(adjacentSubcellIndexInDestinationPatch, unknown) + delta);
        }
      }
    }
  }
  logTraceOut("applyCoarseGridCorrection");
}







