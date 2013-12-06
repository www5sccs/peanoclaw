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
    const Patch& finePatch,
    Patch& coarsePatch,
    int dimension,
    int direction
) const {
  logTraceInWith4Arguments("applyCoarseGridCorrection", finePatch.toString(), coarsePatch.toString(), dimension, direction);

  //Create description of the fine patch's face to be traversed
  tarch::la::Vector<DIMENSIONS, int> face = finePatch.getSubdivisionFactor();
  face(dimension) = 1;
  tarch::la::Vector<DIMENSIONS, int> offset(0);
  if(direction == 1) {
    offset(dimension) = finePatch.getSubdivisionFactor()(dimension) - 1;
  }

  //Create search area that needs to be considered around the neighboring cell in the coarse patch
  tarch::la::Vector<DIMENSIONS, int> searchArea = tarch::la::Vector<DIMENSIONS, int>(2);
  searchArea(dimension) = 1;

  logDebug("applyFluxCorrection", "face=" << face << ", offset=" << offset << ", searchArea=" << searchArea);

  tarch::la::Vector<DIMENSIONS, double> fineSubcellSize = finePatch.getSubcellSize();
  tarch::la::Vector<DIMENSIONS, double> coarseSubcellSize = coarsePatch.getSubcellSize();

  double refinementFactor = 1.0;
  for( int d = 0; d < DIMENSIONS; d++ ) {
    if( d != dimension ) {
      refinementFactor *= coarseSubcellSize(d) / fineSubcellSize(d);
    }
  }

  dfor(subcellIndexInFace, face) {
    tarch::la::Vector<DIMENSIONS, int> subcellIndexInFinePatch = subcellIndexInFace + offset;
    tarch::la::Vector<DIMENSIONS, double> subcellPositionInFinePatch = finePatch.getSubcellPosition(subcellIndexInFinePatch);
    tarch::la::Vector<DIMENSIONS, double> neighboringSubcellPositionInCoarsePatch = subcellPositionInFinePatch;
    neighboringSubcellPositionInCoarsePatch(dimension) += fineSubcellSize(dimension) * direction;
    tarch::la::Vector<DIMENSIONS, int> neighboringSubcellIndexInCoarsePatch =
        (tarch::la::multiplyComponents((neighboringSubcellPositionInCoarsePatch.convertScalar<double>() -coarsePatch.getPosition()), tarch::la::invertEntries(coarseSubcellSize))).convertScalar<int>();

    logDebug("applyFluxCorrection", "Correcting from cell " << subcellIndexInFinePatch);

    dfor(neighborOffset, searchArea) {
      tarch::la::Vector<DIMENSIONS, int> adjacentSubcellIndexInCoarsePatch = neighboringSubcellIndexInCoarsePatch + neighborOffset;

      logDebug("applyFluxCorrection", "Correcting cell " << adjacentSubcellIndexInCoarsePatch);

      if(
          !tarch::la::oneGreater(tarch::la::Vector<DIMENSIONS, int>(0), adjacentSubcellIndexInCoarsePatch)
      && !tarch::la::oneGreaterEquals(adjacentSubcellIndexInCoarsePatch, coarsePatch.getSubdivisionFactor())
      ) {
        //Get interface area
        double interfaceArea = calculateOverlappingArea(
            finePatch.getSubcellPosition(subcellIndexInFinePatch),
            fineSubcellSize,
            coarsePatch.getSubcellPosition(adjacentSubcellIndexInCoarsePatch),
            coarseSubcellSize,
            dimension
        );

        //TODO This depends on the application, since we assume that variable 0 holds the height of the shallow water equation
        //and variables 1 and 2 hold the velocity
        //Estimate the flux through the interface from fine to coarse grid, once from the point of view of the fine subcell and
        //once from the of the coarse subcell
        double fineGridFlux = finePatch.getValueUNew(subcellIndexInFinePatch, 0) * finePatch.getValueUNew(subcellIndexInFinePatch, 1 + dimension) * interfaceArea;
        double coarseGridFlux = coarsePatch.getValueUNew(adjacentSubcellIndexInCoarsePatch, 0) * coarsePatch.getValueUNew(adjacentSubcellIndexInCoarsePatch, 1 + dimension) * interfaceArea;

        //Estimate the according transfered volume during the fine patch's timestep
        double transferedVolumeFineGrid = fineGridFlux * finePatch.getTimeIntervals().getTimestepSize();
        double transferedVolumeCoarseGrid = coarseGridFlux * finePatch.getTimeIntervals().getTimestepSize();

        //        double delta = (transferedVolumeFineGrid * refinementFactor - transferedVolumeCoarseGrid) / coarseSubcellSize(dimension) / refinementFactor;
        double delta = (transferedVolumeFineGrid * refinementFactor - transferedVolumeCoarseGrid) / refinementFactor;

        logDebug("applyFluxCorrection", "Correcting neighbor cell " << adjacentSubcellIndexInCoarsePatch << " from cell " << subcellIndexInFinePatch << " with interfaceArea=" << interfaceArea << std::endl
            << "\tu0=" << finePatch.getValueUNew(subcellIndexInFinePatch, 0) << " u1=" << finePatch.getValueUNew(subcellIndexInFinePatch, 1) << " u2=" << finePatch.getValueUNew(subcellIndexInFinePatch, 2) << std::endl
            << "\tfineGridFlux=" << fineGridFlux << std::endl
            << "\tcoarseGridFlux=" << coarseGridFlux << std::endl
            << "\ttransferedVolumeFineGrid=" << transferedVolumeFineGrid << std::endl
            << "\ttransferedVolumeCoarseGrid=" << transferedVolumeCoarseGrid << std::endl
            << "\tdelta=" << delta << std::endl
            << "\told u=" << coarsePatch.getValueUNew(adjacentSubcellIndexInCoarsePatch, 0) << std::endl
            << "\tnew u=" << coarsePatch.getValueUNew(adjacentSubcellIndexInCoarsePatch, 0) + delta);

        //Scaled down due to inaccuracy
        coarsePatch.setValueUNew(adjacentSubcellIndexInCoarsePatch, 0,
            std::max(0.0, coarsePatch.getValueUNew(adjacentSubcellIndexInCoarsePatch, 0) + delta * 10e-8));
      }
    }
  }
  logTraceOut("applyCoarseGridCorrection");
}







