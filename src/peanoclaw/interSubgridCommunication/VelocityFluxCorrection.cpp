/*
 * VelocityFluxCorrection.cpp
 *
 *  Created on: Mar 7, 2013
 *      Author: kristof
 */

#include "peanoclaw/interSubgridCommunication/VelocityFluxCorrection.h"

#include "peanoclaw/Patch.h"

#include "peano/utils/Loop.h"

tarch::logging::Log peanoclaw::interSubgridCommunication::VelocityFluxCorrection::_log( "peanoclaw::interSubgridCommunication::VelocityFluxCorrection" );

double peanoclaw::interSubgridCommunication::VelocityFluxCorrection::calculateOverlappingRegion(
    tarch::la::Vector<DIMENSIONS, double> position1,
    tarch::la::Vector<DIMENSIONS, double> size1,
    tarch::la::Vector<DIMENSIONS, double> position2,
    tarch::la::Vector<DIMENSIONS, double> size2,
    int projectionAxis
) const {
  double region = 1.0;

  for(int d = 0; d < DIMENSIONS; d++) {
    if(d != projectionAxis) {
      double overlappingInterval =
          std::min(position1(d)+size1(d), position2(d)+size2(d))
      - std::max(position1(d), position2(d));
      region *= overlappingInterval;

      if(region < 0.0) {
        region = 0.0;
      }
    }
  }

  return region;
}

void peanoclaw::interSubgridCommunication::VelocityFluxCorrection::correctFluxBetweenCells(
  int dimension,
  int direction,
  double timestepOverlap,
  const peanoclaw::Patch& sourceSubgrid,
  peanoclaw::Patch& destinationSubgrid,
  peanoclaw::grid::SubgridAccessor& sourceAccessor,
  peanoclaw::grid::SubgridAccessor& destinationAccessor,
  const peanoclaw::grid::TimeIntervals& sourceTimeIntervals,
  const peanoclaw::grid::TimeIntervals& destinationTimeIntervals,
  double destinationSubcellVolume,
  const tarch::la::Vector<DIMENSIONS,double>& sourceSubcellSize,
  const tarch::la::Vector<DIMENSIONS,double>& destinationSubcellSize,
  const tarch::la::Vector<DIMENSIONS,int>& subcellIndexInSourcePatch,
  const tarch::la::Vector<DIMENSIONS,int>& ghostlayerSubcellIndexInSourcePatch,
  const tarch::la::Vector<DIMENSIONS,int>& adjacentSubcellIndexInDestinationPatch,
  const tarch::la::Vector<DIMENSIONS,int>& ghostlayerSubcellIndexInDestinationPatch
) const {
  logDebug("correctFluxBetweenCells", "Correcting cell " << adjacentSubcellIndexInDestinationPatch);
  if(
      !tarch::la::oneGreater(tarch::la::Vector<DIMENSIONS, int>(0), adjacentSubcellIndexInDestinationPatch)
  && !tarch::la::oneGreaterEquals(adjacentSubcellIndexInDestinationPatch, destinationSubgrid.getSubdivisionFactor())
  ) {
    //Get interface region
    double interfaceRegion = calculateOverlappingRegion(
        sourceSubgrid.getSubcellPosition(subcellIndexInSourcePatch),
        sourceSubcellSize,
        destinationSubgrid.getSubcellPosition(adjacentSubcellIndexInDestinationPatch),
        destinationSubcellSize,
        dimension
    );

//    for(int unknown = 0; unknown < sourceSubgrid.getUnknownsPerSubcell(); unknown++) {
    for(int unknown = sourceSubgrid.getUnknownsPerSubcell()-1; unknown >= 0; unknown--) {
//    for(int unknown = 0; unknown < 1; unknown++) {

      double sourceDensityUOld = sourceAccessor.getValueUOld(subcellIndexInSourcePatch, 0);
      double sourceVelocityUOld = direction * sourceAccessor.getValueUOld(subcellIndexInSourcePatch, dimension+1);
      double sourceDensityUNew = sourceAccessor.getValueUNew(subcellIndexInSourcePatch, 0);
      double sourceVelocityUNew = direction * sourceAccessor.getValueUNew(subcellIndexInSourcePatch, dimension+1);
      double sourceDensityGhostCell = sourceAccessor.getValueUOld(ghostlayerSubcellIndexInSourcePatch, 0);
      double sourceVelocityGhostCell = direction * sourceAccessor.getValueUOld(ghostlayerSubcellIndexInSourcePatch, dimension+1);

      double destinationDensityUOld = destinationAccessor.getValueUOld(adjacentSubcellIndexInDestinationPatch, 0);
      double destinationVelocityUOld = direction * destinationAccessor.getValueUOld(adjacentSubcellIndexInDestinationPatch, dimension+1);
      double destinationDensityUNew = destinationAccessor.getValueUNew(adjacentSubcellIndexInDestinationPatch, 0);
      double destinationVelocityUNew = direction * destinationAccessor.getValueUNew(adjacentSubcellIndexInDestinationPatch, dimension+1);
      double destinationDensityGhostCell = destinationAccessor.getValueUOld(ghostlayerSubcellIndexInDestinationPatch, 0);
      double destinationVelocityGhostCell = direction * destinationAccessor.getValueUOld(ghostlayerSubcellIndexInDestinationPatch, dimension+1);


      //TODO This depends on the application, since we assume that variable 0 holds the water height of the shallow water
      //equation or pressure for the euler equation and variables 1, 2, ... hold the impulse in x, y, ... direction
      //Estimate the flux through the interface from fine to coarse grid, once from the point of view of the fine subcell and
      //once from the of the coarse subcell
      double sourceValue = sourceAccessor.getValueUOld(subcellIndexInSourcePatch, unknown);
      if(unknown > 0) sourceValue *= sourceAccessor.getValueUOld(subcellIndexInSourcePatch, 0);
      double sourceValueGhostCell = sourceAccessor.getValueUOld(ghostlayerSubcellIndexInSourcePatch, unknown);
      double sourceFlux
            = 0.5 * (sourceValue * sourceVelocityUOld + sourceValueGhostCell * sourceVelocityGhostCell) * interfaceRegion;

      double destinationValueUOld = destinationAccessor.getValueUOld(adjacentSubcellIndexInDestinationPatch, unknown);
      if(unknown > 0) destinationValueUOld *= destinationAccessor.getValueUOld(adjacentSubcellIndexInDestinationPatch, 0);
      double destinationValueUNew = destinationAccessor.getValueUNew(adjacentSubcellIndexInDestinationPatch, unknown);
      if(unknown > 0) destinationValueUNew *= destinationAccessor.getValueUOld(adjacentSubcellIndexInDestinationPatch, 0);
      double destinationValueGhostCell = destinationAccessor.getValueUOld(ghostlayerSubcellIndexInDestinationPatch, unknown);
      double timeFactor = 0.0; //(sourceTimeIntervals.getCurrentTime() - destinationTimeIntervals.getCurrentTime()) / destinationTimeIntervals.getTimestepSize();
      double destinationValue = destinationValueUOld * (1.0-timeFactor) + destinationValueUNew * timeFactor;
      double destinationVelocity = destinationVelocityUOld * (1.0-timeFactor) + destinationVelocityUNew * timeFactor;
      double destinationFlux
        = 0.5 * (destinationValue * destinationVelocityUOld + destinationValueGhostCell * destinationVelocityGhostCell) * interfaceRegion;

      //Estimate the according transfered volume during the fine patch's timestep
      double transferedSourceVolume = sourceFlux * timestepOverlap;
      double transferedDestinationVolume = destinationFlux * timestepOverlap;
      double delta = transferedSourceVolume - transferedDestinationVolume;

      logDebug("applyFluxCorrection", "Correcting neighbor cell " << adjacentSubcellIndexInDestinationPatch << " from cell " << subcellIndexInSourcePatch << " with interfaceRegion=" << interfaceRegion << std::endl
          << "\tu0=" << sourceSubgrid.getAccessor().getValueUNew(subcellIndexInSourcePatch, 0) << " u1=" << sourceSubgrid.getAccessor().getValueUNew(subcellIndexInSourcePatch, 1) << " u2=" << sourceSubgrid.getAccessor().getValueUNew(subcellIndexInSourcePatch, 2) << std::endl
          << "\ttimestepOverlap=" << timestepOverlap << std::endl
          << "\tsourceFlux=" << sourceFlux << std::endl
          << "\tdestinationFlux=" << destinationFlux << std::endl
          << "\ttransferedSourceVolume=" << transferedSourceVolume << std::endl
          << "\ttransferedDestinationVolume=" << transferedDestinationVolume << std::endl
          << "\tdelta=" << delta << std::endl
          << "\told u=" << destinationSubgrid.getAccessor().getValueUNew(adjacentSubcellIndexInDestinationPatch, 0) << std::endl
          << "\tnew u=" << destinationSubgrid.getAccessor().getValueUNew(adjacentSubcellIndexInDestinationPatch, 0) + delta);

      //TODO unterweg debug
//      std::cout << "sourceFlux=" << fineGridFlux << " destinationFlux=" << coarseGridFlux
//          << " value=" << destinationAccessor.getValueUNew(adjacentSubcellIndexInDestinationPatch, unknown) << " delta=" << delta
//          << " subcellVolume=" << sourceSubcellVolume << " interface=" << interfaceRegion
//          << " h_s=" << sourceAccessor.getValueUOld(subcellIndexInSourcePatch, unknown) << " u_s=" << sourceAccessor.getValueUOld(subcellIndexInSourcePatch, 1 + dimension)
//          << " h_d=" << destinationAccessor.getValueUOld(adjacentSubcellIndexInDestinationPatch, unknown) << " u_d=" << destinationAccessor.getValueUOld(adjacentSubcellIndexInDestinationPatch, 1 + dimension) << std::endl;

      if(unknown == 0) {
        destinationAccessor.setValueUNew(adjacentSubcellIndexInDestinationPatch, unknown,
                  std::max(0.0, destinationAccessor.getValueUNew(adjacentSubcellIndexInDestinationPatch, unknown) - delta / destinationSubcellVolume));
      } else {
        double height = destinationAccessor.getValueUNew(adjacentSubcellIndexInDestinationPatch, 0);
        if(tarch::la::greater(height, 0)) {
          destinationAccessor.setValueUNew(adjacentSubcellIndexInDestinationPatch, unknown,
                    destinationAccessor.getValueUNew(adjacentSubcellIndexInDestinationPatch, unknown) - delta / destinationSubcellVolume / height);
        } else {
          destinationAccessor.setValueUNew(adjacentSubcellIndexInDestinationPatch, unknown, 0.0);
        }
      }

      //TODO unterweg debug
      assertion7(unknown!=0 || tarch::la::greaterEquals(destinationAccessor.getValueUNew(adjacentSubcellIndexInDestinationPatch, unknown), 0),
          sourceSubgrid,
          destinationSubgrid,
          adjacentSubcellIndexInDestinationPatch,
          destinationAccessor.getValueUNew(adjacentSubcellIndexInDestinationPatch, unknown),
          timestepOverlap,
          sourceSubgrid.toStringUOldWithGhostLayer(),
          destinationSubgrid.toStringUOldWithGhostLayer()
      );
    }
  }
}

peanoclaw::interSubgridCommunication::VelocityFluxCorrection::~VelocityFluxCorrection() {
}

void peanoclaw::interSubgridCommunication::VelocityFluxCorrection::applyCorrection(
    Patch& sourceSubgrid,
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

  //Create search region that needs to be considered around the neighboring cell in the coarse patch
  tarch::la::Vector<DIMENSIONS, int> searchRegion = tarch::la::Vector<DIMENSIONS, int>(2);
  searchRegion(dimension) = 1;

  logDebug("applyFluxCorrection", "face=" << face << ", offset=" << offset << ", searchRegion=" << searchRegion);

  tarch::la::Vector<DIMENSIONS, double> sourceSubcellSize = sourceSubgrid.getSubcellSize();
  tarch::la::Vector<DIMENSIONS, double> destinationSubcellSize = destinationSubgrid.getSubcellSize();

  const peanoclaw::grid::TimeIntervals& sourceTimeIntervals = sourceSubgrid.getTimeIntervals();
  const peanoclaw::grid::TimeIntervals& destinationTimeIntervals = destinationSubgrid.getTimeIntervals();
  double timestepOverlap
    = std::max(0.0,   std::min(sourceTimeIntervals.getCurrentTime() + sourceTimeIntervals.getTimestepSize(), destinationTimeIntervals.getCurrentTime() + destinationTimeIntervals.getTimestepSize())
                  - std::max(sourceTimeIntervals.getCurrentTime(), destinationTimeIntervals.getCurrentTime()));

  if(tarch::la::equals(timestepOverlap, 0.0)) {
    return;
  }

//  double refinementFactor = 1.0;
//  for( int d = 0; d < DIMENSIONS; d++ ) {
//    if( d != dimension ) {
//      refinementFactor *= destinationSubcellSize(d) / sourceSubcellSize(d);
//    }
//  }
  double sourceSubcellVolume = tarch::la::volume(sourceSubcellSize);
  double destinationSubcellVolume = tarch::la::volume(destinationSubcellSize);

  dfor(subcellIndexInFace, face) {
    tarch::la::Vector<DIMENSIONS, int> subcellIndexInSourcePatch = subcellIndexInFace + offset;
    tarch::la::Vector<DIMENSIONS, int> ghostlayerSubcellIndexInSourcePatch = subcellIndexInSourcePatch;
    ghostlayerSubcellIndexInSourcePatch(dimension) += direction;

    tarch::la::Vector<DIMENSIONS, double> subcellPositionInSourcePatch = sourceSubgrid.getSubcellPosition(subcellIndexInSourcePatch);
    tarch::la::Vector<DIMENSIONS, double> neighboringSubcellCenterInDestinationPatch = subcellPositionInSourcePatch;
    neighboringSubcellCenterInDestinationPatch(dimension) += destinationSubcellSize(dimension) * direction * 0.5;

    tarch::la::Vector<DIMENSIONS, int> neighboringSubcellIndexInDestinationPatch =
        (tarch::la::multiplyComponents(
             neighboringSubcellCenterInDestinationPatch - destinationSubgrid.getPosition(),
             tarch::la::invertEntries(destinationSubcellSize)
        )).convertScalar<int>();
    tarch::la::Vector<DIMENSIONS, int> ghostlayerSubcellIndexInDestinationPatch = neighboringSubcellIndexInDestinationPatch;
    ghostlayerSubcellIndexInDestinationPatch(dimension) -= direction;

    logDebug("applyFluxCorrection", "Correcting from cell " << subcellIndexInSourcePatch);

    dfor(neighborOffset, searchRegion) {
      tarch::la::Vector<DIMENSIONS, int> adjacentSubcellIndexInDestinationPatch = neighboringSubcellIndexInDestinationPatch + neighborOffset;

      correctFluxBetweenCells(
        dimension,
        direction,
        timestepOverlap,
        sourceSubgrid,
        destinationSubgrid,
        sourceAccessor,
        destinationAccessor,
        sourceTimeIntervals,
        destinationTimeIntervals,
        destinationSubcellVolume,
        sourceSubcellSize,
        destinationSubcellSize,
        subcellIndexInSourcePatch,
        ghostlayerSubcellIndexInSourcePatch,
        adjacentSubcellIndexInDestinationPatch,
        ghostlayerSubcellIndexInDestinationPatch
      );
    }
  }
  logTraceOut("applyCoarseGridCorrection");
}

void peanoclaw::interSubgridCommunication::VelocityFluxCorrection::computeFluxes(Patch& subgrid) const {

}





