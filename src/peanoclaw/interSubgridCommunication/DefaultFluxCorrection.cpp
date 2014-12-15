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

void peanoclaw::interSubgridCommunication::DefaultFluxCorrection::correctFluxBetweenCells(
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
//  if(
//      !tarch::la::oneGreater(tarch::la::Vector<DIMENSIONS, int>(0), adjacentSubcellIndexInDestinationPatch)
//  && !tarch::la::oneGreaterEquals(adjacentSubcellIndexInDestinationPatch, destinationSubgrid.getSubdivisionFactor())
//  ) {
//    //Get interface region
//    double interfaceRegion = calculateOverlappingRegion(
//        sourceSubgrid.getSubcellPosition(subcellIndexInSourcePatch),
//        sourceSubcellSize,
//        destinationSubgrid.getSubcellPosition(adjacentSubcellIndexInDestinationPatch),
//        destinationSubcellSize,
//        dimension
//    );
//
////    for(int unknown = 0; unknown < sourceSubgrid.getUnknownsPerSubcell(); unknown++) {
//    for(int unknown = sourceSubgrid.getUnknownsPerSubcell()-1; unknown >= 0; unknown--) {
////    for(int unknown = 0; unknown < 1; unknown++) {
//
//      double sourceDensityUOld = sourceAccessor.getValueUOld(subcellIndexInSourcePatch, 0);
//      double sourceVelocityUOld = direction * sourceAccessor.getValueUOld(subcellIndexInSourcePatch, dimension+1) / sourceDensityUOld;
//      double sourceDensityUNew = sourceAccessor.getValueUNew(subcellIndexInSourcePatch, 0);
//      double sourceVelocityUNew = direction * sourceAccessor.getValueUNew(subcellIndexInSourcePatch, dimension+1) / sourceDensityUNew;
//      double sourceDensityGhostCell = sourceAccessor.getValueUOld(ghostlayerSubcellIndexInSourcePatch, 0);
//      double sourceVelocityGhostCell = direction * sourceAccessor.getValueUOld(ghostlayerSubcellIndexInSourcePatch, dimension+1) / sourceDensityGhostCell;
//
//      double destinationDensityUOld = destinationAccessor.getValueUOld(adjacentSubcellIndexInDestinationPatch, 0);
//      double destinationVelocityUOld = direction * destinationAccessor.getValueUOld(adjacentSubcellIndexInDestinationPatch, dimension+1) / destinationDensityUOld;
//      double destinationDensityUNew = destinationAccessor.getValueUNew(adjacentSubcellIndexInDestinationPatch, 0);
//      double destinationVelocityUNew = direction * destinationAccessor.getValueUNew(adjacentSubcellIndexInDestinationPatch, dimension+1) / destinationDensityUNew;
//      double destinationDensityGhostCell = destinationAccessor.getValueUOld(ghostlayerSubcellIndexInDestinationPatch, 0);
//      double destinationVelocityGhostCell = direction * destinationAccessor.getValueUOld(ghostlayerSubcellIndexInDestinationPatch, dimension+1) / destinationDensityGhostCell;
//
//
//      //TODO This depends on the application, since we assume that variable 0 holds the water height of the shallow water
//      //equation or pressure for the euler equation and variables 1, 2, ... hold the impulse in x, y, ... direction
//      //Estimate the flux through the interface from fine to coarse grid, once from the point of view of the fine subcell and
//      //once from the of the coarse subcell
//      double sourceValue = sourceAccessor.getValueUOld(subcellIndexInSourcePatch, unknown);
//      double sourceValueGhostCell = sourceAccessor.getValueUOld(ghostlayerSubcellIndexInSourcePatch, unknown);
//      double sourceFlux
////            = sourceValue * sourceVelocityUNew * interfaceRegion;
////            = (sourceAccessor.getValueUOld(subcellIndexInSourcePatch, unknown) + sourceAccessor.getValueUOld(ghostlayerSubcellIndexInSourcePatch, unknown)) / 2.0
////              * sourceAccessor.getValueUOld(subcellIndexInSourcePatch, 1 + dimension) * interfaceRegion;
//            = 0.5 * (sourceValue * sourceVelocityUOld + sourceValueGhostCell * sourceVelocityGhostCell) * interfaceRegion;
//
//      double destinationValueUOld = destinationAccessor.getValueUOld(adjacentSubcellIndexInDestinationPatch, unknown);
//      double destinationValueUNew = destinationAccessor.getValueUNew(adjacentSubcellIndexInDestinationPatch, unknown);
//      double destinationValueGhostCell = destinationAccessor.getValueUOld(ghostlayerSubcellIndexInDestinationPatch, unknown);
////      double destinationImpulseUOld = destinationAccessor.getValueUOld(adjacentSubcellIndexInDestinationPatch, 1+dimension);
////      double destinationImpulseUNew = destinationAccessor.getValueUNew(adjacentSubcellIndexInDestinationPatch, 1+dimension);
//      double timeFactor = 0.0; //(sourceTimeIntervals.getCurrentTime() - destinationTimeIntervals.getCurrentTime()) / destinationTimeIntervals.getTimestepSize();
//      double destinationValue = destinationValueUOld * (1.0-timeFactor) + destinationValueUNew * timeFactor;
//      double destinationVelocity = destinationVelocityUOld * (1.0-timeFactor) + destinationVelocityUNew * timeFactor;
//      double destinationFlux
////        = destinationValue * destinationVelocity * interfaceRegion;
//        = 0.5 * (destinationValue * destinationVelocityUOld + destinationValueGhostCell * destinationVelocityGhostCell) * interfaceRegion;
//
////      assertion1(tarch::la::greaterEquals(timeFactor, 0.0) && tarch::la::smallerEquals(timeFactor, 1.0), timeFactor);
//
//      //double destinationValue = destinationAccessor.getValueUOld(adjacentSubcellIndexInDestinationPatch, unknown);
////      double destinationFlux = destinationValue * destinationAccessor.getValueUOld(adjacentSubcellIndexInDestinationPatch, 1 + dimension) / destinationValue * interfaceRegion;
////        = (destinationAccessor.getValueUOld(adjacentSubcellIndexInDestinationPatch, unknown) + destinationAccessor.getValueUOld(ghostlayerSubcellIndexInDestinationPatch, unknown)) / 2.0
////          * destinationAccessor.getValueUOld(adjacentSubcellIndexInDestinationPatch, 1 + dimension) * interfaceRegion;
//
//      //Estimate the according transfered volume during the fine patch's timestep
//      double transferedSourceVolume = sourceFlux * timestepOverlap;
//      double transferedDestinationVolume = destinationFlux * timestepOverlap;
//      double delta = transferedSourceVolume - transferedDestinationVolume;
//
//      logDebug("applyFluxCorrection", "Correcting neighbor cell " << adjacentSubcellIndexInDestinationPatch << " from cell " << subcellIndexInSourcePatch << " with interfaceRegion=" << interfaceRegion << std::endl
//          << "\tu0=" << sourceSubgrid.getAccessor().getValueUNew(subcellIndexInSourcePatch, 0) << " u1=" << sourceSubgrid.getAccessor().getValueUNew(subcellIndexInSourcePatch, 1) << " u2=" << sourceSubgrid.getAccessor().getValueUNew(subcellIndexInSourcePatch, 2) << std::endl
//          << "\ttimestepOverlap=" << timestepOverlap << std::endl
//          << "\tsourceFlux=" << sourceFlux << std::endl
//          << "\tdestinationFlux=" << destinationFlux << std::endl
//          << "\ttransferedSourceVolume=" << transferedSourceVolume << std::endl
//          << "\ttransferedDestinationVolume=" << transferedDestinationVolume << std::endl
//          << "\tdelta=" << delta << std::endl
//          << "\told u=" << destinationSubgrid.getAccessor().getValueUNew(adjacentSubcellIndexInDestinationPatch, 0) << std::endl
//          << "\tnew u=" << destinationSubgrid.getAccessor().getValueUNew(adjacentSubcellIndexInDestinationPatch, 0) + delta);
//
//      //TODO unterweg debug
////      std::cout << "sourceFlux=" << fineGridFlux << " destinationFlux=" << coarseGridFlux
////          << " value=" << destinationAccessor.getValueUNew(adjacentSubcellIndexInDestinationPatch, unknown) << " delta=" << delta
////          << " subcellVolume=" << sourceSubcellVolume << " interface=" << interfaceRegion
////          << " h_s=" << sourceAccessor.getValueUOld(subcellIndexInSourcePatch, unknown) << " u_s=" << sourceAccessor.getValueUOld(subcellIndexInSourcePatch, 1 + dimension)
////          << " h_d=" << destinationAccessor.getValueUOld(adjacentSubcellIndexInDestinationPatch, unknown) << " u_d=" << destinationAccessor.getValueUOld(adjacentSubcellIndexInDestinationPatch, 1 + dimension) << std::endl;
//
//      destinationAccessor.setValueUNew(adjacentSubcellIndexInDestinationPatch, unknown,
//          destinationAccessor.getValueUNew(adjacentSubcellIndexInDestinationPatch, unknown) - delta / destinationSubcellVolume);
//
//      //TODO unterweg debug
//      assertion7(unknown!=0 || destinationAccessor.getValueUNew(adjacentSubcellIndexInDestinationPatch, unknown) > 0,
//          sourceSubgrid,
//          destinationSubgrid,
//          adjacentSubcellIndexInDestinationPatch,
//          destinationAccessor.getValueUNew(adjacentSubcellIndexInDestinationPatch, unknown),
//          timestepOverlap,
//          sourceSubgrid.toStringUOldWithGhostLayer(),
//          destinationSubgrid.toStringUOldWithGhostLayer()
//      );
//    }
//  }
}

peanoclaw::interSubgridCommunication::DefaultFluxCorrection::~DefaultFluxCorrection() {
}

void peanoclaw::interSubgridCommunication::DefaultFluxCorrection::computeFluxes(Patch& subgrid) const {
  switch(subgrid.getUnknownsPerSubcell()) {
    case 1:
      peanoclaw::interSubgridCommunication::DefaultFluxCorrectionTemplate<1> fluxCorrection1;
      fluxCorrection1.computeFluxes(subgrid);
      break;
    case 2:
      peanoclaw::interSubgridCommunication::DefaultFluxCorrectionTemplate<2> fluxCorrection2;
      fluxCorrection2.computeFluxes(subgrid);
      break;
    case 3:
      peanoclaw::interSubgridCommunication::DefaultFluxCorrectionTemplate<3> fluxCorrection3;
      fluxCorrection3.computeFluxes(subgrid);
      break;
    case 4:
      peanoclaw::interSubgridCommunication::DefaultFluxCorrectionTemplate<4> fluxCorrection4;
      fluxCorrection4.computeFluxes(subgrid);
      break;
    case 5:
      peanoclaw::interSubgridCommunication::DefaultFluxCorrectionTemplate<5> fluxCorrection5;
      fluxCorrection5.computeFluxes(subgrid);
      break;
    case 6:
      peanoclaw::interSubgridCommunication::DefaultFluxCorrectionTemplate<6> fluxCorrection6;
      fluxCorrection6.computeFluxes(subgrid);
      break;
    case 7:
      peanoclaw::interSubgridCommunication::DefaultFluxCorrectionTemplate<7> fluxCorrection7;
      fluxCorrection7.computeFluxes(subgrid);
      break;
    case 8:
      peanoclaw::interSubgridCommunication::DefaultFluxCorrectionTemplate<8> fluxCorrection8;
      fluxCorrection8.computeFluxes(subgrid);
      break;
    case 9:
      peanoclaw::interSubgridCommunication::DefaultFluxCorrectionTemplate<9> fluxCorrection9;
      fluxCorrection9.computeFluxes(subgrid);
      break;
    case 10:
      peanoclaw::interSubgridCommunication::DefaultFluxCorrectionTemplate<10> fluxCorrection10;
      fluxCorrection10.computeFluxes(subgrid);
      break;
  }
}

void peanoclaw::interSubgridCommunication::DefaultFluxCorrection::applyCorrection(
    Patch& sourceSubgrid,
    Patch& destinationSubgrid,
    int dimension,
    int direction
) const {
  switch(sourceSubgrid.getUnknownsPerSubcell()) {
      case 1:
        peanoclaw::interSubgridCommunication::DefaultFluxCorrectionTemplate<1> fluxCorrection1;
        fluxCorrection1.applyCorrection(sourceSubgrid, destinationSubgrid, dimension, direction);
        break;
      case 2:
        peanoclaw::interSubgridCommunication::DefaultFluxCorrectionTemplate<2> fluxCorrection2;
        fluxCorrection2.applyCorrection(sourceSubgrid, destinationSubgrid, dimension, direction);
        break;
      case 3:
        peanoclaw::interSubgridCommunication::DefaultFluxCorrectionTemplate<3> fluxCorrection3;
        fluxCorrection3.applyCorrection(sourceSubgrid, destinationSubgrid, dimension, direction);
        break;
      case 4:
        peanoclaw::interSubgridCommunication::DefaultFluxCorrectionTemplate<4> fluxCorrection4;
        fluxCorrection4.applyCorrection(sourceSubgrid, destinationSubgrid, dimension, direction);
        break;
      case 5:
        peanoclaw::interSubgridCommunication::DefaultFluxCorrectionTemplate<5> fluxCorrection5;
        fluxCorrection5.applyCorrection(sourceSubgrid, destinationSubgrid, dimension, direction);
        break;
      case 6:
        peanoclaw::interSubgridCommunication::DefaultFluxCorrectionTemplate<6> fluxCorrection6;
        fluxCorrection6.applyCorrection(sourceSubgrid, destinationSubgrid, dimension, direction);
        break;
      case 7:
        peanoclaw::interSubgridCommunication::DefaultFluxCorrectionTemplate<7> fluxCorrection7;
        fluxCorrection7.applyCorrection(sourceSubgrid, destinationSubgrid, dimension, direction);
        break;
      case 8:
        peanoclaw::interSubgridCommunication::DefaultFluxCorrectionTemplate<8> fluxCorrection8;
        fluxCorrection8.applyCorrection(sourceSubgrid, destinationSubgrid, dimension, direction);
        break;
      case 9:
        peanoclaw::interSubgridCommunication::DefaultFluxCorrectionTemplate<9> fluxCorrection9;
        fluxCorrection9.applyCorrection(sourceSubgrid, destinationSubgrid, dimension, direction);
        break;
      case 10:
        peanoclaw::interSubgridCommunication::DefaultFluxCorrectionTemplate<10> fluxCorrection10;
        fluxCorrection10.applyCorrection(sourceSubgrid, destinationSubgrid, dimension, direction);
        break;
    }
}


