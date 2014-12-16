/*
 * FullSWOF2D.cpp
 *
 *  Created on: Feb 18, 2012
 *      Author: kristof
 */

#include "peanoclaw/native/FullSWOF2D.h"

#include "peanoclaw/grid/aspects/BoundaryIterator.h"
#include "peanoclaw/Patch.h"
#include "peanoclaw/native/scenarios/FullSWOF2DBoundaryCondition.h"

#include "tarch/timing/Watch.h"
#include "tarch/parallel/Node.h"

#ifndef CHOICE_SCHEME_HPP
#include "choice_scheme.hpp"
#endif

tarch::logging::Log peanoclaw::native::FullSWOF2D::_log("peanoclaw::native::FullSWOF2D");
peanoclaw::native::scenarios::FullSWOF2DBoundaryCondition
  peanoclaw::native::FullSWOF2D::_interSubgridBoundaryCondition(0, 0, 0);

peanoclaw::native::FullSWOF2D::FullSWOF2D(
  peanoclaw::native::scenarios::SWEScenario& scenario,
  peanoclaw::interSubgridCommunication::DefaultTransfer* transfer,
  peanoclaw::interSubgridCommunication::Interpolation*  interpolation,
  peanoclaw::interSubgridCommunication::Restriction*    restriction,
  peanoclaw::interSubgridCommunication::FluxCorrection* fluxCorrection
) : Numerics(transfer, interpolation, restriction, fluxCorrection),
_totalSolverCallbackTime(0.0),
_scenario(scenario),
_wrapperScheme(0)
{
  //import_array();

    // TODO: manually set number of cells and so on ...
}

peanoclaw::native::FullSWOF2D::~FullSWOF2D()
{
}


void peanoclaw::native::FullSWOF2D::initializePatch(
  Patch& patch
) {
  logTraceIn( "initializePatch(...)");

  _scenario.initializePatch(patch);
 
  logTraceOutWith1Argument( "initializePatch(...)", demandedMeshWidth);
}

void peanoclaw::native::FullSWOF2D::solveTimestep(
  Patch& subgrid,
  double maximumTimestepSize,
  bool useDimensionalSplitting,
  tarch::la::Vector<DIMENSIONS_TIMES_TWO, bool> domainBoundaryFlags
) {
  logTraceInWith2Arguments( "solveTimestep(...)", maximumTimestepSize, useDimensionalSplitting);
  #ifdef PEANOCLAW_FULLSWOF2D
  assertion2(tarch::la::greater(maximumTimestepSize, 0.0), "Timestepsize == 0 should be checked outside.", subgrid.getTimeIntervals().getMinimalNeighborTimeConstraint());

  tarch::timing::Watch fullswof2dWatch("", "", false);
  fullswof2dWatch.startTimer();
//  double dtAndEstimatedNextDt[2];
  tarch::la::Vector<DIMENSIONS,double> meshwidth = subgrid.getSubcellSize();
  tarch::la::Vector<DIMENSIONS,int> subdivisionFactor = subgrid.getSubdivisionFactor();
  int peanoClawGhostlayerWidth = subgrid.getGhostlayerWidth();
  int fullswof2DGhostlayerWidth = 1; //Ghostlayer width for FullSWOF is always 1 cell.
 
  double dt; // = std::min(dt_used, maximumTimestepSize);
  double estimatedNextTimestepSize; // = scheme->getMaxTimestep();

  // kick off the computation here -----
#if 1
  {
    tarch::la::Vector<DIMENSIONS_TIMES_TWO, int> margin(0);
    if(domainBoundaryFlags[0]) margin[0] = peanoClawGhostlayerWidth - 1;
    if(domainBoundaryFlags[1]) margin[1] = peanoClawGhostlayerWidth - 1;
    if(domainBoundaryFlags[2]) margin[2] = peanoClawGhostlayerWidth - 1;
    if(domainBoundaryFlags[3]) margin[3] = peanoClawGhostlayerWidth - 1;

      FullSWOF2D_Parameters par(
          fullswof2DGhostlayerWidth,
          subdivisionFactor(0) + 2*peanoClawGhostlayerWidth - 2*fullswof2DGhostlayerWidth - margin[0] - margin[1],
          subdivisionFactor(1) + 2*peanoClawGhostlayerWidth - 2*fullswof2DGhostlayerWidth - margin[2] - margin[3],
          meshwidth(0),
          meshwidth(1),
          _scenario.getDomainSize(),
          1.0, //endTime
          _scenario.enableRain(),
          _scenario.getFrictionCoefficient(),
          domainBoundaryFlags[0] ? _scenario.getBoundaryCondition(0, false) : _interSubgridBoundaryCondition,
          domainBoundaryFlags[1] ? _scenario.getBoundaryCondition(0, true) : _interSubgridBoundaryCondition,
          domainBoundaryFlags[2] ? _scenario.getBoundaryCondition(1, false) : _interSubgridBoundaryCondition,
          domainBoundaryFlags[3] ? _scenario.getBoundaryCondition(1, true) : _interSubgridBoundaryCondition
//          _interSubgridBoundaryCondition,
//          _interSubgridBoundaryCondition,
//          _interSubgridBoundaryCondition,
//          _interSubgridBoundaryCondition
//              _scenario.getBoundaryCondition(0, false),
//              _scenario.getBoundaryCondition(0, true),
//              _scenario.getBoundaryCondition(1, false),
//              _scenario.getBoundaryCondition(1, true)
      );
      //std::cout << "parameters read (meshwidth): " << par.get_dx() << " vs " << meshwidth(0) << " and " << par.get_dy() << " vs " << meshwidth(1) << std::endl;
      //std::cout << "parameters read (cells): " << par.get_Nxcell() << " vs " << subdivisionFactor(0) << " and " << par.get_Nycell() << " vs " << subdivisionFactor(1) << std::endl;

//      Choice_scheme *wrapper_scheme = 0;
      Scheme *scheme = 0;

      do {
        if(_wrapperScheme != 0) {
          delete _wrapperScheme;
        }
        _wrapperScheme = new Choice_scheme(par);
        scheme = _wrapperScheme->getInternalScheme();

        // kick off computation!
        scheme->setTimestep(maximumTimestepSize);
  //        scheme->setMaxTimestep(maximumTimestepSize); // TODO: maximumTimstepSize is ignored and the "real" maxTimestep is computed
        scheme->setMaxTimestep(std::min(maximumTimestepSize, 0.5));
        scheme->usePeanoClaw();

        // overwrite internal values
        copyPatchToScheme(subgrid, scheme, margin);

        scheme->resetTimings();
        scheme->resetN();

        struct timeval start;
        gettimeofday(&start, NULL);

        //TODO unterweg debug
        std::cout << subgrid.getPosition() << std::endl;
        _wrapperScheme->calcul();

        struct timeval stop;
        gettimeofday(&stop, NULL);

//        double time = (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec) / 1000000.0;
        //std::cout << "calculation took " << time << std::endl;

        if (scheme->getVerif() == 0) {
            std::cout << "scheme retry activated!" << std::endl;
//            throw "";
            scheme->setMaxTimestep(scheme->getTimestep());
            maximumTimestepSize = scheme->getTimestep();
        }
      } while (scheme->getVerif() == 0); // internal error detection of FullSWOF2D

      // copy back internal values but skip ghostlayer
      copySchemeToPatch(scheme, subgrid, margin);

      // working scheme
      //double dt = std::min(scheme->getTimestep(), maximumTimestepSize);
      //double estimatedNextTimestepSize = scheme->getTimestep(); // dt;
     
      // strange scheme almost reasonable
      //double dt = std::min(scheme->getMaxTimestep(), maximumTimestepSize);
      //double estimatedNextTimestepSize = scheme->getTimestep();
     
      // looks VERY good (was used for the nice pictures)
      dt = std::min(scheme->getTimestep(), maximumTimestepSize);
      estimatedNextTimestepSize = scheme->getMaxTimestep();

      // BENCHMARK
      //dt = 0.00001;
      //estimatedNextTimestepSize = 0.00001;

      //std::cout << "\nComputation finished!" << endl;
      delete _wrapperScheme;
      _wrapperScheme = 0;
      // computation is done -> back to peanoclaw 
  }
#endif

#if 0
  { 
        unsigned int strideinfo[3];
        const int nr_patches = 1;
        const int patchid = 0;

        MekkaFlood_solver::InputArrays input;
        MekkaFlood_solver::TempArrays temp;
        MekkaFlood_solver::Constants constants(subdivisionFactor(0)+2, subdivisionFactor(1)+2, meshwidth(0), meshwidth(1));

        MekkaFlood_solver::initializeStrideinfo(constants, 3, strideinfo);
        MekkaFlood_solver::allocateInput(nr_patches, 3, strideinfo, input);
        MekkaFlood_solver::allocateTemp(nr_patches, 3, strideinfo, temp);

        copyPatchToSet(subgrid, strideinfo,input, temp);
        double dt_used = MekkaFlood_solver::calcul(patchid, 3, strideinfo, input, temp, constants, maximumTimestepSize);
        //std::cout << "dt_used " << dt_used << " maximumTimestepSize " << maximumTimestepSize << std::endl;
        copySetToPatch(strideinfo,input, temp, subgrid);

        dt = std::min(dt_used, maximumTimestepSize);
        estimatedNextTimestepSize = maximumTimestepSize;

        MekkaFlood_solver::freeInput(input);
        MekkaFlood_solver::freeTemp(temp);
  }
#endif

  fullswof2dWatch.stopTimer();
  _totalSolverCallbackTime += fullswof2dWatch.getCalendarTime();

  assertion4(
      tarch::la::greater(subgrid.getTimeIntervals().getTimestepSize(), 0.0)
      || tarch::la::greater(estimatedNextTimestepSize, 0.0)
      || tarch::la::equals(maximumTimestepSize, 0.0)
      || tarch::la::equals(subgrid.getTimeIntervals().getEstimatedNextTimestepSize(), 0.0),
      subgrid, maximumTimestepSize, estimatedNextTimestepSize, subgrid.toStringUNew());
  assertion(subgrid.getTimeIntervals().getTimestepSize() < std::numeric_limits<double>::infinity());

  if (tarch::la::greater(dt, 0.0)) {
    subgrid.getTimeIntervals().advanceInTime();
    subgrid.getTimeIntervals().setTimestepSize(dt);
  }
  subgrid.getTimeIntervals().setEstimatedNextTimestepSize(estimatedNextTimestepSize);

  //TODO unterweg debug
//  std::cout << "Timestep" << std::endl;
//  std::cout << domainBoundaryFlags << " " << subgrid.toString() << std::endl;
//  std::cout << "UOld" << std::endl << subgrid.toStringUOldWithGhostLayer() << std::endl << "UNew" << std::endl << subgrid.toStringUNew() << std::endl;
//  throw "";

  #endif
  logTraceOut( "solveTimestep(...)");
}

tarch::la::Vector<DIMENSIONS, double> peanoclaw::native::FullSWOF2D::getDemandedMeshWidth(Patch& patch, bool isInitializing) {
  return _scenario.computeDemandedMeshWidth(patch, isInitializing);
}

void peanoclaw::native::FullSWOF2D::addPatchToSolution(Patch& patch) {
}

void peanoclaw::native::FullSWOF2D::fillBoundaryLayer(Patch& subgrid, int dimension, bool setUpper) {
  logTraceInWith3Arguments("fillBoundaryLayerInPyClaw", subgrid, dimension, setUpper);

  logDebug("fillBoundaryLayerInPyClaw", "Setting left boundary for " << subgrid.getPosition() << ", dim=" << dimension << ", setUpper=" << setUpper);

   //std::cout << "------ setUpper " << setUpper << " dimension " << dimension << std::endl;
   //std::cout << patch << std::endl;
   //std::cout << "++++++" << std::endl;
   //std::cout << patch.toStringUOldWithGhostLayer() << std::endl;
   //std::cout << "||||||" << std::endl;
 
#if 1
  int ghostlayerWidth = subgrid.getGhostlayerWidth();
  int fullswofGhostlayerWidth = ghostlayerWidth;
  // implement a wall boundary
  tarch::la::Vector<DIMENSIONS, int> src_subcellIndex;
  tarch::la::Vector<DIMENSIONS, int> dest_subcellIndex;
  peanoclaw::grid::SubgridAccessor accessor = subgrid.getAccessor();

  for(int layer = 1; layer <= subgrid.getGhostlayerWidth(); layer++) {
    for (int i = -fullswofGhostlayerWidth; i < subgrid.getSubdivisionFactor()(1-dimension)+fullswofGhostlayerWidth; i++) {
      src_subcellIndex(dimension) = setUpper ? subgrid.getSubdivisionFactor()(dimension)-1 - layer : layer;
      src_subcellIndex(1-dimension) = i;
      dest_subcellIndex(dimension) = setUpper ? subgrid.getSubdivisionFactor()(dimension)-1 + layer : -layer;
      dest_subcellIndex(1-dimension) = i;
      accessor.setParameterWithGhostlayer(dest_subcellIndex, 0, accessor.getParameterWithGhostlayer(src_subcellIndex, 0));
      for(int unknown = 0; unknown < subgrid.getUnknownsPerSubcell(); unknown++) {
        accessor.setValueUOld(dest_subcellIndex, unknown, accessor.getValueUOld(src_subcellIndex, unknown));
      }
      //Invert impulse for dimension
      //accessor.setValueUOld(dest_subcellIndex, 1+dimension, -accessor.getValueUOld(dest_subcellIndex, 1+dimension));
    }
  }
#endif

   //std::cout << "++++++" << std::endl;
   //std::cout << patch.toStringUOldWithGhostLayer() << std::endl;
   //std::cout << "||||||" << std::endl;

  //Fill scenario boundary condition
  peanoclaw::grid::aspects::BoundaryIterator<peanoclaw::native::scenarios::SWEScenario> scenarioBoundaryIterator(_scenario);
  scenarioBoundaryIterator.iterate(subgrid, accessor, dimension, setUpper);

  logTraceOut("fillBoundaryLayer");
}

void peanoclaw::native::FullSWOF2D::update(Patch& finePatch) {
    _scenario.update(finePatch);
}

#ifdef PEANOCLAW_FULLSWOF2D
void peanoclaw::native::FullSWOF2D::copyPatchToScheme(Patch& patch, Scheme* scheme, tarch::la::Vector<DIMENSIONS_TIMES_TWO, int> margin) {
  tarch::la::Vector<DIMENSIONS,int> subdivisionFactor = patch.getSubdivisionFactor();
  tarch::la::Vector<DIMENSIONS,int> subcellIndex;
  peanoclaw::grid::SubgridAccessor& accessor = patch.getAccessor();

  // FullSWOF2D has a mixture of 0->nxcell+1 and 1->nxcell
  int ghostlayerWidth = patch.getGhostlayerWidth();
  int fullswof2DGhostlayerWidth = 1;

  TAB& h = scheme->getH();
  TAB& u = scheme->getU();
  TAB& v = scheme->getV();
  TAB& z = scheme->getZ();
//  for (int x = -ghostlayerWidth; x < subdivisionFactor(0)+ghostlayerWidth; x++) {
//        for (int y = -ghostlayerWidth; y < subdivisionFactor(1)+ghostlayerWidth; y++) {
  for (int x = 0; x < subdivisionFactor(0)+2*ghostlayerWidth-margin[0]-margin[1]; x++) {
    for (int y = 0; y < subdivisionFactor(1)+2*ghostlayerWidth-margin[2]-margin[3]; y++) {
      subcellIndex(0) = x - ghostlayerWidth + margin[0];
      subcellIndex(1) = y - ghostlayerWidth + margin[2];
      /** Water height.*/
      h[x][y] = accessor.getValueUOld(subcellIndex, 0);
      /** X Velocity.*/
      u[x][y] = accessor.getValueUOld(subcellIndex, 1);
      /** Y Velocity.*/
      v[x][y] = accessor.getValueUOld(subcellIndex, 2);
      /** Topography.*/
      z[x][y] = accessor.getParameterWithGhostlayer(subcellIndex, 0);
    }
  }

#if 0 // TODO: we probably need this
  /** compute Discharge. (1->nxcell) */
  TAB& q1 = scheme->getQ1();

  for (int x = -1; x < subdivisionFactor(0)+1; x++) {
    for (int y = -1; y < subdivisionFactor(1)+1; y++) {
        subcellIndex(0) = x;
        subcellIndex(1) = y;
        // we have to initialize this because the FullSWOF2D does not compute the momentum on the ghostlayer
        double q = patch.getAccessor().getValueUOld(subcellIndex, 0) * patch.getAccessor().getValueUOld(subcellIndex, 1);
        q1[x+fullswofGhostlayerWidth][y+fullswofGhostlayerWidth] = q;

    }
  }

  /** compute Discharge. (1->nycell)*/
  TAB& q2 = scheme->getQ2();
  for (int x = -1; x < subdivisionFactor(0)+1; x++) {
    for (int y = -1; y < subdivisionFactor(1)+1; y++) {
        subcellIndex(0) = x;
        subcellIndex(1) = y;
        double q = patch.getAccessor().getValueUOld(subcellIndex, 0) * patch.getAccessor().getValueUOld(subcellIndex, 2);
        // we have to initialize this because the FullSWOF2D does not compute the momentum on the ghostlayer
        q2[x+fullswofGhostlayerWidth][y+fullswofGhostlayerWidth] = q;
    }
  }
#endif
}

void peanoclaw::native::FullSWOF2D::copySchemeToPatch(Scheme* scheme, Patch& patch, tarch::la::Vector<DIMENSIONS_TIMES_TWO, int> margin) {
  tarch::la::Vector<DIMENSIONS,int> subdivisionFactor = patch.getSubdivisionFactor();
  tarch::la::Vector<DIMENSIONS,int> subcellIndex;
  tarch::la::Vector<DIMENSIONS,int> fullswof2DSubcellIndex;

  int peanoClawGhostlayerWidth = patch.getGhostlayerWidth();
  int fullswof2DGhostlayerWidth = 1;

  //Copy inner part of subgrid
  TAB& h = scheme->getH();
  TAB& u = scheme->getU();
  TAB& v = scheme->getV();
  TAB& z = scheme->getZ();
  for (int x = 0; x < subdivisionFactor(0); x++) {
    for (int y = 0; y < subdivisionFactor(1); y++) {
      subcellIndex(0) = x;
      subcellIndex(1) = y;
      fullswof2DSubcellIndex(0) = x + peanoClawGhostlayerWidth - margin[0];
      fullswof2DSubcellIndex(1) = y + peanoClawGhostlayerWidth - margin[2];

      //TODO unterweg debug
//      std::cout << "Copying " << fullswof2DSubcellIndex[0] << "," << fullswof2DSubcellIndex[1] << ":" << h[fullswof2DSubcellIndex[0]][fullswof2DSubcellIndex[1]]
//         << " to " << subcellIndex << std::endl;

      /** Water height after one step of the scheme.*/
      patch.getAccessor().setValueUNew(subcellIndex, 0, h[fullswof2DSubcellIndex[0]][fullswof2DSubcellIndex[1]]);
      /** X Velocity after one step of the scheme.*/
      patch.getAccessor().setValueUNew(subcellIndex, 1, u[fullswof2DSubcellIndex[0]][fullswof2DSubcellIndex[1]]);
      /** Y Velocity after one step of the scheme.*/
      patch.getAccessor().setValueUNew(subcellIndex, 2, v[fullswof2DSubcellIndex[0]][fullswof2DSubcellIndex[1]]);
      /** Topography.*/
      patch.getAccessor().setParameterWithGhostlayer(subcellIndex, 0, z[fullswof2DSubcellIndex[0]][fullswof2DSubcellIndex[1]]);
    }
  }
 
#if 0 // TODO: we probably needs this
  /** compute Discharge. (1->nxcell) */
  TAB& q1 = scheme->getQ1();
  for (int x = 0; x < subdivisionFactor(0); x++) {
    for (int y = 0; y < subdivisionFactor(1); y++) {
        subcellIndex(0) = x;
        subcellIndex(1) = y;
        patch.getAccessor().setValueUNew(subcellIndex, 4, q1[x+fullswofGhostlayerWidth][y+fullswofGhostlayerWidth]);
    }
  }

  /** compute Discharge. (1->nycell)*/
  TAB& q2 = scheme->getQ2();
  for (int x = 0; x < subdivisionFactor(0); x++) {
    for (int y = 0; y < subdivisionFactor(1); y++) {
        subcellIndex(0) = x;
        subcellIndex(1) = y;
        patch.getAccessor().setValueUNew(subcellIndex, 5, q2[x+fullswofGhostlayerWidth][y+fullswofGhostlayerWidth]);
    }
  }
#endif

  //Copy ghostlayer for debugging
  for (int x = 0; x < subdivisionFactor(0)+2*peanoClawGhostlayerWidth-margin[0]-margin[1]; x++) {
    for (int y = 0; y < subdivisionFactor(1)+2*peanoClawGhostlayerWidth-margin[2]-margin[3]; y++) {
      subcellIndex(0) = x - peanoClawGhostlayerWidth + margin[0];
      subcellIndex(1) = y - peanoClawGhostlayerWidth + margin[2];
      if(!tarch::la::allGreaterEquals(subcellIndex, 0) || tarch::la::oneGreaterEquals(subcellIndex, subdivisionFactor)) {
        /** Water height.*/
//        patch.getAccessor().setValueUOld(subcellIndex, 0, h[x][y]);
//        /** X Velocity.*/
//        patch.getAccessor().setValueUOld(subcellIndex, 1, u[x][y]);
//        /** Y Velocity.*/
//        patch.getAccessor().setValueUOld(subcellIndex, 2, v[x][y]);
      }
    }
  }

}
#endif

//void peanoclaw::native::FullSWOF2D::copyPatchToSet(Patch& patch, unsigned int *strideinfo, MekkaFlood_solver::InputArrays& input, MekkaFlood_solver::TempArrays& temp) {
//    const int patchid = 0; // TODO: make this generic
//
//  tarch::la::Vector<DIMENSIONS,int> subdivisionFactor = patch.getSubdivisionFactor();
//  tarch::la::Vector<DIMENSIONS,int> subcellIndex;
//
//  // FullSWOF2D has a mixture of 0->nxcell+1 and 1->nxcell
//  int ghostlayerWidth = patch.getGhostlayerWidth();
//  int fullswofGhostlayerWidth = ghostlayerWidth;
//
//  /** Water height.*/
//  double* h = input.h;
//  for (int x = -ghostlayerWidth; x < subdivisionFactor(0)+ghostlayerWidth; x++) {
//        for (int y = -ghostlayerWidth; y < subdivisionFactor(1)+ghostlayerWidth; y++) {
//            subcellIndex(0) = x;
//            subcellIndex(1) = y;
//
//            unsigned int index[3];
//            index[0] = y+ghostlayerWidth;
//            index[1] = x+ghostlayerWidth;
//            index[2] = patchid;
//            unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);
//
//            h[centerIndex] = patch.getAccessor().getValueUOld(subcellIndex, 0);
//        }
//  }
//
//  /** X Velocity.*/
//  double* u = input.u;
//  for (int x = -ghostlayerWidth; x < subdivisionFactor(0)+ghostlayerWidth; x++) {
//        for (int y = -ghostlayerWidth; y < subdivisionFactor(1)+ghostlayerWidth; y++) {
//            subcellIndex(0) = x;
//            subcellIndex(1) = y;
//
//            unsigned int index[3];
//            index[0] = y+ghostlayerWidth;
//            index[1] = x+ghostlayerWidth;
//            index[2] = patchid;
//            unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);
//
//            u[centerIndex] = patch.getAccessor().getValueUOld(subcellIndex, 1);
//        }
//  }
//
//  /** Y Velocity.*/
//  double* v = input.v;
//  for (int x = -ghostlayerWidth; x < subdivisionFactor(0)+ghostlayerWidth; x++) {
//        for (int y = -ghostlayerWidth; y < subdivisionFactor(1)+ghostlayerWidth; y++) {
//            subcellIndex(0) = x;
//            subcellIndex(1) = y;
//
//            unsigned int index[3];
//            index[0] = y+ghostlayerWidth;
//            index[1] = x+ghostlayerWidth;
//            index[2] = patchid;
//            unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);
//
//            v[centerIndex] = patch.getAccessor().getValueUOld(subcellIndex, 2);
//        }
//  }
//
//  /** Topography.*/
//  double* z = input.z;
//  for (int x = -ghostlayerWidth; x < subdivisionFactor(0)+ghostlayerWidth; x++) {
//        for (int y = -ghostlayerWidth; y < subdivisionFactor(1)+ghostlayerWidth; y++) {
//            subcellIndex(0) = x;
//            subcellIndex(1) = y;
//
//            unsigned int index[3];
//            index[0] = y+ghostlayerWidth;
//            index[1] = x+ghostlayerWidth;
//            index[2] = patchid;
//            unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);
//
//            z[centerIndex] = patch.getAccessor().getParameterWithGhostlayer(subcellIndex, 0);
//        }
//  }
//
//#if 1 // TODO: we probably need this
//  /** compute Discharge. (1->nxcell) */
//  double* q1 = temp.q1;
//  for (int x = -1; x < subdivisionFactor(0)+1; x++) {
//    for (int y = -1; y < subdivisionFactor(1)+1; y++) {
//        subcellIndex(0) = x;
//        subcellIndex(1) = y;
//
//        unsigned int index[3];
//        index[0] = y+fullswofGhostlayerWidth;
//        index[1] = x+fullswofGhostlayerWidth;
//        index[2] = patchid;
//        unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);
//
//        // we have to initialize this because the FullSWOF2D does not compute the momentum on the ghostlayer
//        double q = patch.getAccessor().getValueUOld(subcellIndex, 0) * patch.getAccessor().getValueUOld(subcellIndex, 1);
//        q1[centerIndex] = q;
//
//    }
//  }
//
//  /** compute Discharge. (1->nycell)*/
//  double* q2 = temp.q2;
//  for (int x = -1; x < subdivisionFactor(0)+1; x++) {
//    for (int y = -1; y < subdivisionFactor(1)+1; y++) {
//        subcellIndex(0) = x;
//        subcellIndex(1) = y;
//
//        unsigned int index[3];
//        index[0] = y+fullswofGhostlayerWidth;
//        index[1] = x+fullswofGhostlayerWidth;
//        index[2] = patchid;
//        unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);
//
//        double q = patch.getAccessor().getValueUOld(subcellIndex, 0) * patch.getAccessor().getValueUOld(subcellIndex, 2);
//        // we have to initialize this because the FullSWOF2D does not compute the momentum on the ghostlayer
//        q2[centerIndex] = q;
//    }
//  }
//#endif
//}

//void peanoclaw::native::FullSWOF2D::copySetToPatch(unsigned int *strideinfo, MekkaFlood_solver::InputArrays& input, MekkaFlood_solver::TempArrays& temp, Patch& patch) {
//  const int patchid = 0; // TODO: make this generic
//
//  tarch::la::Vector<DIMENSIONS,int> subdivisionFactor = patch.getSubdivisionFactor();
//  tarch::la::Vector<DIMENSIONS,int> subcellIndex;
//
//  int ghostlayerWidth = patch.getGhostlayerWidth();
//  int fullswofGhostlayerWidth = ghostlayerWidth;
//
//  /** Water height after one step of the scheme.*/
//  double* h = input.h;
//  for (int x = 0; x < subdivisionFactor(0); x++) {
//        for (int y = 0; y < subdivisionFactor(1); y++) {
//            subcellIndex(0) = x;
//            subcellIndex(1) = y;
//
//            unsigned int index[3];
//            index[0] = y+ghostlayerWidth;
//            index[1] = x+ghostlayerWidth;
//            index[2] = patchid;
//            unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);
//
//            patch.getAccessor().setValueUNew(subcellIndex, 0, h[centerIndex]);
//        }
//  }
//
//  /** X Velocity after one step of the scheme.*/
//  double* u = input.u;
//  for (int x = 0; x < subdivisionFactor(0); x++) {
//        for (int y = 0; y < subdivisionFactor(1); y++) {
//            subcellIndex(0) = x;
//            subcellIndex(1) = y;
//
//            unsigned int index[3];
//            index[0] = y+ghostlayerWidth;
//            index[1] = x+ghostlayerWidth;
//            index[2] = patchid;
//            unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);
//
//            patch.getAccessor().setValueUNew(subcellIndex, 1, u[centerIndex]);
//        }
//  }
//
//  /** Y Velocity after one step of the scheme.*/
//  double* v = input.v;
//  for (int x = 0; x < subdivisionFactor(0); x++) {
//        for (int y = 0; y < subdivisionFactor(1); y++) {
//            subcellIndex(0) = x;
//            subcellIndex(1) = y;
//
//            unsigned int index[3];
//            index[0] = y+ghostlayerWidth;
//            index[1] = x+ghostlayerWidth;
//            index[2] = patchid;
//            unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);
//
//            patch.getAccessor().setValueUNew(subcellIndex, 2, v[centerIndex]);
//        }
//  }
//
//  /** Topography.*/
//  double* z = input.z;
//  for (int x = 0; x < subdivisionFactor(0); x++) {
//        for (int y = 0; y < subdivisionFactor(1); y++) {
//            subcellIndex(0) = x;
//            subcellIndex(1) = y;
//
//            unsigned int index[3];
//            index[0] = y+ghostlayerWidth;
//            index[1] = x+ghostlayerWidth;
//            index[2] = patchid;
//            unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);
//
//            patch.getAccessor().setParameterWithGhostlayer(subcellIndex, 3, z[centerIndex]);
//        }
//  }
//
//#if 1 // TODO: we probably needs this
//  /** compute Discharge. (1->nxcell) */
//  double* q1 = temp.q1;
//  for (int x = 0; x < subdivisionFactor(0); x++) {
//    for (int y = 0; y < subdivisionFactor(1); y++) {
//        subcellIndex(0) = x;
//        subcellIndex(1) = y;
//
//        unsigned int index[3];
//        index[0] = y+fullswofGhostlayerWidth;
//        index[1] = x+fullswofGhostlayerWidth;
//        index[2] = patchid;
//        unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);
//
//        patch.getAccessor().setValueUNew(subcellIndex, 4, q1[centerIndex]);
//    }
//  }
//
//  /** compute Discharge. (1->nycell)*/
//  double* q2 = temp.q2;
//  for (int x = 0; x < subdivisionFactor(0); x++) {
//    for (int y = 0; y < subdivisionFactor(1); y++) {
//        subcellIndex(0) = x;
//        subcellIndex(1) = y;
//
//        unsigned int index[3];
//        index[0] = y+fullswofGhostlayerWidth;
//        index[1] = x+fullswofGhostlayerWidth;
//        index[2] = patchid;
//        unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);
//
//        patch.getAccessor().setValueUNew(subcellIndex, 5, q2[centerIndex]);
//    }
//  }
//#endif
//
//}

peanoclaw::native::FullSWOF2D_Parameters::FullSWOF2D_Parameters(
  int ghostlayerWidth,
  int nx,
  int ny,
  double meshwidth_x,
  double meshwidth_y,
  tarch::la::Vector<DIMENSIONS,double> domainSize,
  double endTime,
  bool enableRain,
  double friction,
  peanoclaw::native::scenarios::FullSWOF2DBoundaryCondition leftBoundaryCondition,
  peanoclaw::native::scenarios::FullSWOF2DBoundaryCondition rightBoundaryCondition,
  peanoclaw::native::scenarios::FullSWOF2DBoundaryCondition bottomBoundaryCondition,
  peanoclaw::native::scenarios::FullSWOF2DBoundaryCondition topBoundaryCondition,
  int select_order,
  int select_rec
) : _endTime(endTime) {
    // seed parameters based on Input file
    //setparameters("./fullswof2d_parameters.txt");
 
//    int fullswofGhostlayerWidth = ghostlayerWidth-1;
    dx = meshwidth_x;
    dy = meshwidth_y;

    // now override the peanoclaw specific ones
    Nxcell = nx+2*(ghostlayerWidth-1); // FullSWOF2D already provides a boundary layer of width 1
    Nycell = ny+2*(ghostlayerWidth-1); // FullSWOF2D already provides a boundary layer of width 1

    //std::cout << "nxcell " << Nxcell << " nycell " << Nycell << std::endl;

    scheme_type = 1; // 1= fixed cfl => we get timestamp and maximum timestep
                     // 2=fixed dt

    // we probably have to provide dx / dy values directly and we do it see below!
    // FullSOWF2D uses L and l just to compute dx and dy
    L = domainSize[0]; //meshwidth_x*(Nxcell); // length of domain in x direction (TODO: i multiply be Nxcell because the code will divide later on...)
    l = domainSize[1]; //meshwidth_y*(Nycell); // length of domain in y direction (TODO: i multiply be Nycell because the code will divide later on...)

    //T = maximumTimestepSize; this is a bad idea: the algorithm internally works similar to peanoclaw: either dt caused by cfl or the remaining part in the time interval
    //(see line 198 of order2.cpp)
    T = 1000; // that should be enough as we only do one timestep anyway
 
    order = select_order; // order 1 or order 2

   cfl_fix = 0.5; // TODO: what is this actually good for? (working setting 0.5 with order 1, 0.8 got stuck with order 1 and 2)
//   cfl_fix = 0.3857;
   dt_fix = 0.05; // TODO: what is this actually good for
   nbtimes = 1;


   // we need some sort of continuous boundary
   // TODO: what is the proper boundary in our case?
   // can we shrink the ghostlayer with this boundary setting?
   Lbound = leftBoundaryCondition.getType();
   Rbound = rightBoundaryCondition.getType();
   Bbound = bottomBoundaryCondition.getType();
   Tbound = topBoundaryCondition.getType();
	
   flux = 2; // 1 = rusanov 2 = HLL 3 = HLL2
   rec = select_rec; // 1 = MUSCL, 2 = ENO, 3 = ENO_mod

   // really interesting effects if enabled, makes the breaking dam collapse!
   fric = tarch::la::equals(friction, 0) ? 0 : 1; // Friction law (0=NoFriction 1=Manning 2=Darcy-Weisbach)  <fric>:: 0
   inf = 0; // Infiltration model (0=No Infiltration 1=Green-Ampt)

   lim = 1;
   topo = 2; // flat topogrophy, just prevent it from loading data as we will fill the data in later on
   huv_init = 2; // initialize h,v and u to 0
   rain = enableRain ? 2 : 0; // 2: rain is generated, basically its just an auxillary array which is filled with time dependent data (we can couple this evolveToTime)
   amortENO = 0.25;
   modifENO = 0.9;
   //frotcoef is not actually used though it is existing in parameters
   friccoef = friction;
   fric_init = 2;

  //for infiltration model
  Kc_init = 2;
  Kc_coef = 1.8e-5;

  Ks_init = 2;
  Ks_coef = 0.0;

  dtheta_init = 2;
  dtheta_coef = 0.254;

  Psi_init = 2;
  Psi_coef = 0.167;

  zcrust_init = 2;
  zcrust_coef = 0.01;

  imax_init = 2;
  imax_coef = 5.7e-4;

  //  SCALAR  Kssoil;

  left_imp_discharge = leftBoundaryCondition.getImpliedDischarge(); //0.0;
  left_imp_h = leftBoundaryCondition.getImpliedHeight(); //0.1;
  right_imp_discharge = rightBoundaryCondition.getImpliedDischarge(); //0.0;
  right_imp_h = rightBoundaryCondition.getImpliedHeight(); //0.1;
  bottom_imp_discharge = bottomBoundaryCondition.getImpliedDischarge(); //0.0;
  bottom_imp_h = bottomBoundaryCondition.getImpliedHeight(); //0.1;
  top_imp_discharge = topBoundaryCondition.getImpliedDischarge(); //0.0;
  top_imp_h = topBoundaryCondition.getImpliedHeight(); //0.1;

  output_format = 0; // disable all output
}

peanoclaw::native::FullSWOF2D_Parameters::~FullSWOF2D_Parameters() {

}

double peanoclaw::native::FullSWOF2D_Parameters::get_T() const {
  return _endTime;
}

