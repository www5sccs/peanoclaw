/*
 * PyClaw.cpp
 *
 *  Created on: Feb 18, 2012
 *      Author: kristof
 */

#include "peanoclaw/Patch.h"
#include "tarch/timing/Watch.h"
#include "tarch/parallel/Node.h"

#include "peanoclaw/native/FullSWOF2D.h"

#include "choice_scheme.hpp"

tarch::logging::Log peanoclaw::native::FullSWOF2D::_log("peanoclaw::native::FullSWOF2D");

peanoclaw::native::FullSWOF2D::FullSWOF2D(
  peanoclaw::native::scenarios::SWEScenario& scenario,
  peanoclaw::interSubgridCommunication::DefaultTransfer* transfer,
  peanoclaw::interSubgridCommunication::Interpolation*  interpolation,
  peanoclaw::interSubgridCommunication::Restriction*    restriction,
  peanoclaw::interSubgridCommunication::FluxCorrection* fluxCorrection
) : Numerics(transfer, interpolation, restriction, fluxCorrection),
_totalSolverCallbackTime(0.0),
_scenario(scenario)
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

void peanoclaw::native::FullSWOF2D::solveTimestep(Patch& patch, double maximumTimestepSize, bool useDimensionalSplitting) {
  logTraceInWith2Arguments( "solveTimestep(...)", maximumTimestepSize, useDimensionalSplitting);

  assertion2(tarch::la::greater(maximumTimestepSize, 0.0), "Timestepsize == 0 should be checked outside.", patch.getTimeIntervals().getMinimalNeighborTimeConstraint());

  tarch::timing::Watch pyclawWatch("", "", false);
  pyclawWatch.startTimer();
//  double dtAndEstimatedNextDt[2];
  tarch::la::Vector<DIMENSIONS,double> meshwidth = patch.getSubcellSize();
  tarch::la::Vector<DIMENSIONS,int> subdivisionFactor = patch.getSubdivisionFactor();
  int ghostlayerWidth = patch.getGhostlayerWidth();
 
  double dt; // = std::min(dt_used, maximumTimestepSize);
  double estimatedNextTimestepSize; // = scheme->getMaxTimestep();

  // kick off the computation here -----
#if 1
  {
      FullSWOF2D_Parameters par(ghostlayerWidth, subdivisionFactor(0), subdivisionFactor(1), meshwidth(0), meshwidth(1));
      //std::cout << "parameters read (meshwidth): " << par.get_dx() << " vs " << meshwidth(0) << " and " << par.get_dy() << " vs " << meshwidth(1) << std::endl;
      //std::cout << "parameters read (cells): " << par.get_Nxcell() << " vs " << subdivisionFactor(0) << " and " << par.get_Nycell() << " vs " << subdivisionFactor(1) << std::endl;

      Choice_scheme *wrapper_scheme = new Choice_scheme(par);
      Scheme *scheme = wrapper_scheme->getInternalScheme();

      // overwrite internal values
      copyPatchToScheme(patch, scheme);
      
      // kick off computation!
      scheme->setTimestep(maximumTimestepSize);
      scheme->setMaxTimestep(maximumTimestepSize); // TODO: maximumTimstepSize is ignored and the "real" maxTimestep is computed

      do {
        scheme->resetTimings();
        scheme->resetN();

        struct timeval start;
        gettimeofday(&start, NULL);

        wrapper_scheme->calcul();

        struct timeval stop;
        gettimeofday(&stop, NULL);

//        double time = (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec) / 1000000.0;
        //std::cout << "calculation took " << time << std::endl;

        if (scheme->getVerif() == 0) {
            std::cout << "scheme retry activated!" << std::endl;
        }
      } while (scheme->getVerif() == 0); // internal error detection of FullSWOF2D

      // copy back internal values but skip ghostlayer
      copySchemeToPatch(scheme, patch);

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
      delete wrapper_scheme;
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

        copyPatchToSet(patch, strideinfo,input, temp);
        double dt_used = MekkaFlood_solver::calcul(patchid, 3, strideinfo, input, temp, constants, maximumTimestepSize);
        //std::cout << "dt_used " << dt_used << " maximumTimestepSize " << maximumTimestepSize << std::endl;
        copySetToPatch(strideinfo,input, temp, patch);

        dt = std::min(dt_used, maximumTimestepSize);
        estimatedNextTimestepSize = maximumTimestepSize;

        MekkaFlood_solver::freeInput(input);
        MekkaFlood_solver::freeTemp(temp);
  }
#endif

  pyclawWatch.stopTimer();
  _totalSolverCallbackTime += pyclawWatch.getCalendarTime();

  assertion4(
      tarch::la::greater(patch.getTimeIntervals().getTimestepSize(), 0.0)
      || tarch::la::greater(estimatedNextTimestepSize, 0.0)
      || tarch::la::equals(maximumTimestepSize, 0.0)
      || tarch::la::equals(patch.getTimeIntervals().getEstimatedNextTimestepSize(), 0.0),
      patch, maximumTimestepSize, estimatedNextTimestepSize, patch.toStringUNew());
  assertion(patch.getTimeIntervals().getTimestepSize() < std::numeric_limits<double>::infinity());

  if (tarch::la::greater(dt, 0.0)) {
    patch.getTimeIntervals().advanceInTime();
    patch.getTimeIntervals().setTimestepSize(dt);
  }
  patch.getTimeIntervals().setEstimatedNextTimestepSize(estimatedNextTimestepSize);

  logTraceOut( "solveTimestep(...)");
}

tarch::la::Vector<DIMENSIONS, double> peanoclaw::native::FullSWOF2D::getDemandedMeshWidth(Patch& patch, bool isInitializing) {
  return _scenario.computeDemandedMeshWidth(patch, isInitializing);
}

void peanoclaw::native::FullSWOF2D::addPatchToSolution(Patch& patch) {
}

void peanoclaw::native::FullSWOF2D::fillBoundaryLayer(Patch& patch, int dimension, bool setUpper) {
  logTraceInWith3Arguments("fillBoundaryLayerInPyClaw", patch, dimension, setUpper);

  logDebug("fillBoundaryLayerInPyClaw", "Setting left boundary for " << patch.getPosition() << ", dim=" << dimension << ", setUpper=" << setUpper);

   //std::cout << "------ setUpper " << setUpper << " dimension " << dimension << std::endl;
   //std::cout << patch << std::endl;
   //std::cout << "++++++" << std::endl;
   //std::cout << patch.toStringUOldWithGhostLayer() << std::endl;
   //std::cout << "||||||" << std::endl;
 
#if 0
  int ghostlayerWidth = patch.getGhostlayerWidth();
  int fullswofGhostlayerWidth = ghostlayerWidth-1;
   // implement a wall boundary
    tarch::la::Vector<DIMENSIONS, int> src_subcellIndex;
    tarch::la::Vector<DIMENSIONS, int> dest_subcellIndex;

    if (dimension == 0) { // left and right boundary
        for (int yi = -fullswofGhostlayerWidth; yi < patch.getSubdivisionFactor()(1)+fullswofGhostlayerWidth; yi++) {
            int xi = setUpper ? patch.getSubdivisionFactor()(0)+fullswofGhostlayerWidth-1 : -fullswofGhostlayerWidth;
            dest_subcellIndex(0) = xi;
            dest_subcellIndex(1) = yi;
            patch.setParameterWithGhostlayer(dest_subcellIndex, 0, 1000.0);

            // only mirror orthogonal velocities
            patch.getAccessor().setValueUOld(dest_subcellIndex, 1, -patch.getAccessor().getValueUOld(dest_subcellIndex, 1));
            //patch.getAccessor().setValueUOld(dest_subcellIndex, 4, -patch.getAccessor().getValueUOld(dest_subcellIndex, 5));
        }
    } else { // top and bottom boundary
        for (int xi = -fullswofGhostlayerWidth; xi < patch.getSubdivisionFactor()(0)+fullswofGhostlayerWidth; xi++) {
            int yi = setUpper ? patch.getSubdivisionFactor()(1)+fullswofGhostlayerWidth-1 : -fullswofGhostlayerWidth;
            dest_subcellIndex(0) = xi;
            dest_subcellIndex(1) = yi;
            patch.setParameterWithGhostlayer(dest_subcellIndex, 0, 1000.0);
            
            // only mirror orthogonal velocities
            patch.getAccessor().setValueUOld(dest_subcellIndex, 2, -patch.getAccessor().getValueUOld(dest_subcellIndex, 2));
            //patch.getAccessor().setValueUOld(dest_subcellIndex, 5, -patch.getAccessor().getValueUOld(dest_subcellIndex, 5));
        }
    }
#endif

   //std::cout << "++++++" << std::endl;
   //std::cout << patch.toStringUOldWithGhostLayer() << std::endl;
   //std::cout << "||||||" << std::endl;

      logTraceOut("fillBoundaryLayerInPyClaw");
}

void peanoclaw::native::FullSWOF2D::update(Patch& finePatch) {
    _scenario.update(finePatch);
}

void peanoclaw::native::FullSWOF2D::copyPatchToScheme(Patch& patch, Scheme* scheme) {
  tarch::la::Vector<DIMENSIONS,int> subdivisionFactor = patch.getSubdivisionFactor();
  tarch::la::Vector<DIMENSIONS,int> subcellIndex;

  // FullSWOF2D has a mixture of 0->nxcell+1 and 1->nxcell
  int ghostlayerWidth = patch.getGhostlayerWidth();
  int fullswofGhostlayerWidth = ghostlayerWidth;

  /** Water height.*/
  TAB& h = scheme->getH();
  for (int x = -ghostlayerWidth; x < subdivisionFactor(0)+ghostlayerWidth; x++) {
        for (int y = -ghostlayerWidth; y < subdivisionFactor(1)+ghostlayerWidth; y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;
            h[x+ghostlayerWidth][y+ghostlayerWidth] = patch.getAccessor().getValueUOld(subcellIndex, 0);
        }
  }
 
  /** X Velocity.*/
  TAB& u = scheme->getU();
  for (int x = -ghostlayerWidth; x < subdivisionFactor(0)+ghostlayerWidth; x++) {
        for (int y = -ghostlayerWidth; y < subdivisionFactor(1)+ghostlayerWidth; y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;
            u[x+ghostlayerWidth][y+ghostlayerWidth] = patch.getAccessor().getValueUOld(subcellIndex, 1);
        }
  }

  /** Y Velocity.*/
  TAB& v = scheme->getV();
  for (int x = -ghostlayerWidth; x < subdivisionFactor(0)+ghostlayerWidth; x++) {
        for (int y = -ghostlayerWidth; y < subdivisionFactor(1)+ghostlayerWidth; y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;
            v[x+ghostlayerWidth][y+ghostlayerWidth] = patch.getAccessor().getValueUOld(subcellIndex, 2);
        }
  }
 
  /** Topography.*/
  TAB& z = scheme->getZ();
  for (int x = -ghostlayerWidth; x < subdivisionFactor(0)+ghostlayerWidth; x++) {
        for (int y = -ghostlayerWidth; y < subdivisionFactor(1)+ghostlayerWidth; y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;
            z[x+ghostlayerWidth][y+ghostlayerWidth] = patch.getAccessor().getParameterWithGhostlayer(subcellIndex, 0);
        }
  }

#if 1 // TODO: we probably need this
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

void peanoclaw::native::FullSWOF2D::copySchemeToPatch(Scheme* scheme, Patch& patch) {
  tarch::la::Vector<DIMENSIONS,int> subdivisionFactor = patch.getSubdivisionFactor();
  tarch::la::Vector<DIMENSIONS,int> subcellIndex;

  int ghostlayerWidth = patch.getGhostlayerWidth();
  int fullswofGhostlayerWidth = ghostlayerWidth;

  /** Water height after one step of the scheme.*/
  TAB& h = scheme->getH();
  for (int x = 0; x < subdivisionFactor(0); x++) {
        for (int y = 0; y < subdivisionFactor(1); y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;
            patch.getAccessor().setValueUNew(subcellIndex, 0, h[x+ghostlayerWidth][y+ghostlayerWidth]);
        }
  }
 
  /** X Velocity after one step of the scheme.*/
  TAB& u = scheme->getU();
  for (int x = 0; x < subdivisionFactor(0); x++) {
        for (int y = 0; y < subdivisionFactor(1); y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;
            patch.getAccessor().setValueUNew(subcellIndex, 1, u[x+ghostlayerWidth][y+ghostlayerWidth]);
        }
  }

  /** Y Velocity after one step of the scheme.*/
  TAB& v = scheme->getV();
  for (int x = 0; x < subdivisionFactor(0); x++) {
        for (int y = 0; y < subdivisionFactor(1); y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;
            patch.getAccessor().setValueUNew(subcellIndex, 2, v[x+ghostlayerWidth][y+ghostlayerWidth]);
        }
  }

  /** Topography.*/
  TAB& znew = scheme->getZ();
  for (int x = 0; x < subdivisionFactor(0); x++) {
    for (int y = 0; y < subdivisionFactor(1); y++) {
        subcellIndex(0) = x;
        subcellIndex(1) = y;
        patch.getAccessor().setParameterWithGhostlayer(subcellIndex, 0, znew[x+ghostlayerWidth][y+ghostlayerWidth]);
    }
  }
 
#if 1 // TODO: we probably needs this
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
  /** Water height.*/
//  for (int x = -ghostlayerWidth; x < subdivisionFactor(0)+ghostlayerWidth; x++) {
//    for (int y = -ghostlayerWidth; y < subdivisionFactor(1)+ghostlayerWidth; y++) {
//      subcellIndex(0) = x;
//      subcellIndex(1) = y;
//      if(!tarch::la::allGreaterEquals(subcellIndex, 0) || tarch::la::oneGreaterEquals(subcellIndex, subdivisionFactor)) {
//        patch.getAccessor().setValueUOld(subcellIndex, 0, h[x+ghostlayerWidth][y+ghostlayerWidth]);
//      }
//    }
//  }
//
//  /** X Velocity.*/
//  for (int x = -ghostlayerWidth; x < subdivisionFactor(0)+ghostlayerWidth; x++) {
//        for (int y = -ghostlayerWidth; y < subdivisionFactor(1)+ghostlayerWidth; y++) {
//            subcellIndex(0) = x;
//            subcellIndex(1) = y;
//            if(!tarch::la::allGreaterEquals(subcellIndex, 0) || tarch::la::oneGreaterEquals(subcellIndex, subdivisionFactor)) {
//              patch.getAccessor().setValueUOld(subcellIndex, 1, u[x+ghostlayerWidth][y+ghostlayerWidth]);
//            }
//        }
//  }
//
//  /** Y Velocity.*/
//  for (int x = -ghostlayerWidth; x < subdivisionFactor(0)+ghostlayerWidth; x++) {
//        for (int y = -ghostlayerWidth; y < subdivisionFactor(1)+ghostlayerWidth; y++) {
//            subcellIndex(0) = x;
//            subcellIndex(1) = y;
//            if(!tarch::la::allGreaterEquals(subcellIndex, 0) || tarch::la::oneGreaterEquals(subcellIndex, subdivisionFactor)) {
//              patch.getAccessor().setValueUOld(subcellIndex, 2, v[x+ghostlayerWidth][y+ghostlayerWidth]);
//            }
//        }
//  }

}

void peanoclaw::native::FullSWOF2D::copyPatchToSet(Patch& patch, unsigned int *strideinfo, MekkaFlood_solver::InputArrays& input, MekkaFlood_solver::TempArrays& temp) {
    const int patchid = 0; // TODO: make this generic

  tarch::la::Vector<DIMENSIONS,int> subdivisionFactor = patch.getSubdivisionFactor();
  tarch::la::Vector<DIMENSIONS,int> subcellIndex;

  // FullSWOF2D has a mixture of 0->nxcell+1 and 1->nxcell
  int ghostlayerWidth = patch.getGhostlayerWidth();
  int fullswofGhostlayerWidth = ghostlayerWidth;

  /** Water height.*/
  double* h = input.h;
  for (int x = -ghostlayerWidth; x < subdivisionFactor(0)+ghostlayerWidth; x++) {
        for (int y = -ghostlayerWidth; y < subdivisionFactor(1)+ghostlayerWidth; y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;

            unsigned int index[3];
            index[0] = y+ghostlayerWidth;
            index[1] = x+ghostlayerWidth;
            index[2] = patchid;
            unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);

            h[centerIndex] = patch.getAccessor().getValueUOld(subcellIndex, 0);
        }
  }
 
  /** X Velocity.*/
  double* u = input.u;
  for (int x = -ghostlayerWidth; x < subdivisionFactor(0)+ghostlayerWidth; x++) {
        for (int y = -ghostlayerWidth; y < subdivisionFactor(1)+ghostlayerWidth; y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;
 
            unsigned int index[3];
            index[0] = y+ghostlayerWidth;
            index[1] = x+ghostlayerWidth;
            index[2] = patchid;
            unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);

            u[centerIndex] = patch.getAccessor().getValueUOld(subcellIndex, 1);
        }
  }

  /** Y Velocity.*/
  double* v = input.v;
  for (int x = -ghostlayerWidth; x < subdivisionFactor(0)+ghostlayerWidth; x++) {
        for (int y = -ghostlayerWidth; y < subdivisionFactor(1)+ghostlayerWidth; y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;

            unsigned int index[3];
            index[0] = y+ghostlayerWidth;
            index[1] = x+ghostlayerWidth;
            index[2] = patchid;
            unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);

            v[centerIndex] = patch.getAccessor().getValueUOld(subcellIndex, 2);
        }
  }
 
  /** Topography.*/
  double* z = input.z;
  for (int x = -ghostlayerWidth; x < subdivisionFactor(0)+ghostlayerWidth; x++) {
        for (int y = -ghostlayerWidth; y < subdivisionFactor(1)+ghostlayerWidth; y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;

            unsigned int index[3];
            index[0] = y+ghostlayerWidth;
            index[1] = x+ghostlayerWidth;
            index[2] = patchid;
            unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);

            z[centerIndex] = patch.getAccessor().getParameterWithGhostlayer(subcellIndex, 0);
        }
  }

#if 1 // TODO: we probably need this
  /** compute Discharge. (1->nxcell) */
  double* q1 = temp.q1;
  for (int x = -1; x < subdivisionFactor(0)+1; x++) {
    for (int y = -1; y < subdivisionFactor(1)+1; y++) {
        subcellIndex(0) = x;
        subcellIndex(1) = y;

        unsigned int index[3];
        index[0] = y+fullswofGhostlayerWidth;
        index[1] = x+fullswofGhostlayerWidth;
        index[2] = patchid;
        unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);

        // we have to initialize this because the FullSWOF2D does not compute the momentum on the ghostlayer
        double q = patch.getAccessor().getValueUOld(subcellIndex, 0) * patch.getAccessor().getValueUOld(subcellIndex, 1);
        q1[centerIndex] = q;

    }
  }

  /** compute Discharge. (1->nycell)*/
  double* q2 = temp.q2;
  for (int x = -1; x < subdivisionFactor(0)+1; x++) {
    for (int y = -1; y < subdivisionFactor(1)+1; y++) {
        subcellIndex(0) = x;
        subcellIndex(1) = y;

        unsigned int index[3];
        index[0] = y+fullswofGhostlayerWidth;
        index[1] = x+fullswofGhostlayerWidth;
        index[2] = patchid;
        unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);

        double q = patch.getAccessor().getValueUOld(subcellIndex, 0) * patch.getAccessor().getValueUOld(subcellIndex, 2);
        // we have to initialize this because the FullSWOF2D does not compute the momentum on the ghostlayer
        q2[centerIndex] = q;
    }
  }
#endif
}

void peanoclaw::native::FullSWOF2D::copySetToPatch(unsigned int *strideinfo, MekkaFlood_solver::InputArrays& input, MekkaFlood_solver::TempArrays& temp, Patch& patch) {
  const int patchid = 0; // TODO: make this generic

  tarch::la::Vector<DIMENSIONS,int> subdivisionFactor = patch.getSubdivisionFactor();
  tarch::la::Vector<DIMENSIONS,int> subcellIndex;

  int ghostlayerWidth = patch.getGhostlayerWidth();
  int fullswofGhostlayerWidth = ghostlayerWidth;

  /** Water height after one step of the scheme.*/
  double* h = input.h;
  for (int x = 0; x < subdivisionFactor(0); x++) {
        for (int y = 0; y < subdivisionFactor(1); y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;
 
            unsigned int index[3];
            index[0] = y+ghostlayerWidth;
            index[1] = x+ghostlayerWidth;
            index[2] = patchid;
            unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);

            patch.getAccessor().setValueUNew(subcellIndex, 0, h[centerIndex]);
        }
  }
 
  /** X Velocity after one step of the scheme.*/
  double* u = input.u;
  for (int x = 0; x < subdivisionFactor(0); x++) {
        for (int y = 0; y < subdivisionFactor(1); y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;

            unsigned int index[3];
            index[0] = y+ghostlayerWidth;
            index[1] = x+ghostlayerWidth;
            index[2] = patchid;
            unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);

            patch.getAccessor().setValueUNew(subcellIndex, 1, u[centerIndex]);
        }
  }

  /** Y Velocity after one step of the scheme.*/
  double* v = input.v;
  for (int x = 0; x < subdivisionFactor(0); x++) {
        for (int y = 0; y < subdivisionFactor(1); y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;

            unsigned int index[3];
            index[0] = y+ghostlayerWidth;
            index[1] = x+ghostlayerWidth;
            index[2] = patchid;
            unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);

            patch.getAccessor().setValueUNew(subcellIndex, 2, v[centerIndex]);
        }
  }

  /** Topography.*/
  double* z = input.z;
  for (int x = 0; x < subdivisionFactor(0); x++) {
        for (int y = 0; y < subdivisionFactor(1); y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;

            unsigned int index[3];
            index[0] = y+ghostlayerWidth;
            index[1] = x+ghostlayerWidth;
            index[2] = patchid;
            unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);

            patch.getAccessor().setParameterWithGhostlayer(subcellIndex, 3, z[centerIndex]);
        }
  }
 
#if 1 // TODO: we probably needs this
  /** compute Discharge. (1->nxcell) */
  double* q1 = temp.q1;
  for (int x = 0; x < subdivisionFactor(0); x++) {
    for (int y = 0; y < subdivisionFactor(1); y++) {
        subcellIndex(0) = x;
        subcellIndex(1) = y;

        unsigned int index[3];
        index[0] = y+fullswofGhostlayerWidth;
        index[1] = x+fullswofGhostlayerWidth;
        index[2] = patchid;
        unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);

        patch.getAccessor().setValueUNew(subcellIndex, 4, q1[centerIndex]);
    }
  }

  /** compute Discharge. (1->nycell)*/
  double* q2 = temp.q2;
  for (int x = 0; x < subdivisionFactor(0); x++) {
    for (int y = 0; y < subdivisionFactor(1); y++) {
        subcellIndex(0) = x;
        subcellIndex(1) = y;
 
        unsigned int index[3];
        index[0] = y+fullswofGhostlayerWidth;
        index[1] = x+fullswofGhostlayerWidth;
        index[2] = patchid;
        unsigned int centerIndex = MekkaFlood_solver::linearizeIndex(3, index, strideinfo);

        patch.getAccessor().setValueUNew(subcellIndex, 5, q2[centerIndex]);
    }
  }
#endif

}

peanoclaw::native::FullSWOF2D_Parameters::FullSWOF2D_Parameters(int ghostlayerWidth, int nx, int ny, double meshwidth_x, double meshwidth_y, int select_order, int select_rec) {
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
    L = meshwidth_x*(Nxcell); // length of domain in x direction (TODO: i multiply be Nxcell because the code will divide later on...)
    l = meshwidth_y*(Nycell); // length of domain in y direction (TODO: i multiply be Nycell because the code will divide later on...)

    //T = maximumTimestepSize; this is a bad idea: the algorithm internally works similar to peanoclaw: either dt caused by cfl or the remaining part in the time interval
    //(see line 198 of order2.cpp)
    T = 1000; // that should be enough as we only do one timestep anyway
 
    order = select_order; // order 1 or order 2

   cfl_fix = 0.5; // TODO: what is this actually good for? (working setting 0.5 with order 1, 0.8 got stuck with order 1 and 2)
   dt_fix = 0.05; // TODO: what is this actually good for
   nbtimes = 1;


   // we need some sort of continuous boundary
   // TODO: what is the proper boundary in our case?
   // can we shrink the ghostlayer with this boundary setting?
   Lbound = 2;
   Rbound = 2;
   Bbound = 2;
   Tbound = 2;
	
   flux = 2; // 1 = rusanov 2 = HLL 3 = HLL2
   rec = select_rec; // 1 = MUSCL, 2 = ENO, 3 = ENO_mod

   // really interesting effects if enabled, makes the breaking dam collapse!
   fric = 0; // Friction law (0=NoFriction 1=Manning 2=Darcy-Weisbach)  <fric>:: 0
   inf = 0; // Infiltration model (0=No Infiltration 1=Green-Ampt)

   lim = 1;
   topo = 2; // flat topogrophy, just prevent it from loading data as we will fill the data in later on
   huv_init = 2; // initialize h,v and u to 0
   rain = 2; // 2: rain is generated, basically its just an auxillary array which is filled with time dependent data (we can couple this evolveToTime)
   amortENO = 0.25;
   modifENO = 0.9;
   //frotcoef is not actually used though it is existing in parameters
   friccoef = 0;

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

  left_imp_discharge = 0.0;
  left_imp_h = 0.1;
  right_imp_discharge = 0.0;
  right_imp_h = 0.1;
  bottom_imp_discharge = 0.0;
  bottom_imp_h = 0.1;
  top_imp_discharge = 0.0;
  top_imp_h = 0.1;

  output_format = 0; // disable all output
}

peanoclaw::native::FullSWOF2D_Parameters::~FullSWOF2D_Parameters() {

}



