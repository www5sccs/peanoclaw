/*
 * PyClaw.cpp
 *
 *  Created on: Feb 18, 2012
 *      Author: kristof
 */

#include <Python.h>
//#include <numpy/arrayobject.h>
#include "peanoclaw/native/SWEKernel.h"
//#include "peanoclaw/pyclaw/PyClaw.h"
//#include "peanoclaw/pyclaw/PyClawState.h"
#include "peanoclaw/Patch.h"
#include "tarch/timing/Watch.h"
#include "tarch/parallel/Node.h"

#include "peanoclaw/native/SWE_WavePropagationBlock_patch.hh"

tarch::logging::Log peanoclaw::native::SWEKernel::_log("peanoclaw::pyclaw::SWEKernel");

peanoclaw::native::SWEKernel::SWEKernel(
  /*InitializationCallback         initializationCallback,
  BoundaryConditionCallback      boundaryConditionCallback,
  SolverCallback                 solverCallback,
  AddPatchToSolutionCallback     addPatchToSolutionCallback,*/
  peanoclaw::interSubgridCommunication::Interpolation*  interpolation,
  peanoclaw::interSubgridCommunication::Restriction*    restriction,
  peanoclaw::interSubgridCommunication::FluxCorrection* fluxCorrection
) : Numerics(interpolation, restriction, fluxCorrection),
/*_initializationCallback(initializationCallback),
_boundaryConditionCallback(boundaryConditionCallback),
_solverCallback(solverCallback),
_addPatchToSolutionCallback(addPatchToSolutionCallback),*/
_totalSolverCallbackTime(0.0)
{
  //import_array();
}

peanoclaw::native::SWEKernel::~SWEKernel()
{
}


double compute_demandedMeshWidth(peanoclaw::Patch& patch) {
    double max_gradient = 0.0;
    const tarch::la::Vector<DIMENSIONS, double> meshWidth = patch.getSubcellSize();
    
    tarch::la::Vector<DIMENSIONS, int> this_subcellIndex;
    tarch::la::Vector<DIMENSIONS, int> next_subcellIndex_x;
    tarch::la::Vector<DIMENSIONS, int> next_subcellIndex_y;
    for (int yi = 0; yi < patch.getSubdivisionFactor()(1)-1; yi++) {
        for (int xi = 0; xi < patch.getSubdivisionFactor()(0)-1; xi++) {
            this_subcellIndex(0) = xi;
            this_subcellIndex(1) = yi;
  
            next_subcellIndex_x(0) = xi+1;
            next_subcellIndex_x(1) = yi;
  
            next_subcellIndex_y(0) = xi;
            next_subcellIndex_y(1) = yi+1;
 
            double q0 =  patch.getValueUNew(this_subcellIndex, 0);
            double q0_x =  (patch.getValueUNew(next_subcellIndex_x, 0) - q0) / meshWidth(0);
            double q0_y =  (patch.getValueUNew(next_subcellIndex_y, 0) - q0) / meshWidth(1);

            max_gradient = fmax(max_gradient, q0_x);
            max_gradient = fmax(max_gradient, q0_y);
        }
    }
  
    double demandedMeshWidth = 0;
    if (max_gradient > 0.05) {
        demandedMeshWidth = 1.0/243;
        //demandedMeshWidth = 1.0/81;
    } else {
        demandedMeshWidth = 1.0/81;
    }

    return demandedMeshWidth;
}

double peanoclaw::native::SWEKernel::initializePatch(
  Patch& patch
) {
  logTraceIn( "initializePatch(...)");

#if 0
  peanoclaw::pyclaw::PyClawState state(patch);

  double demandedMeshWidth = _initializationCallback(
    state._q,
    state._qbc,
    state._aux,
    patch.getSubdivisionFactor()(0),
    patch.getSubdivisionFactor()(1),
    #ifdef Dim3
    patch.getSubdivisionFactor()(2),
    #else
      0,
    #endif
    patch.getUnknownsPerSubcell(),
    patch.getAuxiliarFieldsPerSubcell(),
    patch.getSize()(0),
    patch.getSize()(1),
    #ifdef Dim3
    patch.getSize()(2),
    #else
      0,
    #endif
    patch.getPosition()(0),
    patch.getPosition()(1),
    #ifdef Dim3
    patch.getPosition()(2)
    #else
      0
    #endif
  );

#else
    // dam coordinates
    double x0=0.5;
    double y0=0.5;
    
    // Riemann states of the dam break problem
    double radDam = 0.5;
    double hl = 2.;
    double ul = 0.;
    double vl = 0.;
    double hr = 1.;
    double ur = 0.;
    double vr = 0.;
    
    // compute from mesh data
    const tarch::la::Vector<DIMENSIONS, double> patchSize = patch.getSize();
    const tarch::la::Vector<DIMENSIONS, double> patchPosition = patch.getPosition();
    const tarch::la::Vector<DIMENSIONS, double> meshWidth = patch.getSubcellSize();

    tarch::la::Vector<DIMENSIONS, int> subcellIndex;
    for (int yi = 0; yi < patch.getSubdivisionFactor()(1); yi++) {
        for (int xi = 0; xi < patch.getSubdivisionFactor()(0); xi++) {
            subcellIndex(0) = xi;
            subcellIndex(1) = yi;
 
            double X = patchPosition(0) + xi*meshWidth(0);
            double Y = patchPosition(1) + yi*meshWidth(1);
 
            double r = sqrt((X-x0)*(X-x0) + (Y-y0)*(Y-y0));
            double q0 = hl*(r<=radDam) + hr*(r>radDam);
            double q1 = hl*ul*(r<=radDam) + hr*ur*(r>radDam);
            double q2 = hl*vl*(r<=radDam) + hr*vr*(r>radDam);
  
            patch.setValueUNew(subcellIndex, 0, q0);
            patch.setValueUNew(subcellIndex, 1, q1);
            patch.setValueUNew(subcellIndex, 2, q2);
        }
    }

    double demandedMeshWidth = compute_demandedMeshWidth(patch);
#endif

  logTraceOutWith1Argument( "initializePatch(...)", demandedMeshWidth);
  return demandedMeshWidth;
}

double peanoclaw::native::SWEKernel::solveTimestep(Patch& patch, double maximumTimestepSize, bool useDimensionalSplitting) {
  logTraceInWith2Arguments( "solveTimestep(...)", maximumTimestepSize, useDimensionalSplitting);

  assertion2(tarch::la::greater(maximumTimestepSize, 0.0), "Timestepsize == 0 should be checked outside.", patch.getMinimalNeighborTimeConstraint());

  tarch::timing::Watch pyclawWatch("", "", false);
  pyclawWatch.startTimer();
  double dtAndEstimatedNextDt[2];

  SWE_WavePropagationBlock_patch swe(patch);

  swe.computeNumericalFluxes();

  double dt = fmin(swe.getMaxTimestep(), maximumTimestepSize);
  double estimatedNextTimestepSize = swe.getMaxTimestep();

  swe.updateUnknowns(dt);

  pyclawWatch.stopTimer();
  _totalSolverCallbackTime += pyclawWatch.getCalendarTime();

  assertion4(
      tarch::la::greater(patch.getTimestepSize(), 0.0)
      || tarch::la::greater(estimatedNextTimestepSize, 0.0)
      || tarch::la::equals(maximumTimestepSize, 0.0)
      || tarch::la::equals(patch.getEstimatedNextTimestepSize(), 0.0),
      patch, maximumTimestepSize, estimatedNextTimestepSize, patch.toStringUNew());
  assertion(patch.getTimestepSize() < std::numeric_limits<double>::infinity());

  if (tarch::la::greater(dt, 0.0)) {
    patch.advanceInTime();
    patch.setTimestepSize(dt);
  }
  patch.setEstimatedNextTimestepSize(estimatedNextTimestepSize);

  logTraceOutWith1Argument( "solveTimestep(...)", requiredMeshWidth);
  return compute_demandedMeshWidth(patch);
}

void peanoclaw::native::SWEKernel::addPatchToSolution(Patch& patch) {

#if 0
    peanoclaw::pyclaw::PyClawState state(patch);

  assertion(_addPatchToSolutionCallback != 0);
  _addPatchToSolutionCallback(
    state._q,
    state._qbc,
    patch.getGhostLayerWidth(),
    patch.getSize()(0),
    patch.getSize()(1),
    #ifdef Dim3
    patch.getSize()(2),
    #else
    0,
    #endif
    patch.getPosition()(0),
    patch.getPosition()(1),
    #ifdef Dim3
    patch.getPosition()(2),
    #else
    0,
    #endif
    patch.getCurrentTime()+patch.getTimestepSize()
  );
#endif
}

void peanoclaw::native::SWEKernel::fillBoundaryLayer(Patch& patch, int dimension, bool setUpper) const {
  logTraceInWith3Arguments("fillBoundaryLayerInPyClaw", patch, dimension, setUpper);

  logDebug("fillBoundaryLayerInPyClaw", "Setting left boundary for " << patch.getPosition() << ", dim=" << dimension << ", setUpper=" << setUpper);

#if 0
  peanoclaw::pyclaw::PyClawState state(patch);

  _boundaryConditionCallback(state._q, state._qbc, dimension, setUpper ? 1 : 0);
    
#else 
   //std::cout << "------ setUpper " << setUpper << " dimension " << dimension << std::endl;
   //std::cout << patch << std::endl;
   //std::cout << "++++++" << std::endl;
   //std::cout << patch.toStringUOldWithGhostLayer() << std::endl;
   //std::cout << "||||||" << std::endl;

  // implement a wall boundary
    tarch::la::Vector<DIMENSIONS, int> src_subcellIndex;
    tarch::la::Vector<DIMENSIONS, int> dest_subcellIndex;

    if (dimension == 0) {
        for (int yi = -1; yi < patch.getSubdivisionFactor()(1)+1; yi++) {
            int xi = setUpper ? patch.getSubdivisionFactor()(0) : -1;
            src_subcellIndex(0) = xi;
            src_subcellIndex(1) = yi;
            src_subcellIndex(dimension) += setUpper ? -1 : +1; 

            dest_subcellIndex(0) = xi;
            dest_subcellIndex(1) = yi;
     
            for (int unknown=0; unknown < patch.getUnknownsPerSubcell(); unknown++) {
                double q = patch.getValueUOld(src_subcellIndex, unknown);

                if (unknown == dimension + 1) {
                    patch.setValueUOld(dest_subcellIndex, unknown, -q);
                } else {
                    patch.setValueUOld(dest_subcellIndex, unknown, q);
                }
            }
        }

    } else {
        for (int xi = -1; xi < patch.getSubdivisionFactor()(0)+1; xi++) {
            int yi = setUpper ? patch.getSubdivisionFactor()(1) : -1;
            src_subcellIndex(0) = xi;
            src_subcellIndex(1) = yi;
            src_subcellIndex(dimension) += setUpper ? -1 : +1; 

            dest_subcellIndex(0) = xi;
            dest_subcellIndex(1) = yi;
     
            for (int unknown=0; unknown < patch.getUnknownsPerSubcell(); unknown++) {
                double q = patch.getValueUOld(src_subcellIndex, unknown);

                if (unknown == dimension + 1) {
                    patch.setValueUOld(dest_subcellIndex, unknown, -q);
                } else {
                    patch.setValueUOld(dest_subcellIndex, unknown, q);
                }
            }
        }
    }

   //std::cout << "++++++" << std::endl;
   //std::cout << patch.toStringUOldWithGhostLayer() << std::endl;
   //std::cout << "||||||" << std::endl;
#endif

      logTraceOut("fillBoundaryLayerInPyClaw");
    }

