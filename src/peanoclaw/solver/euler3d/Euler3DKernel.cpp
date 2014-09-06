/*
 * Euler3DKernel.cpp
 *
 *  Created on: Jul 24, 2014
 *      Author: kristof
 */
#include "peanoclaw/solver/euler3d/Euler3DKernel.h"

#include "peanoclaw/grid/aspects/BoundaryIterator.h"
#include "peanoclaw/solver/euler3d/Cell.h"
#include "peanoclaw/grid/SubgridAccessor.h"
#include "peanoclaw/Patch.h"

#include "Uni/EulerEquations/RoeSolver"

#include <iomanip>

tarch::logging::Log peanoclaw::solver::euler3d::Euler3DKernel::_log("peanoclaw::solver::euler3d::Euler3DKernel");

peanoclaw::solver::euler3d::Euler3DKernel::Euler3DKernel(
  peanoclaw::native::scenarios::SWEScenario& scenario,
  peanoclaw::interSubgridCommunication::DefaultTransfer* transfer,
  peanoclaw::interSubgridCommunication::Interpolation*   interpolation,
  peanoclaw::interSubgridCommunication::Restriction*     restriction,
  peanoclaw::interSubgridCommunication::FluxCorrection*  fluxCorrection
) : Numerics(transfer, interpolation, restriction, fluxCorrection),
    _scenario(scenario)
{
}


void peanoclaw::solver::euler3d::Euler3DKernel::initializePatch(
  Patch& patch
) {
  logTraceIn( "initializePatch(...)");

  _scenario.initializePatch(patch);

  logTraceOutWith1Argument( "initializePatch(...)", demandedMeshWidth);
}

void peanoclaw::solver::euler3d::Euler3DKernel::update(Patch& subgrid) {
  _scenario.update(subgrid);
}

void peanoclaw::solver::euler3d::Euler3DKernel::solveTimestep(
  Patch& subgrid,
  double maximumTimestepSize,
  bool useDimensionalSplitting
) {
  #ifdef Dim3
  double dt = std::min(subgrid.getTimeIntervals().getEstimatedNextTimestepSize(), maximumTimestepSize);
  assertion(tarch::la::greaterEquals(dt, 0.0));

  //TODO unterweg debug
//  std::cout << subgrid.toStringUOldWithGhostLayer() << std::endl;

  tarch::la::Vector<DIMENSIONS,int> subdivisionFactor = subgrid.getSubdivisionFactor();
  int ghostlayerWidth = subgrid.getGhostlayerWidth();
  int numberOfCellsUNew = tarch::la::volume(subdivisionFactor);
  int numberOfCellsUOld = tarch::la::volume(subdivisionFactor + 2*ghostlayerWidth);

  peanoclaw::grid::SubgridAccessor accessor = subgrid.getAccessor();
  peanoclaw::grid::SubgridIterator<NUMBER_OF_EULER_UNKNOWNS> iteratorUNew = accessor.getSubgridIterator<NUMBER_OF_EULER_UNKNOWNS>(
    tarch::la::Vector<DIMENSIONS,int>(0),
    subdivisionFactor
  );
  peanoclaw::grid::SubgridIterator<NUMBER_OF_EULER_UNKNOWNS> iteratorUOld = accessor.getSubgridIterator<NUMBER_OF_EULER_UNKNOWNS>(
    tarch::la::Vector<DIMENSIONS,int>(-subgrid.getGhostlayerWidth()),
    subdivisionFactor +  2*subgrid.getGhostlayerWidth()
  );

  //Create copy of subgrid
  std::vector<peanoclaw::solver::euler3d::Cell> cellsUNew;
  cellsUNew.reserve(numberOfCellsUNew);
  std::vector<peanoclaw::solver::euler3d::Cell> cellsUOld;
  cellsUOld.reserve(numberOfCellsUOld);

  while(iteratorUNew.moveToNextCell()) {
    cellsUNew.push_back(peanoclaw::solver::euler3d::Cell(iteratorUNew.getUnknownsUNew()));
  }
  while(iteratorUOld.moveToNextCell()) {
    cellsUOld.push_back(peanoclaw::solver::euler3d::Cell(iteratorUOld.getUnknownsUOld()));
  }

  //Run update
  double estimatedDt = dt;
  double cfl;
  double maximalCFL = 0.9;
  int iterations = 0;
  do {
    logDebug("solveTimestep(...)", "Solving timestep with dt=" << estimatedDt);
    double maxLambda = computeTimestep(estimatedDt, subgrid, cellsUNew, cellsUOld);
    dt = estimatedDt;
    cfl = dt * maxLambda / tarch::la::min(subgrid.getSubcellSize());
    estimatedDt = estimatedDt * (maximalCFL / cfl) * 0.9;
    assertion(tarch::la::greaterEquals(dt, 0.0));
    iterations++;
    assertion(iterations < 10);
  } while(tarch::la::greater(cfl, maximalCFL));

  //Copy changes back
  iteratorUNew.restart();
  std::vector<peanoclaw::solver::euler3d::Cell>::iterator iterator = cellsUNew.begin();
  while(iteratorUNew.moveToNextCell()) {
    iteratorUNew.setUnknownsUNew(iterator->getUnknowns());
    iterator++;
  }

  subgrid.getTimeIntervals().advanceInTime();
  subgrid.getTimeIntervals().setEstimatedNextTimestepSize(estimatedDt);
  subgrid.getTimeIntervals().setTimestepSize(dt);

  #ifdef Asserts
  dfor(subcellIndex, subgrid.getSubdivisionFactor()) {
    assertion5(
      tarch::la::greater(accessor.getValueUNew(subcellIndex, 0), 0.0),
      subgrid.toStringUNew(),
      subgrid.toStringUOldWithGhostLayer(),
      accessor.getValueUNew(subcellIndex, 0),
      subcellIndex,
      cfl
    );
    assertion5(
      tarch::la::greater(accessor.getValueUNew(subcellIndex, 4), 0.0),
      subgrid.toStringUNew(),
      subgrid.toStringUOldWithGhostLayer(),
      accessor.getValueUNew(subcellIndex, 4),
      subcellIndex,
      cfl
    );
  }
  #endif

  //TODO unterweg debug
//  std::cout << "After: " << std::endl << subgrid.toStringUNew() << std::endl;
//  assertionFail("");
  #endif
}

double peanoclaw::solver::euler3d::Euler3DKernel::computeTimestep(
  double dt,
  peanoclaw::Patch& subgrid,
  std::vector<peanoclaw::solver::euler3d::Cell>& cellsUNew,
  std::vector<peanoclaw::solver::euler3d::Cell>& cellsUOld
) {
  tarch::la::Vector<DIMENSIONS,int> subdivisionFactor = subgrid.getSubdivisionFactor();
  int ghostlayerWidth = subgrid.getGhostlayerWidth();
  int xRow = subdivisionFactor[0] + 2*ghostlayerWidth;
  int xyPlane = (subdivisionFactor[0] + 2*ghostlayerWidth) * (subdivisionFactor[1] + 2*ghostlayerWidth);
  tarch::la::Vector<DIMENSIONS,double> cellSize = subgrid.getSubcellSize();

  Uni::EulerEquations::RoeSolver solver;
  solver.courantNumber(dt, cellSize[0], cellSize[1], cellSize[2]);

  //TODO unterweg debug
//  std::cout << subgrid.toStringUOldWithGhostLayer() << std::endl;

  double maxLambda = 0.0;
  double maxSoundspeed = 0.0;
  int linearUNewIndex = 0;
  int linearUOldIndex = xyPlane*ghostlayerWidth + xRow*ghostlayerWidth + ghostlayerWidth;
  for(int x = 0; x < subdivisionFactor[0]; x++) {
    for(int y = 0; y < subdivisionFactor[1]; y++) {
      for(int z = 0; z < subdivisionFactor[2]; z++) {
        tarch::la::Vector<DIMENSIONS,int> subcellIndex;
        assignList(subcellIndex) = x, y, z;

        //TODO unterweg debug
        //std::cout << "back: " << (linearUOldIndex-xyPlane) << " bottom: " << (linearUOldIndex-xRow)

//        peanoclaw::solver::euler3d::Cell& backCell = cellsUOld[linearUOldIndex-xyPlane];
//        peanoclaw::solver::euler3d::Cell& bottomCell = cellsUOld[linearUOldIndex-xRow];
//        peanoclaw::solver::euler3d::Cell& leftCell = cellsUOld[linearUOldIndex-1];
//        peanoclaw::solver::euler3d::Cell& centerCell = cellsUOld[linearUOldIndex];
//        peanoclaw::solver::euler3d::Cell& rightCell = cellsUOld[linearUOldIndex+1];
//        peanoclaw::solver::euler3d::Cell& topCell = cellsUOld[linearUOldIndex+xRow];
//        peanoclaw::solver::euler3d::Cell& frontCell = cellsUOld[linearUOldIndex+xyPlane];

        peanoclaw::solver::euler3d::Cell& leftCell = cellsUOld[linearUOldIndex-xyPlane];
        peanoclaw::solver::euler3d::Cell& bottomCell = cellsUOld[linearUOldIndex-xRow];
        peanoclaw::solver::euler3d::Cell& backCell = cellsUOld[linearUOldIndex-1];
        peanoclaw::solver::euler3d::Cell& centerCell = cellsUOld[linearUOldIndex];
        peanoclaw::solver::euler3d::Cell& frontCell = cellsUOld[linearUOldIndex+1];
        peanoclaw::solver::euler3d::Cell& topCell = cellsUOld[linearUOldIndex+xRow];
        peanoclaw::solver::euler3d::Cell& rightCell = cellsUOld[linearUOldIndex+xyPlane];

        //TODO unterweg debug
        bool plot =
//            false;
//          (x > 3 && x < 6) && (y > 3 && y < 6) && (z > 3 && z < 6);
            x == 2 && y == 0 && z == 0;
//            x < 3 && y == 0 && z == 0;
        if(plot) {
          std::cout << x << ", " << y << ", " << z << std::endl;
          std::cout << "dt=" << dt << std::endl;
          std::cout << "left: density=" << std::setprecision(3) << leftCell.density() << ", momentum=" << leftCell.velocity()(0) << "," << leftCell.velocity()(1) << "," << leftCell.velocity()(2) << ", energy=" << leftCell.energy() << std::endl;
          std::cout << "right: density=" << std::setprecision(3) << rightCell.density() << ", momentum=" << rightCell.velocity()(0) << "," << rightCell.velocity()(1) << "," << rightCell.velocity()(2) << ", energy=" << rightCell.energy() << std::endl;
          std::cout << "bottom: density=" << std::setprecision(3) << bottomCell.density() << ", momentum=" << bottomCell.velocity()(0) << "," << bottomCell.velocity()(1) << "," << bottomCell.velocity()(2) << ", energy=" << bottomCell.energy() << std::endl;
          std::cout << "top: density=" << std::setprecision(3) << topCell.density() << ", momentum=" << topCell.velocity()(0) << "," << topCell.velocity()(1) << "," << topCell.velocity()(2) << ", energy=" << topCell.energy() << std::endl;
          std::cout << "back: density=" << std::setprecision(3) << backCell.density() << ", momentum=" << backCell.velocity()(0) << "," << backCell.velocity()(1) << "," << backCell.velocity()(2) << ", energy=" << backCell.energy() << std::endl;
          std::cout << "front: density=" << std::setprecision(3) << frontCell.density() << ", momentum=" << frontCell.velocity()(0) << "," << frontCell.velocity()(1) << "," << frontCell.velocity()(2) << ", energy=" << frontCell.energy() << std::endl;
          std::cout << "center: density=" << std::setprecision(3) << centerCell.density() << ", momentum=" << centerCell.velocity()(0) << "," << centerCell.velocity()(1) << "," << centerCell.velocity()(2) << ", energy=" << centerCell.energy() << std::endl;
        }

        peanoclaw::solver::euler3d::Cell& newCell = cellsUNew[linearUNewIndex];

        double localMaxLambda;
        solver.apply(
          leftCell,
          rightCell,
          bottomCell,
          topCell,
          backCell,
          frontCell,
          centerCell,
          newCell,
          localMaxLambda
        );
        maxLambda = std::max(maxLambda, localMaxLambda);

        //TODO unterweg debug
        if(plot) {
          std::cout << "  new cell density=" << newCell.density() << ", momentum=" << newCell.velocity()(0) << "," << newCell.velocity()(1) << "," << newCell.velocity()(2) << ", energy=" << newCell.energy() << std::endl;
        }

        maxSoundspeed = std::max(maxSoundspeed, centerCell.soundSpeed());

        //TODO unterweg debug
//        std::cout << "Accessing uOld=" << linearUOldIndex << " uNew=" << linearUNewIndex << std::endl;

        linearUOldIndex++;
        linearUNewIndex++;
      }
      linearUOldIndex += 2*ghostlayerWidth;
    }
    linearUOldIndex += 2*xRow;
  }

  //TODO unterweg debug
//  std::cout << "max soundspeed=" << maxSoundspeed << std::endl;

  return maxLambda;
}

tarch::la::Vector<DIMENSIONS, double> peanoclaw::solver::euler3d::Euler3DKernel::getDemandedMeshWidth(Patch& patch, bool isInitializing) {
  return _scenario.computeDemandedMeshWidth(patch, isInitializing);
}

void peanoclaw::solver::euler3d::Euler3DKernel::addPatchToSolution(Patch& patch) {
}

void peanoclaw::solver::euler3d::Euler3DKernel::fillBoundaryLayer(Patch& subgrid, int dimension, bool setUpper) {

  peanoclaw::grid::SubgridAccessor accessor = subgrid.getAccessor();

  //Fill default boundary condition
  ExtrapolateBoundaryCondition extrapolateBoundaryCondition;
  peanoclaw::grid::aspects::BoundaryIterator<ExtrapolateBoundaryCondition> defaultBoundaryIterator(extrapolateBoundaryCondition);
  defaultBoundaryIterator.iterate(subgrid, accessor, dimension, setUpper);

  //Fill scenario boundary condition
  peanoclaw::grid::aspects::BoundaryIterator<peanoclaw::native::scenarios::SWEScenario> scenarioBoundaryIterator(_scenario);
  scenarioBoundaryIterator.iterate(subgrid, accessor, dimension, setUpper);
}


void peanoclaw::solver::euler3d::ExtrapolateBoundaryCondition::setBoundaryCondition(
    peanoclaw::Patch& subgrid,
    peanoclaw::grid::SubgridAccessor& accessor,
    int dimension,
    bool setUpper,
    tarch::la::Vector<DIMENSIONS,int> sourceSubcellIndex,
    tarch::la::Vector<DIMENSIONS,int> destinationSubcellIndex
) {
   //Copy
   for(int unknown = 0; unknown < subgrid.getUnknownsPerSubcell(); unknown++) {
     accessor.setValueUOld(destinationSubcellIndex, unknown, accessor.getValueUOld(sourceSubcellIndex, unknown));
   }
}



