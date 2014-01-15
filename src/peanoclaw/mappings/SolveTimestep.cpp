#include "peanoclaw/mappings/SolveTimestep.h"

#include "peanoclaw/Heap.h"
#include "peanoclaw/ParallelSubgrid.h"
#include "peanoclaw/Patch.h"
#include "peanoclaw/interSubgridCommunication/Extrapolation.h"

#include "peano/grid/aspects/VertexStateAnalysis.h"

#include "tarch/parallel/Node.h"

/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::SolveTimestep::touchVertexLastTimeSpecification() {
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::AvoidCoarseGridRaces,false);
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::SolveTimestep::touchVertexFirstTimeSpecification() { 
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::AvoidCoarseGridRaces,false);
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::SolveTimestep::enterCellSpecification() {
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::AvoidCoarseGridRaces,false);
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::SolveTimestep::leaveCellSpecification() {
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::AvoidCoarseGridRaces,false);
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::SolveTimestep::ascendSpecification() {
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::AvoidCoarseGridRaces,false);
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::SolveTimestep::descendSpecification() {
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::AvoidCoarseGridRaces,false);
}


tarch::logging::Log                peanoclaw::mappings::SolveTimestep::_log( "peanoclaw::mappings::SolveTimestep" );

void peanoclaw::mappings::SolveTimestep::fillBoundaryLayers(
  peanoclaw::Patch& patch,
  peanoclaw::Vertex * const                fineGridVertices,
  const peano::grid::VertexEnumerator&     fineGridVerticesEnumerator
) {
  // Set boundary conditions (For hanging nodes the isBoundary() is not valid. Therefore, we simulate it by checking for the domainoffset and -size.)
  if((fineGridVertices[fineGridVerticesEnumerator(0)].isBoundary()
          || !tarch::la::allGreater(fineGridVerticesEnumerator.getVertexPosition(0), _domainOffset) || !tarch::la::allGreater(_domainOffset+_domainSize, fineGridVerticesEnumerator.getVertexPosition(0)))
          && fineGridVertices[fineGridVerticesEnumerator(0)].getAdjacentCellDescriptionIndex(1) == -1) {
    _numerics->fillBoundaryLayer(patch, 0, false);
  }
  if((fineGridVertices[fineGridVerticesEnumerator(2)].isBoundary()
      || !tarch::la::allGreater(fineGridVerticesEnumerator.getVertexPosition(2), _domainOffset) || !tarch::la::allGreater(_domainOffset+_domainSize, fineGridVerticesEnumerator.getVertexPosition(2)))
      && fineGridVertices[fineGridVerticesEnumerator(2)].getAdjacentCellDescriptionIndex(0) == -1) {
    _numerics->fillBoundaryLayer(patch, 1, true);
  }
  #ifdef Dim3
  if((fineGridVertices[fineGridVerticesEnumerator(0)].isBoundary()
      || !tarch::la::allGreater(fineGridVerticesEnumerator.getVertexPosition(0), _domainOffset) || !tarch::la::allGreater(_domainOffset+_domainSize, fineGridVerticesEnumerator.getVertexPosition(0)))
      && fineGridVertices[fineGridVerticesEnumerator(0)].getAdjacentCellDescriptionIndex(4) == -1) {
    _numerics->fillBoundaryLayer(patch, 2, false);
  }
  #endif
  if((fineGridVertices[fineGridVerticesEnumerator(1)].isBoundary()
      || !tarch::la::allGreater(fineGridVerticesEnumerator.getVertexPosition(1), _domainOffset) || !tarch::la::allGreater(_domainOffset+_domainSize, fineGridVerticesEnumerator.getVertexPosition(1)))
      && fineGridVertices[fineGridVerticesEnumerator(1)].getAdjacentCellDescriptionIndex(0) == -1) {
    _numerics->fillBoundaryLayer(patch, 0, true);
  }
  if((fineGridVertices[fineGridVerticesEnumerator(0)].isBoundary()
      || !tarch::la::allGreater(fineGridVerticesEnumerator.getVertexPosition(0), _domainOffset) || !tarch::la::allGreater(_domainOffset+_domainSize, fineGridVerticesEnumerator.getVertexPosition(0)))
      && fineGridVertices[fineGridVerticesEnumerator(0)].getAdjacentCellDescriptionIndex(2) == -1) {
    _numerics->fillBoundaryLayer(patch, 1, false);
  }
  #ifdef Dim3
  if((fineGridVertices[fineGridVerticesEnumerator(4)].isBoundary()
      || !tarch::la::allGreater(fineGridVerticesEnumerator.getVertexPosition(4), _domainOffset) || !tarch::la::allGreater(_domainOffset+_domainSize, fineGridVerticesEnumerator.getVertexPosition(4)))
      && fineGridVertices[fineGridVerticesEnumerator(4)].getAdjacentCellDescriptionIndex(0) == -1) {
    _numerics->fillBoundaryLayer(patch, 2, true);
  }
  #endif
}


peanoclaw::mappings::SolveTimestep::SolveTimestep() {
  logTraceIn( "SolveTimestep()" );
  // @todo Insert your code here
  logTraceOut( "SolveTimestep()" );
}


peanoclaw::mappings::SolveTimestep::~SolveTimestep() {
  logTraceIn( "~SolveTimestep()" );
  // @todo Insert your code here
  logTraceOut( "~SolveTimestep()" );
}


#if defined(SharedMemoryParallelisation)
peanoclaw::mappings::SolveTimestep::SolveTimestep(const SolveTimestep&  masterThread)
: _numerics(masterThread._numerics),
  _globalTimestepEndTime(masterThread._globalTimestepEndTime),
  _domainOffset(masterThread._domainOffset),
  _domainSize(masterThread._domainSize),
  _initialMaximalSubgridSize(masterThread._initialMaximalSubgridSize),
  _probeList(masterThread._probeList),
  _useDimensionalSplittingOptimization(masterThread._useDimensionalSplittingOptimization),
  _subgridStatistics(masterThread._subgridStatistics)
{
  logTraceIn( "SolveTimestep(SolveTimestep)" );
  // @todo Insert your code here
  logTraceOut( "SolveTimestep(SolveTimestep)" );
}


void peanoclaw::mappings::SolveTimestep::mergeWithWorkerThread(const SolveTimestep& workerThread) {
  logTraceIn( "mergeWithWorkerThread(SolveTimestep)" );

  _subgridStatistics.merge(workerThread._subgridStatistics);

  logTraceOut( "mergeWithWorkerThread(SolveTimestep)" );
}
#endif


void peanoclaw::mappings::SolveTimestep::createHangingVertex(
      peanoclaw::Vertex&     fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                fineGridH,
      peanoclaw::Vertex * const   coarseGridVertices,
      const peano::grid::VertexEnumerator&      coarseGridVerticesEnumerator,
      peanoclaw::Cell&       coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                   fineGridPositionOfVertex
) {
  logTraceInWith6Arguments( "createHangingVertex(...)", fineGridVertex, fineGridX, fineGridH, coarseGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfVertex );
  // @todo Insert your code here
  logTraceOutWith1Argument( "createHangingVertex(...)", fineGridVertex );
}


void peanoclaw::mappings::SolveTimestep::destroyHangingVertex(
      const peanoclaw::Vertex&   fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridH,
      peanoclaw::Vertex * const  coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&           coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfVertex
) {
  logTraceInWith6Arguments( "destroyHangingVertex(...)", fineGridVertex, fineGridX, fineGridH, coarseGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfVertex );
  // @todo Insert your code here
  logTraceOutWith1Argument( "destroyHangingVertex(...)", fineGridVertex );
}


void peanoclaw::mappings::SolveTimestep::createInnerVertex(
      peanoclaw::Vertex&               fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridH,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfVertex
) {
  logTraceInWith6Arguments( "createInnerVertex(...)", fineGridVertex, fineGridX, fineGridH, coarseGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfVertex );
  // @todo Insert your code here
  logTraceOutWith1Argument( "createInnerVertex(...)", fineGridVertex );
}


void peanoclaw::mappings::SolveTimestep::createBoundaryVertex(
      peanoclaw::Vertex&               fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridH,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfVertex
) {
  logTraceInWith6Arguments( "createBoundaryVertex(...)", fineGridVertex, fineGridX, fineGridH, coarseGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfVertex );
  // @todo Insert your code here
  logTraceOutWith1Argument( "createBoundaryVertex(...)", fineGridVertex );
}


void peanoclaw::mappings::SolveTimestep::destroyVertex(
      const peanoclaw::Vertex&   fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridH,
      peanoclaw::Vertex * const  coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&           coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfVertex
) {
  logTraceInWith6Arguments( "destroyVertex(...)", fineGridVertex, fineGridX, fineGridH, coarseGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfVertex );
  // @todo Insert your code here
  logTraceOutWith1Argument( "destroyVertex(...)", fineGridVertex );
}


void peanoclaw::mappings::SolveTimestep::createCell(
      peanoclaw::Cell&                 fineGridCell,
      peanoclaw::Vertex * const        fineGridVertices,
      const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell
) {
  logTraceInWith4Arguments( "createCell(...)", fineGridCell, fineGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfCell );

//  if(static_cast<int>(_levelStatistics.size()) < fineGridVerticesEnumerator.getLevel()) {
//    _levelStatistics.push_back(peanoclaw::statistics::LevelInformation());
//  }
  //peanoclaw::statistics::LevelInformation& levelInformation = _levelStatistics.at(fineGridVerticesEnumerator.getLevel()-1);
  //levelInformation._createdPatches++;

  logTraceOutWith1Argument( "createCell(...)", fineGridCell );
}


void peanoclaw::mappings::SolveTimestep::destroyCell(
      const peanoclaw::Cell&           fineGridCell,
      peanoclaw::Vertex * const        fineGridVertices,
      const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell
) {
  logTraceInWith4Arguments( "destroyCell(...)", fineGridCell, fineGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfCell );
 
//  if(static_cast<int>(_levelStatistics.size()) < fineGridVerticesEnumerator.getLevel()) {
//    _levelStatistics.push_back(peanoclaw::statistics::LevelInformation());
//  }
  //peanoclaw::statistics::LevelInformation& levelInformation = _levelStatistics.at(fineGridVerticesEnumerator.getLevel()-1);
  //levelInformation._destroyedPatches++;

  _subgridStatistics.destroyedSubgrid(fineGridCell.getCellDescriptionIndex());

  logTraceOutWith1Argument( "destroyCell(...)", fineGridCell );
}

#ifdef Parallel
void peanoclaw::mappings::SolveTimestep::mergeWithNeighbour(
  peanoclaw::Vertex&  vertex,
  const peanoclaw::Vertex&  neighbour,
  int                                           fromRank,
  const tarch::la::Vector<DIMENSIONS,double>&   fineGridX,
  const tarch::la::Vector<DIMENSIONS,double>&   fineGridH,
  int                                           level
) {
  logTraceInWith6Arguments( "mergeWithNeighbour(...)", vertex, neighbour, fromRank, fineGridX, fineGridH, level );
  // @todo Insert your code here
  logTraceOut( "mergeWithNeighbour(...)" );
}

void peanoclaw::mappings::SolveTimestep::prepareSendToNeighbour(
  peanoclaw::Vertex&  vertex,
  int                                           toRank,
  const tarch::la::Vector<DIMENSIONS,double>&   x,
  const tarch::la::Vector<DIMENSIONS,double>&   h,
  int                                           level
) {
  logTraceInWith3Arguments( "prepareSendToNeighbour(...)", vertex, toRank, level );
  // @todo Insert your code here
  logTraceOut( "prepareSendToNeighbour(...)" );
}

void peanoclaw::mappings::SolveTimestep::prepareCopyToRemoteNode(
  peanoclaw::Vertex&  localVertex,
  int                                           toRank,
  const tarch::la::Vector<DIMENSIONS,double>&   x,
  const tarch::la::Vector<DIMENSIONS,double>&   h,
  int                                           level
) {
  logTraceInWith2Arguments( "prepareCopyToRemoteNode(...)", localVertex, toRank );
  // @todo Insert your code here
  logTraceOut( "prepareCopyToRemoteNode(...)" );
}

void peanoclaw::mappings::SolveTimestep::prepareCopyToRemoteNode(
  peanoclaw::Cell&  localCell,
  int  toRank,
  const tarch::la::Vector<DIMENSIONS,double>&  cellCentre,
  const tarch::la::Vector<DIMENSIONS,double>&  cellSize,
  int                                       level
) {
  logTraceInWith5Arguments( "prepareCopyToRemoteNode(...)", localCell, toRank, cellCentre, cellSize, level );
  // @todo Insert your code here
  logTraceOut( "prepareCopyToRemoteNode(...)" );
}

void peanoclaw::mappings::SolveTimestep::mergeWithRemoteDataDueToForkOrJoin(
  peanoclaw::Vertex&  localVertex,
  const peanoclaw::Vertex&  masterOrWorkerVertex,
  int                                       fromRank,
  const tarch::la::Vector<DIMENSIONS,double>&  x,
  const tarch::la::Vector<DIMENSIONS,double>&  h,
  int                                       level
) {
  logTraceInWith6Arguments( "mergeWithRemoteDataDueToForkOrJoin(...)", localVertex, masterOrWorkerVertex, fromRank, x, h, level );
  // @todo Insert your code here
  logTraceOut( "mergeWithRemoteDataDueToForkOrJoin(...)" );
}

void peanoclaw::mappings::SolveTimestep::mergeWithRemoteDataDueToForkOrJoin(
  peanoclaw::Cell&  localCell,
  const peanoclaw::Cell&  masterOrWorkerCell,
  int                                       fromRank,
  const tarch::la::Vector<DIMENSIONS,double>&  x,
  const tarch::la::Vector<DIMENSIONS,double>&  h,
  int                                       level
) {
  logTraceInWith3Arguments( "mergeWithRemoteDataDueToForkOrJoin(...)", localCell, masterOrWorkerCell, fromRank );
  // @todo Insert your code here
  logTraceOut( "mergeWithRemoteDataDueToForkOrJoin(...)" );
}

bool peanoclaw::mappings::SolveTimestep::prepareSendToWorker(
  peanoclaw::Cell&                 fineGridCell,
  peanoclaw::Vertex * const        fineGridVertices,
  const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
  peanoclaw::Vertex * const        coarseGridVertices,
  const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
  peanoclaw::Cell&                 coarseGridCell,
  const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell,
  int                                                                  worker
) {
  logTraceIn( "prepareSendToWorker(...)" );

  if(_estimatedRemainingIterationsUntilGlobalTimestep.find(worker) == _estimatedRemainingIterationsUntilGlobalTimestep.end()) {
    _estimatedRemainingIterationsUntilGlobalTimestep[worker] = 1;
  }

  logTraceOut( "prepareSendToWorker(...)" );

  //TODO unterweg debug
//  std::cout << "Estimated iterations: " << _estimatedRemainingIterationsUntilGlobalTimestep[worker] << std::endl;

  if(_estimatedRemainingIterationsUntilGlobalTimestep[worker] == 0) {
    return false;
  } else {
    assertion(_estimatedRemainingIterationsUntilGlobalTimestep[worker] > 0);
    _estimatedRemainingIterationsUntilGlobalTimestep[worker]--;

    if(_estimatedRemainingIterationsUntilGlobalTimestep[worker] > 0) {
      _subgridStatistics.restrictionFromWorkerSkipped();
    }

    //TODO unterweg debug
//    if(_estimatedRemainingIterationsUntilGlobalTimestep[worker] == 0) {
//      std::cout << "Reducing" << std::endl;
//    } else {
//      std::cout << "Avoiding 1" << std::endl;
//    }

    return (_estimatedRemainingIterationsUntilGlobalTimestep[worker] == 0);
  }
//  return true;
}

void peanoclaw::mappings::SolveTimestep::prepareSendToMaster(
  peanoclaw::Cell&                       localCell,
  peanoclaw::Vertex *                    vertices,
  const peano::grid::VertexEnumerator&       verticesEnumerator,
  const peanoclaw::Vertex * const        coarseGridVertices,
  const peano::grid::VertexEnumerator&       coarseGridVerticesEnumerator,
  const peanoclaw::Cell&                 coarseGridCell,
  const tarch::la::Vector<DIMENSIONS,int>&   fineGridPositionOfCell
) {
  logTraceInWith2Arguments( "prepareSendToMaster(...)", localCell, verticesEnumerator.toString() );


  //TODO unterweg debug
//  std::cout << "Estimated on worker: " << _subgridStatistics.getEstimatedIterationsUntilGlobalTimestep() << std::endl;

  _subgridStatistics.sendToMaster(tarch::parallel::NodePool::getInstance().getMasterRank());

  LevelStatisticsHeap::getInstance().finishedToSendSynchronousData();

  logTraceOut( "prepareSendToMaster(...)" );
}


void peanoclaw::mappings::SolveTimestep::mergeWithMaster(
  const peanoclaw::Cell&           workerGridCell,
  peanoclaw::Vertex * const        workerGridVertices,
 const peano::grid::VertexEnumerator& workerEnumerator,
  peanoclaw::Cell&                 fineGridCell,
  peanoclaw::Vertex * const        fineGridVertices,
  const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
  peanoclaw::Vertex * const        coarseGridVertices,
  const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
  peanoclaw::Cell&                 coarseGridCell,
  const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell,
  int                                                                  worker,
  const peanoclaw::State&          workerState,
  peanoclaw::State&                masterState
) {
  logTraceIn( "mergeWithMaster(...)" );

  //TODO unterweg debug
//  std::cout << "Merging on Master" << std::endl;

  peanoclaw::statistics::SubgridStatistics workerSubgridStatistics(worker);
  _subgridStatistics.merge(workerSubgridStatistics);

  //Reduction of reduction ;-)
  assertion1(_estimatedRemainingIterationsUntilGlobalTimestep.find(worker) != _estimatedRemainingIterationsUntilGlobalTimestep.end(), worker);
  _estimatedRemainingIterationsUntilGlobalTimestep[worker] = std::max(1, workerSubgridStatistics.getEstimatedIterationsUntilGlobalTimestep() / 2);

  //TODO unterweg debug
//  std::cout << "Estimated iterations on worker " << worker << ": " << workerSubgridStatistics.getEstimatedIterationsUntilGlobalTimestep() << std::endl;

  if(workerState.isJoiningWithMaster()) {
    _estimatedRemainingIterationsUntilGlobalTimestep.erase(worker);
  }

  logTraceOut( "mergeWithMaster(...)" );
}


void peanoclaw::mappings::SolveTimestep::receiveDataFromMaster(
  peanoclaw::Cell&                        receivedCell,
  peanoclaw::Vertex *                     receivedVertices,
  const peano::grid::VertexEnumerator&        receivedVerticesEnumerator,
  peanoclaw::Vertex * const               receivedCoarseGridVertices,
  const peano::grid::VertexEnumerator&        receivedCoarseGridVerticesEnumerator,
  peanoclaw::Cell&                        receivedCoarseGridCell,
  peanoclaw::Vertex * const               workersCoarseGridVertices,
  const peano::grid::VertexEnumerator&        workersCoarseGridVerticesEnumerator,
  peanoclaw::Cell&                        workersCoarseGridCell,
  const tarch::la::Vector<DIMENSIONS,int>&    fineGridPositionOfCell
) {
  logTraceInWith2Arguments( "receiveDataFromMaster(...)", receivedCell.toString(), receivedVerticesEnumerator.toString() );
  // @todo Insert your code here
  logTraceOut( "receiveDataFromMaster(...)" );
}


void peanoclaw::mappings::SolveTimestep::mergeWithWorker(
  peanoclaw::Cell&           localCell, 
  const peanoclaw::Cell&     receivedMasterCell,
  const tarch::la::Vector<DIMENSIONS,double>&  cellCentre,
  const tarch::la::Vector<DIMENSIONS,double>&  cellSize,
  int                                          level
) {
  logTraceInWith2Arguments( "mergeWithWorker(...)", localCell.toString(), receivedMasterCell.toString() );
  // @todo Insert your code here
  logTraceOutWith1Argument( "mergeWithWorker(...)", localCell.toString() );
}


void peanoclaw::mappings::SolveTimestep::mergeWithWorker(
  peanoclaw::Vertex&        localVertex,
  const peanoclaw::Vertex&  receivedMasterVertex,
  const tarch::la::Vector<DIMENSIONS,double>&   x,
  const tarch::la::Vector<DIMENSIONS,double>&   h,
  int                                           level
) {
  logTraceInWith2Arguments( "mergeWithWorker(...)", localVertex.toString(), receivedMasterVertex.toString() );
  // @todo Insert your code here
  logTraceOutWith1Argument( "mergeWithWorker(...)", localVertex.toString() );
}
#endif

void peanoclaw::mappings::SolveTimestep::touchVertexFirstTime(
      peanoclaw::Vertex&               fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridH,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfVertex
) {
  logTraceInWith6Arguments( "touchVertexFirstTime(...)", fineGridVertex, fineGridX, fineGridH, coarseGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfVertex );

  // Application driven refinement control
  if(
      fineGridVertex.shouldRefine()
      && fineGridVertex.getRefinementControl() == peanoclaw::Vertex::Records::Unrefined
    ) {
    //TODO unterweg debug
//    logInfo("", "Refining vertex at " << fineGridX << " on level " << (coarseGridVerticesEnumerator.getLevel()+1)
//      #ifdef Parallel
//      << " on rank " << tarch::parallel::Node::getInstance().getRank()
//      #endif
//    );
    fineGridVertex.refine();
  } else if (
      fineGridVertex.shouldErase()
      && fineGridVertex.getRefinementControl() == peanoclaw::Vertex::Records::Refined
//      && fineGridVertex.getCurrentAdjacentCellsHeight() == 1
    ) {
    //TODO unterweg debug
//    logInfo("", "Erasing vertex at " << fineGridX << " on level " << (coarseGridVerticesEnumerator.getLevel()+1)
//      #ifdef Parallel
//      << " on rank " << tarch::parallel::Node::getInstance().getRank()
//      #endif
//    );
    fineGridVertex.erase();
  }
  fineGridVertex.setShouldRefine(false);
  fineGridVertex.resetSubcellsEraseVeto();

  logTraceOutWith1Argument( "touchVertexFirstTime(...)", fineGridVertex );
}


void peanoclaw::mappings::SolveTimestep::touchVertexLastTime(
      peanoclaw::Vertex&         fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridH,
      peanoclaw::Vertex * const  coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&           coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfVertex
) {
  logTraceInWith6Arguments( "touchVertexLastTime(...)", fineGridVertex, fineGridX, fineGridH, coarseGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfVertex );
 
  logTraceOutWith1Argument( "touchVertexLastTime(...)", fineGridVertex );
}


void peanoclaw::mappings::SolveTimestep::enterCell(
      peanoclaw::Cell&                 fineGridCell,
      peanoclaw::Vertex * const        fineGridVertices,
      const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell
) {
  logTraceInWith4Arguments( "enterCell(...)", fineGridCell, fineGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfCell );

  if(fineGridCell.isInside()) {
    //Create patch
    Patch patch (
      fineGridCell
    );

    //Solve timestep for this patch
    if(fineGridCell.isLeaf()) {

      #ifdef Asserts
      CellDescription& cellDescription = CellDescriptionHeap::getInstance().getData(fineGridCell.getCellDescriptionIndex()).at(0);
      double startTime = cellDescription.getTime();
      double endTime = cellDescription.getTime() + cellDescription.getTimestepSize();
      assertionEquals1(patch.getTimeIntervals().getCurrentTime(), startTime, patch.toString());
      assertionEquals1(patch.getTimeIntervals().getCurrentTime() + patch.getTimeIntervals().getTimestepSize(), endTime, patch.toString());
      assertion(patch.isLeaf() || patch.isVirtual());
      #endif

      #ifdef Parallel
      ParallelSubgrid parallelSubgrid(fineGridCell.getCellDescriptionIndex());
      parallelSubgrid.markCurrentStateAsSent(true);
      #endif

      //Perform timestep
      double maximumTimestepDueToGlobalTimestep = _globalTimestepEndTime - (patch.getTimeIntervals().getCurrentTime() + patch.getTimeIntervals().getTimestepSize());
      if(patch.getTimeIntervals().isAllowedToAdvanceInTime(
        maximumTimestepDueToGlobalTimestep,
        fineGridVertices,
        fineGridVerticesEnumerator,
        coarseGridVertices,
        coarseGridVerticesEnumerator
      )) {
        // Copy uNew to uOld
        patch.copyUNewToUOld();

        //Extrapolate ghostlayer if necessary
        if(_useDimensionalSplittingOptimization) {
          peanoclaw::interSubgridCommunication::Extrapolation extrapolation(patch);
          extrapolation.extrapolateGhostlayer();
        }

        // Filling boundary layers for the given patch...
        fillBoundaryLayers(
          patch,
          fineGridVertices,
          fineGridVerticesEnumerator
        );

        // Do one timestep...
        double requiredMeshWidth = _numerics->solveTimestep(
                                                patch,
                                                maximumTimestepDueToGlobalTimestep,
                                                _useDimensionalSplittingOptimization
                                              );
        patch.setDemandedMeshWidth(requiredMeshWidth);

        #ifdef Parallel
        ParallelSubgrid parallelSubgrid(fineGridCell.getCellDescriptionIndex());
        parallelSubgrid.markCurrentStateAsSent(false);
        #endif

        // Coarse grid correction
        for(int i = 0; i < TWO_POWER_D; i++) {
          if(fineGridVertices[fineGridVerticesEnumerator(i)].isHangingNode()) {
            fineGridVertices[fineGridVerticesEnumerator(i)].applyFluxCorrection(*_numerics);
          }
        }

        //Statistics
        assertion1(tarch::la::greater(patch.getTimeIntervals().getTimestepSize(), 0.0), patch);
        assertion1(patch.getTimeIntervals().getTimestepSize() != std::numeric_limits<double>::infinity(), patch);
        _subgridStatistics.processSubgridAfterUpdate(patch, coarseGridCell.getCellDescriptionIndex());

        //Probes
        for(std::vector<peanoclaw::statistics::Probe>::iterator i = _probeList.begin();
            i != _probeList.end();
            i++) {
          i->plotDataIfContainedInPatch(patch);
        }

        logDebug("enterCell(...)", "New time interval of patch " << fineGridVerticesEnumerator.getCellCenter() << " on level " << fineGridVerticesEnumerator.getLevel() << " is [" << patch.getTimeIntervals().getCurrentTime() << ", " << (patch.getTimeIntervals().getCurrentTime() + patch.getTimeIntervals().getTimestepSize()) << "]");
      } else {
        logDebug("enterCell(...)", "Unchanged time interval of patch " << fineGridVerticesEnumerator.getCellCenter() << " on level " << fineGridVerticesEnumerator.getLevel() << " is [" << patch.getTimeIntervals().getCurrentTime() << ", " << (patch.getTimeIntervals().getCurrentTime() + patch.getTimeIntervals().getTimestepSize()) << "]");
        patch.reduceGridIterationsToBeSkipped();

//        //TODO unterweg debug
//        logInfo("enterCell", "Processing subgrid(1): " << patch
//            << ", coarsening: " << peano::grid::aspects::VertexStateAnalysis::doesOneVertexCarryRefinementFlag
//            (
//              coarseGridVertices,
//              coarseGridVerticesEnumerator,
//              peanoclaw::records::Vertex::Erasing
//            )
//        << ", global: " << tarch::la::greaterEquals(patch.getTimeIntervals().getCurrentTime() + patch.getTimeIntervals().getTimestepSize(), _globalTimestepEndTime));

        //Statistics
        _subgridStatistics.processSubgrid(
          patch,
          coarseGridCell.getCellDescriptionIndex()
        );
        _subgridStatistics.updateMinimalSubgridBlockReason(
          patch,
          coarseGridVertices,
          coarseGridVerticesEnumerator,
          _globalTimestepEndTime
        );
      }

      assertion2(!tarch::la::smaller(patch.getTimeIntervals().getCurrentTime(), startTime), patch, startTime);
      assertion2(!tarch::la::smaller(patch.getTimeIntervals().getCurrentTime() + patch.getTimeIntervals().getTimestepSize(), endTime), patch.getTimeIntervals().getCurrentTime() + patch.getTimeIntervals().getTimestepSize(), endTime);

      #ifdef Asserts
      if(patch.containsNaN()) {
        logError("", "Invalid solution"
            << " in patch " << patch.toString()
            << std::endl << patch.toStringUNew() << std::endl << patch.toStringUOldWithGhostLayer());
        if(coarseGridCell.getCellDescriptionIndex() != -2) {
          Patch coarsePatch(CellDescriptionHeap::getInstance().getData(coarseGridCell.getCellDescriptionIndex()).at(0));
          logError("", "Coarse Patch:" << std::endl << coarsePatch.toString() << std::endl << coarsePatch.toStringUNew())
        }

        for(int i = 0; i < TWO_POWER_D; i++) {
          logError("", "Vertex" << i <<": " << fineGridVertices[fineGridVerticesEnumerator(i)].toString());
          for(int j = 0; j < TWO_POWER_D; j++) {
            if(fineGridVertices[fineGridVerticesEnumerator(i)].getAdjacentCellDescriptionIndex(j) != -1) {
              CellDescription& cellDescription = CellDescriptionHeap::getInstance().getData(fineGridVertices[fineGridVerticesEnumerator(i)].getAdjacentCellDescriptionIndex(j)).at(0);
              Patch neighborPatch(cellDescription);
              logError("", i << " " << j << std::endl
                  << neighborPatch.toString() << std::endl
                  << neighborPatch.toStringUNew() << std::endl
                  << neighborPatch.toStringUOldWithGhostLayer());
            } else {
              logError("", i << " " << j << ": Invalid patch");
            }
          }
          //std::cout << std::endl;
        }
        //std::cout << std::endl;
        assertion(false);
        throw "";
      }
      #endif

      logTraceOutWith2Arguments( "enterCell(...)", cellDescription.getTimeIntervals().getTimestepSize(), cellDescription.getTime() + cellDescription.getTimeIntervals().getTimestepSize() );
    } else {
      logTraceOut( "enterCell(...)" );
    }

    patch.increaseAgeByOneGridIteration();

    //TODO unterweg debug
    assertion1(!tarch::la::smaller(patch.getTimeIntervals().getTimestepSize(), 0.0) || !patch.isLeaf(), patch);
  }
}


void peanoclaw::mappings::SolveTimestep::leaveCell(
      peanoclaw::Cell&           fineGridCell,
      peanoclaw::Vertex * const  fineGridVertices,
      const peano::grid::VertexEnumerator&          fineGridVerticesEnumerator,
      peanoclaw::Vertex * const  coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&           coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfCell
) {
  logTraceInWith4Arguments( "leaveCell(...)", fineGridCell, fineGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfCell );
  Patch patch(fineGridCell);

  //Refinement criterion
  assertion1(tarch::la::greater(patch.getDemandedMeshWidth(), 0), patch);

  if(tarch::la::oneGreater(patch.getSubcellSize(), tarch::la::Vector<DIMENSIONS, double>(patch.getDemandedMeshWidth()))) {
    // Refine
    for(int i = 0; i < TWO_POWER_D; i++) {
      if(!fineGridVertices[fineGridVerticesEnumerator(i)].isHangingNode()) {
        fineGridVertices[fineGridVerticesEnumerator(i)].setShouldRefine(true);
        coarseGridVertices[coarseGridVerticesEnumerator(i)].setSubcellEraseVeto(i);
      }
    }
  } else if (!tarch::la::oneGreater(patch.getSubcellSize() * 3.0, tarch::la::Vector<DIMENSIONS, double>(patch.getDemandedMeshWidth()))) {
    // Coarsen -> default behavior is to coarsen, so do nothing here (i.e. don't set an erase/coarsen veto)
  } else {
    for(int i = 0; i < TWO_POWER_D; i++) {
      coarseGridVertices[coarseGridVerticesEnumerator(i)].setSubcellEraseVeto(i);
      if(fineGridVertices[fineGridVerticesEnumerator(i)].isHangingNode()
          && !coarseGridVertices[coarseGridVerticesEnumerator(i)].isHangingNode()) {
        coarseGridVertices[coarseGridVerticesEnumerator(i)].setShouldRefine(true);
      }
    }
  }

  //TODO unterweg dissertation
  //Veto Coarsening if current cell is refined or if it belongs to a remote rank.
  //In the first case, coarsening the coarse vertices contradicts the restriction to only erase one level at a time
  //In the second case, we can assume that the remote cell is refined (otherwise it couldn't be forked), so the same
  //  reason as in the first case holds.
  if(!fineGridCell.isLeaf()
      #ifdef Parallel
      || fineGridCell.isAssignedToRemoteRank()
      #endif
    ) {
    for(int i = 0; i < TWO_POWER_D; i++) {
      coarseGridVertices[coarseGridVerticesEnumerator(i)].setSubcellEraseVeto(i);
    }
  }
  logTraceOutWith1Argument( "leaveCell(...)", fineGridCell );
}


void peanoclaw::mappings::SolveTimestep::beginIteration(
  peanoclaw::State&  solverState
) {
  logTraceInWith1Argument( "beginIteration(State)", solverState );

  if(!tarch::la::equals(_globalTimestepEndTime, solverState.getGlobalTimestepEndTime())) {
    _globalTimestepEndTime = solverState.getGlobalTimestepEndTime();

    for(std::map<int, int>::iterator i = _estimatedRemainingIterationsUntilGlobalTimestep.begin(); i != _estimatedRemainingIterationsUntilGlobalTimestep.end(); i++) {
      i->second = 1;
    }
  }
 
  _numerics = solverState.getNumerics();
  _domainOffset = solverState.getDomainOffset();
  _domainSize = solverState.getDomainSize();
  _initialMaximalSubgridSize = solverState.getInitialMaximalSubgridSize();
  _probeList = solverState.getProbeList();
  _useDimensionalSplittingOptimization = solverState.useDimensionalSplittingOptimization();
  _subgridStatistics = peanoclaw::statistics::SubgridStatistics(solverState);

  #ifdef Parallel
  LevelStatisticsHeap::getInstance().startToSendSynchronousData();
  LevelStatisticsHeap::getInstance().startToSendBoundaryData(solverState.isTraversalInverted());
  TimeIntervalStatisticsHeap::getInstance().startToSendSynchronousData();
  TimeIntervalStatisticsHeap::getInstance().startToSendSynchronousData();
  #endif
 
  logTraceOutWith1Argument( "beginIteration(State)", solverState);
}


void peanoclaw::mappings::SolveTimestep::endIteration(
  peanoclaw::State&  solverState
) {
  logTraceInWith1Argument( "endIteration(State)", solverState );
 
  _subgridStatistics.finalizeIteration(solverState);

  LevelStatisticsHeap::getInstance().finishedToSendBoundaryData(solverState.isTraversalInverted());
  if(tarch::parallel::Node::getInstance().isGlobalMaster()) {
    LevelStatisticsHeap::getInstance().finishedToSendSynchronousData();
  }
  TimeIntervalStatisticsHeap::getInstance().finishedToSendBoundaryData(solverState.isTraversalInverted());
  if(tarch::parallel::Node::getInstance().isGlobalMaster()) {
    TimeIntervalStatisticsHeap::getInstance().finishedToSendSynchronousData();
  }


  logTraceOutWith1Argument( "endIteration(State)", solverState);
}



void peanoclaw::mappings::SolveTimestep::descend(
  peanoclaw::Cell * const          fineGridCells,
  peanoclaw::Vertex * const        fineGridVertices,
  const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
  peanoclaw::Vertex * const        coarseGridVertices,
  const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
  peanoclaw::Cell&                 coarseGridCell
) {
  logTraceInWith2Arguments( "descend(...)", coarseGridCell.toString(), coarseGridVerticesEnumerator.toString() );
  // @todo Insert your code here
  logTraceOut( "descend(...)" );
}


void peanoclaw::mappings::SolveTimestep::ascend(
  peanoclaw::Cell * const    fineGridCells,
  peanoclaw::Vertex * const  fineGridVertices,
  const peano::grid::VertexEnumerator&          fineGridVerticesEnumerator,
  peanoclaw::Vertex * const  coarseGridVertices,
  const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
  peanoclaw::Cell&           coarseGridCell
) {
  logTraceInWith2Arguments( "ascend(...)", coarseGridCell.toString(), coarseGridVerticesEnumerator.toString() );
  // @todo Insert your code here
  logTraceOut( "ascend(...)" );
}
