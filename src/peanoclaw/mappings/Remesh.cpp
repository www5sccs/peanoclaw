#include "peanoclaw/mappings/Remesh.h"

#include "peanoclaw/Heap.h"
#include "peanoclaw/Numerics.h"
#include "peanoclaw/ParallelSubgrid.h"
#include "peanoclaw/interSubgridCommunication/GridLevelTransfer.h"
#include "peanoclaw/parallel/NeighbourCommunicator.h"
#include "peanoclaw/parallel/MasterWorkerAndForkJoinCommunicator.h"

#include "peano/grid/aspects/VertexStateAnalysis.h"

#include "tarch/parallel/Node.h"

peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids::VertexMap peanoclaw::mappings::Remesh::_vertexPositionToIndexMap;
peanoclaw::parallel::NeighbourCommunicator::RemoteSubgridMap               peanoclaw::mappings::Remesh::_remoteSubgridMap;

/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::Remesh::touchVertexLastTimeSpecification() {
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::AvoidFineGridRaces,false);
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::Remesh::touchVertexFirstTimeSpecification() { 
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::AvoidFineGridRaces,false);
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::Remesh::enterCellSpecification() {
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::AvoidFineGridRaces,false);
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::Remesh::leaveCellSpecification() {
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::AvoidFineGridRaces,false);
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::Remesh::ascendSpecification() {
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::AvoidFineGridRaces,false);
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::Remesh::descendSpecification() {
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::AvoidFineGridRaces,false);
}

tarch::logging::Log                peanoclaw::mappings::Remesh::_log( "peanoclaw::mappings::Remesh" ); 

peanoclaw::mappings::Remesh::Remesh()
: _unknownsPerSubcell(-1),
  _auxiliarFieldsPerSubcell(-1),
  _defaultSubdivisionFactor(-1),
  _defaultGhostLayerWidth(-1),
  _initialTimestepSize(0.0),
  _numerics(0),
  _gridLevelTransfer(),
  _isInitializing(false),
  _useDimensionalSplittingOptimization(false),
  _parallelStatistics(""),
  _state() {
  logTraceIn( "Remesh()" );
  // @todo Insert your code here
  logTraceOut( "Remesh()" );
}


peanoclaw::mappings::Remesh::~Remesh() {
  logTraceIn( "~Remesh()" );
  // @todo Insert your code here
  logTraceOut( "~Remesh()" );
}


#if defined(SharedMemoryParallelisation)
peanoclaw::mappings::Remesh::Remesh(const Remesh&  masterThread)
: _unknownsPerSubcell(masterThread._unknownsPerSubcell),
  _auxiliarFieldsPerSubcell(masterThread._auxiliarFieldsPerSubcell),
  _defaultSubdivisionFactor(masterThread._defaultSubdivisionFactor),
  _defaultGhostLayerWidth(masterThread._defaultGhostLayerWidth),
  _initialTimestepSize(masterThread._initialTimestepSize),
  _numerics(masterThread._numerics),
  _domainOffset(masterThread._domainOffset),
  _domainSize(masterThread._domainSize),
  _gridLevelTransfer(masterThread._gridLevelTransfer),
  _initialMinimalMeshWidth(masterThread._initialMinimalMeshWidth),
  _isInitializing(masterThread._isInitializing),
  _useDimensionalSplittingOptimization(masterThread._useDimensionalSplittingOptimization),
  _parallelStatistics(masterThread._parallelStatistics),
  _state(masterThread._state)
{
  logTraceIn( "Remesh(Remesh)" );
  // @todo Insert your code here
  logTraceOut( "Remesh(Remesh)" );
}


void peanoclaw::mappings::Remesh::mergeWithWorkerThread(const Remesh& workerThread) {
  logTraceIn( "mergeWithWorkerThread(Remesh)" );

  _parallelStatistics.merge(workerThread._parallelStatistics);

  logTraceOut( "mergeWithWorkerThread(Remesh)" );
}
#endif


void peanoclaw::mappings::Remesh::createHangingVertex(
      peanoclaw::Vertex&     fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                fineGridH,
      peanoclaw::Vertex * const   coarseGridVertices,
      const peano::grid::VertexEnumerator&      coarseGridVerticesEnumerator,
      peanoclaw::Cell&       coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                   fineGridPositionOfVertex
) {
  logTraceInWith6Arguments( "createHangingVertex(...)", fineGridVertex, fineGridX, fineGridH, coarseGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfVertex );

  fineGridVertex.setShouldRefine(false);
  fineGridVertex.resetSubcellsEraseVeto();

  peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids adjacentSubgrids(
    fineGridVertex,
    _vertexPositionToIndexMap,
    fineGridX,
    (coarseGridVerticesEnumerator.getLevel() + 1)
  );
  adjacentSubgrids.createHangingVertex(
    coarseGridVertices,
    coarseGridVerticesEnumerator,
    fineGridPositionOfVertex,
    _domainOffset,
    _domainSize,
    *_gridLevelTransfer
  );

  logTraceOutWith1Argument( "createHangingVertex(...)", fineGridVertex );
}


void peanoclaw::mappings::Remesh::destroyHangingVertex(
      const peanoclaw::Vertex&   fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridH,
      peanoclaw::Vertex * const  coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&           coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfVertex
) {
  logTraceInWith6Arguments( "destroyHangingVertex(...)", fineGridVertex, fineGridX, fineGridH, coarseGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfVertex );

  //Handle refinement flags
  if(tarch::la::allGreater(fineGridX, _domainOffset) && tarch::la::allGreater(_domainOffset + _domainSize, fineGridX)) {
    _gridLevelTransfer->restrictRefinementFlagsToCoarseVertices(
      coarseGridVertices,
      coarseGridVerticesEnumerator,
      fineGridVertex,
      fineGridPositionOfVertex
    );
  }

  peanoclaw::Vertex vertex = fineGridVertex;
  peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids adjacentSubgrids(
    vertex,
    _vertexPositionToIndexMap,
    fineGridX,
    (coarseGridVerticesEnumerator.getLevel() + 1)
  );
  adjacentSubgrids.destroyHangingVertex(
    _domainOffset,
    _domainSize
  );

  logTraceOutWith1Argument( "destroyHangingVertex(...)", fineGridVertex );
}


void peanoclaw::mappings::Remesh::createInnerVertex(
      peanoclaw::Vertex&               fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridH,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfVertex
) {
  logTraceInWith6Arguments( "createInnerVertex(...)", fineGridVertex, fineGridX, fineGridH, coarseGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfVertex );

  fineGridVertex.setShouldRefine(false);
  fineGridVertex.resetSubcellsEraseVeto();

  peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids adjacentSubgrids(
    fineGridVertex,
    _vertexPositionToIndexMap,
    fineGridX,
    (coarseGridVerticesEnumerator.getLevel() + 1)
  );
  adjacentSubgrids.convertHangingVertexToPersistentVertex();

  logTraceOutWith1Argument( "createInnerVertex(...)", fineGridVertex );
}


void peanoclaw::mappings::Remesh::createBoundaryVertex(
      peanoclaw::Vertex&               fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridH,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfVertex
) {
  logTraceInWith6Arguments( "createBoundaryVertex(...)", fineGridVertex, fineGridX, fineGridH, coarseGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfVertex );
 
  fineGridVertex.setShouldRefine(false);
  fineGridVertex.resetSubcellsEraseVeto();

  peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids adjacentSubgrids(
    fineGridVertex,
    _vertexPositionToIndexMap,
    fineGridX,
    (coarseGridVerticesEnumerator.getLevel() + 1)
  );
  adjacentSubgrids.convertHangingVertexToPersistentVertex();

  logTraceOutWith1Argument( "createBoundaryVertex(...)", fineGridVertex );
}


void peanoclaw::mappings::Remesh::destroyVertex(
      const peanoclaw::Vertex&   fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridH,
      peanoclaw::Vertex * const  coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&           coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfVertex
) {
  logTraceInWith6Arguments( "destroyVertex(...)", fineGridVertex, fineGridX, fineGridH, coarseGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfVertex );

  peanoclaw::Vertex vertex = fineGridVertex;
  peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids adjacentSubgrids(
    vertex,
    _vertexPositionToIndexMap,
    fineGridX,
    (coarseGridVerticesEnumerator.getLevel() + 1)
  );
  adjacentSubgrids.convertPersistentToHangingVertex();

  logTraceOutWith1Argument( "destroyVertex(...)", fineGridVertex );
}


void peanoclaw::mappings::Remesh::createCell(
      peanoclaw::Cell&                 fineGridCell,
      peanoclaw::Vertex * const        fineGridVertices,
      const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell
) {
  logTraceInWith6Arguments( "createCell(...)", fineGridCell, fineGridVerticesEnumerator.toString(), coarseGridCell, coarseGridVerticesEnumerator.toString(), fineGridPositionOfCell, fineGridVerticesEnumerator.getCellCenter() );

  //Initialise new Patch
  Patch fineGridPatch = Patch(
    fineGridVerticesEnumerator.getVertexPosition(0),
    fineGridVerticesEnumerator.getCellSize(),
    _unknownsPerSubcell,
    _auxiliarFieldsPerSubcell,
    _defaultSubdivisionFactor,
    _defaultGhostLayerWidth,
    _initialTimestepSize,
    fineGridVerticesEnumerator.getLevel()
  );
  fineGridCell.setCellDescriptionIndex(fineGridPatch.getCellDescriptionIndex());

//  std::cout << "Creating cell on rank "
//      #ifdef Parallel
//      << tarch::parallel::Node::getInstance().getRank() << ": "
//      #endif
//      << fineGridVerticesEnumerator.getVertexPosition(0) << ", "
//      << fineGridVerticesEnumerator.getCellSize()
//      << ", index=" << fineGridCell.getCellDescriptionIndex()
//      << ", level=" << fineGridVerticesEnumerator.getLevel()
//      << std::endl;

  if(fineGridCell.isLeaf()) {
    assertion1(!fineGridPatch.isLeaf(), fineGridPatch);
    fineGridPatch.switchToVirtual();
    fineGridPatch.switchToLeaf();
  }

  //Transfer data from coarse to fine patch
  if(!coarseGridCell.isRoot()) {
    assertion4(coarseGridCell.getCellDescriptionIndex() > -1, coarseGridCell.getCellDescriptionIndex(), fineGridVerticesEnumerator.getCellSize(), fineGridVerticesEnumerator.getLevel(), fineGridVerticesEnumerator.getVertexPosition());
    Patch coarseGridPatch(
      coarseGridCell
    );
    assertion1(tarch::la::greaterEquals(coarseGridPatch.getTimestepSize(), 0.0) || coarseGridPatch.isVirtual(), coarseGridPatch);

    if(!_isInitializing && (coarseGridPatch.isVirtual() || coarseGridPatch.isLeaf())) {
      //TODO unterweg dissertation: The grid is skipped directly after the creation in enterCell.
      //Therefore, we need to skip at least two iterations to ensure that all ghostlayers have been set.
      fineGridPatch.setSkipNextGridIteration(2);

      fineGridPatch.setCurrentTime(coarseGridPatch.getCurrentTime());
      fineGridPatch.setTimestepSize(coarseGridPatch.getTimestepSize());
      fineGridPatch.setEstimatedNextTimestepSize(coarseGridPatch.getEstimatedNextTimestepSize());
      fineGridPatch.updateMinimalNeighborTimeConstraint(
        coarseGridPatch.getMinimalNeighborTimeConstraint(),
        coarseGridPatch.getCellDescriptionIndex()
      );

      //Only interpolate if not forking
      if(
        #ifdef Parallel
        !_state->isNewWorkerDueToForkOfExistingDomain()
        #else
        true
        #endif
      ) {
        _numerics->interpolate(
          fineGridPatch.getSubdivisionFactor(),
          0,
          coarseGridPatch,
          fineGridPatch,
          false
        );
        _numerics->interpolate(
          fineGridPatch.getSubdivisionFactor(),
          0,
          coarseGridPatch,
          fineGridPatch,
          true
        );
      }
    }
  }

  //Set indices on adjacent vertices and on this cell
  for(int i = 0; i < TWO_POWER_D; i++) {
    peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids adjacentSubgrids(
      fineGridVertices[fineGridVerticesEnumerator(i)],
      _vertexPositionToIndexMap,
      fineGridVerticesEnumerator.getVertexPosition(i),
      fineGridVerticesEnumerator.getLevel()
    );
    adjacentSubgrids.createdAdjacentSubgrid(
      fineGridCell.getCellDescriptionIndex(),
      i
    );
  }

  logTraceOutWith2Arguments( "createCell(...)", fineGridCell, fineGridPatch );
}


void peanoclaw::mappings::Remesh::destroyCell(
      const peanoclaw::Cell&           fineGridCell,
      peanoclaw::Vertex * const        fineGridVertices,
      const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell
) {
  logTraceInWith4Arguments( "destroyCell(...)", fineGridCell, fineGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfCell );

  assertion5(
    fineGridCell.getCellDescriptionIndex() != -2,
    fineGridCell.toString(),
    fineGridVerticesEnumerator.getVertexPosition(0),
    fineGridVerticesEnumerator.getCellSize(),
    fineGridVerticesEnumerator.getLevel(),
    #ifdef Parallel
    tarch::parallel::Node::getInstance().getRank()
    #else
    0
    #endif
  );

  Patch finePatch(
    fineGridCell
  );

  bool isDestroyedDueToForkOrJoin = fineGridCell.isAssignedToRemoteRank();
  bool isRootOfNewWorker = isDestroyedDueToForkOrJoin && !coarseGridCell.isAssignedToRemoteRank();

  if(fineGridCell.isInside() && !isRootOfNewWorker && !fineGridCell.isAssignedToRemoteRank()) {
      //Delete patch data and description from this cell
      assertion3(fineGridCell.getCellDescriptionIndex() > -1,
        fineGridCell,
        fineGridVerticesEnumerator.getVertexPosition(0),
        fineGridVerticesEnumerator.getCellSize()
      );

	  //Create patch in parent cell if it doesn't exist
	  if(!coarseGridCell.isRoot() && coarseGridCell.isInside()) {
	    Patch coarseSubgrid(CellDescriptionHeap::getInstance().getData(coarseGridCell.getCellDescriptionIndex()).at(0));
	    _gridLevelTransfer->restrictDestroyedSubgrid(
	      finePatch,
	      coarseSubgrid,
	      fineGridVertices,
	      fineGridVerticesEnumerator
	    );
	  } else {
		for(int i = 0; i < TWO_POWER_D; i++) {
          fineGridVertices[fineGridVerticesEnumerator(i)].setAdjacentCellDescriptionIndex(i, -1);
		}
	  }

	  finePatch.deleteData();
  } else if(fineGridCell.isAssignedToRemoteRank()) {
    //Patch got moved to other rank, check whether it is now adjacent to the local domain.
    ParallelSubgrid parallelSubgrid(fineGridCell.getCellDescriptionIndex());

    //If it is adjacent -> Now remote
    //If not -> Delete it
    if(parallelSubgrid.isAdjacentToLocalSubdomain(coarseGridCell, fineGridVertices, fineGridVerticesEnumerator)) {
      #ifdef Parallel
      peanoclaw::parallel::NeighbourCommunicator communicator(
        -1,
        fineGridVerticesEnumerator.getVertexPosition(0),
        fineGridVerticesEnumerator.getLevel(),
        fineGridVerticesEnumerator.getCellSize(),
        _remoteSubgridMap,
        _parallelStatistics);
      communicator.switchToRemote(finePatch);
      #endif
    } else {
      finePatch.deleteData();
      for(int i = 0; i < TWO_POWER_D; i++) {
        fineGridVertices[fineGridVerticesEnumerator(i)].setAdjacentCellDescriptionIndex(i, -1);
      }
    }

    #ifdef Parallel
    assertion1(!finePatch.isValid() || finePatch.isRemote(), finePatch);
    #endif
  }

  logTraceOutWith1Argument( "destroyCell(...)", fineGridCell );
}

#ifdef Parallel
void peanoclaw::mappings::Remesh::mergeWithNeighbour(
  peanoclaw::Vertex&  vertex,
  const peanoclaw::Vertex&  neighbour,
  int                                           fromRank,
  const tarch::la::Vector<DIMENSIONS,double>&   fineGridX,
  const tarch::la::Vector<DIMENSIONS,double>&   fineGridH,
  int                                           level
) {
  logTraceInWith6Arguments( "mergeWithNeighbour(...)", vertex, neighbour, fromRank, fineGridX, fineGridH, level );

  peanoclaw::parallel::NeighbourCommunicator communicator(fromRank, fineGridX, level, fineGridH, _remoteSubgridMap, _parallelStatistics);
  communicator.receiveSubgridsForVertex(
    vertex,
    neighbour,
    fineGridX,
    fineGridH,
    level
  );

  logTraceOut( "mergeWithNeighbour(...)" );
}

void peanoclaw::mappings::Remesh::prepareSendToNeighbour(
  peanoclaw::Vertex&  vertex,
  int                                           toRank,
  const tarch::la::Vector<DIMENSIONS,double>&   x,
  const tarch::la::Vector<DIMENSIONS,double>&   h,
  int                                           level
) {
  logTraceInWith3Arguments( "prepareSendToNeighbour(...)", vertex, toRank, level );

  peanoclaw::parallel::NeighbourCommunicator communicator(toRank, x, level, h, _remoteSubgridMap, _parallelStatistics);
  communicator.sendSubgridsForVertex(vertex, x, h, level);

  logTraceOut( "prepareSendToNeighbour(...)" );
}

void peanoclaw::mappings::Remesh::prepareCopyToRemoteNode(
  peanoclaw::Vertex&  localVertex,
  int                                           toRank,
  const tarch::la::Vector<DIMENSIONS,double>&   x,
  const tarch::la::Vector<DIMENSIONS,double>&   h,
  int                                           level
) {
  logTraceInWith2Arguments( "prepareCopyToRemoteNode(...)", localVertex, toRank);
  // @todo Insert your code here

  logTraceOut( "prepareCopyToRemoteNode(...)" );
}

void peanoclaw::mappings::Remesh::prepareCopyToRemoteNode(
  peanoclaw::Cell&  localCell,
  int  toRank,
  const tarch::la::Vector<DIMENSIONS,double>&  cellCentre,
  const tarch::la::Vector<DIMENSIONS,double>&  cellSize,
  int                                          level
) {
  logTraceInWith5Arguments( "prepareCopyToRemoteNode(...)", localCell, toRank, cellCentre, cellSize, level );

  if(localCell.isInside() && localCell.getRankOfRemoteNode() == toRank) {
   assertion7(
     localCell.getCellDescriptionIndex() >= 0,
     localCell.getCellDescriptionIndex(),
     cellCentre - cellSize / 2.0,
     cellSize,
     level,
     localCell.isInside(),
     localCell.getRankOfRemoteNode(),
     localCell.isAssignedToRemoteRank()
   );

    peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator communicator(toRank, cellCentre, level, true);
    communicator.sendPatch(localCell.getCellDescriptionIndex());

    //Switch to remote after having sent the patch away...
    Patch patch(localCell);
    patch.setIsRemote(true);
  }
  logTraceOut( "prepareCopyToRemoteNode(...)" );
}

void peanoclaw::mappings::Remesh::mergeWithRemoteDataDueToForkOrJoin(
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

void peanoclaw::mappings::Remesh::mergeWithRemoteDataDueToForkOrJoin(
  peanoclaw::Cell&  localCell,
  const peanoclaw::Cell&  masterOrWorkerCell,
  int                                       fromRank,
  const tarch::la::Vector<DIMENSIONS,double>&  cellCentre,
  const tarch::la::Vector<DIMENSIONS,double>&  cellSize,
  int                                       level
) {
  logTraceInWith6Arguments( "mergeWithRemoteDataDueToForkOrJoin(...)", localCell, masterOrWorkerCell, fromRank, cellCentre, cellSize, level );

  assertion3(localCell.isAssignedToRemoteRank() || localCell.getCellDescriptionIndex() != -2, localCell.toString(), cellCentre, cellSize);

  peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator communicator(fromRank, cellCentre, level, true);
  communicator.mergeCellDuringForkOrJoin(
    localCell,
    masterOrWorkerCell,
    cellSize,
    *_state
  );

  logTraceOut( "mergeWithRemoteDataDueToForkOrJoin(...)" );
}

bool peanoclaw::mappings::Remesh::prepareSendToWorker(
  peanoclaw::Cell&                 fineGridCell,
  peanoclaw::Vertex * const        fineGridVertices,
  const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
  peanoclaw::Vertex * const        coarseGridVertices,
  const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
  peanoclaw::Cell&                 coarseGridCell,
  const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell,
  int                                                                  worker
) {
  logTraceInWith7Arguments( "prepareSendToWorker(...)",
    fineGridCell,
    fineGridVerticesEnumerator.toString(),
    fineGridVerticesEnumerator.getVertexPosition(0),
    coarseGridCell,
    coarseGridVerticesEnumerator.toString(),
    fineGridPositionOfCell,
    worker
  );

  if(fineGridCell.isInside()){

    assertion4(
      CellDescriptionHeap::getInstance().isValidIndex(fineGridCell.getCellDescriptionIndex()),
      fineGridVerticesEnumerator.getVertexPosition(0),
      fineGridVerticesEnumerator.getCellSize(),
      fineGridCell.getCellDescriptionIndex(),
      worker
    );

    peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator communicator(worker, fineGridVerticesEnumerator.getCellCenter(), fineGridVerticesEnumerator.getLevel(), false);
    communicator.sendPatch(fineGridCell.getCellDescriptionIndex());
  }

  logTraceOut( "prepareSendToWorker(...)" );
  return true;
}

void peanoclaw::mappings::Remesh::prepareSendToMaster(
  peanoclaw::Cell&                       localCell,
  peanoclaw::Vertex *                    vertices,
  const peano::grid::VertexEnumerator&       verticesEnumerator,
  const peanoclaw::Vertex * const        coarseGridVertices,
  const peano::grid::VertexEnumerator&       coarseGridVerticesEnumerator,
  const peanoclaw::Cell&                 coarseGridCell,
  const tarch::la::Vector<DIMENSIONS,int>&   fineGridPositionOfCell
) {
  logTraceInWith3Arguments( "prepareSendToMaster(...)", localCell, verticesEnumerator.toString(), verticesEnumerator.getVertexPosition(0) );
  
  int toRank = tarch::parallel::NodePool::getInstance().getMasterRank();
  if(localCell.isInside()){
    peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator communicator(toRank, verticesEnumerator.getCellCenter(), verticesEnumerator.getLevel(), false);
    communicator.sendPatch(localCell.getCellDescriptionIndex());
  }

  CellDescriptionHeap::getInstance().finishedToSendSynchronousData();
  DataHeap::getInstance().finishedToSendSynchronousData();

  logTraceOut( "prepareSendToMaster(...)" );
}


void peanoclaw::mappings::Remesh::mergeWithMaster(
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
  logTraceInWith7Arguments( "mergeWithMaster(...)", workerGridCell, fineGridCell, fineGridVerticesEnumerator.toString(), coarseGridCell, coarseGridVerticesEnumerator.toString(), fineGridPositionOfCell, worker );

  peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator communicator(
    worker,
    fineGridVerticesEnumerator.getCellCenter(),
    fineGridVerticesEnumerator.getLevel(),
    false
  );

  communicator.mergeWorkerStateIntoMasterState(workerState, masterState);

  if(fineGridCell.isInside()) {
    communicator.receivePatch(fineGridCell.getCellDescriptionIndex());

    assertionEquals1(
      fineGridCell.getCellDescriptionIndex(),
      CellDescriptionHeap::getInstance().getData(fineGridCell.getCellDescriptionIndex()).at(0).getCellDescriptionIndex(),
      fineGridCell
    );
  }

  logTraceOut( "mergeWithMaster(...)" );
}


void peanoclaw::mappings::Remesh::receiveDataFromMaster(
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
  
  //TODO unterweg debug
//  std::cout << "Receiving data from master: " << receivedVerticesEnumerator.getVertexPosition(tarch::la::Vector<DIMENSIONS, int>(0)) << " " << receivedVerticesEnumerator.getLevel()
//      << " size " << receivedVerticesEnumerator.getCellSize()
//      << " on rank " << tarch::parallel::Node::getInstance().getRank() << std::endl;

  if(receivedCell.isInside()) {
    peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator communicator(
      tarch::parallel::NodePool::getInstance().getMasterRank(),
      receivedVerticesEnumerator.getCellCenter(),
      receivedVerticesEnumerator.getLevel(),
      false
    );

    int temporaryCellDescriptionIndex = CellDescriptionHeap::getInstance().createData();
    CellDescription temporaryCellDescription;
    temporaryCellDescription.setUNewIndex(-1);
//    temporaryCellDescription.setUOldIndex(-1);
//    temporaryCellDescription.setAuxIndex(-1);
    temporaryCellDescription.setPosition(
      receivedVerticesEnumerator.getVertexPosition(tarch::la::Vector<DIMENSIONS, int>(0))
    );
    temporaryCellDescription.setSize(receivedVerticesEnumerator.getCellSize());
    temporaryCellDescription.setLevel(receivedVerticesEnumerator.getLevel());
    CellDescriptionHeap::getInstance().getData(temporaryCellDescriptionIndex).push_back(temporaryCellDescription);
    receivedCell.setCellDescriptionIndex(temporaryCellDescriptionIndex);

    communicator.receivePatch(temporaryCellDescriptionIndex);
  } else {
    receivedCell.setCellDescriptionIndex(-1);
  }

  logTraceOut( "receiveDataFromMaster(...)" );
}


void peanoclaw::mappings::Remesh::mergeWithWorker(
  peanoclaw::Cell&           localCell, 
  const peanoclaw::Cell&     receivedMasterCell,
  const tarch::la::Vector<DIMENSIONS,double>&  cellCentre,
  const tarch::la::Vector<DIMENSIONS,double>&  cellSize,
  int                                          level
) {
  logTraceInWith2Arguments( "mergeWithWorker(...)", localCell.toString(), receivedMasterCell.toString() );

  if(!_state->isNewWorkerDueToForkOfExistingDomain()) {
    //Avoid this in first iteration for new worker, since
    //prepareSendToMaster is not called in such an
    //iteration.
    _gridLevelTransfer->updatePatchStateDuringMergeWithWorker(
      localCell.getCellDescriptionIndex(),
      receivedMasterCell.getCellDescriptionIndex()
    );
  }

  #ifdef Asserts
  {
  CellDescription& localCellDescription = CellDescriptionHeap::getInstance().getData(localCell.getCellDescriptionIndex()).at(0);
  Patch localPatch(localCellDescription);
  assertionEquals1(localPatch.getLevel(), level, localPatch);
  }
  #endif

  logTraceOutWith1Argument( "mergeWithWorker(...)", localCell.toString() );
}


void peanoclaw::mappings::Remesh::mergeWithWorker(
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

void peanoclaw::mappings::Remesh::touchVertexFirstTime(
      peanoclaw::Vertex&               fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridH,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfVertex
) {
  logTraceInWith6Arguments( "touchVertexFirstTime(...)", fineGridVertex, fineGridX, fineGridH, coarseGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfVertex );
  // @todo Insert your code here
  logTraceOutWith1Argument( "touchVertexFirstTime(...)", fineGridVertex );
}


void peanoclaw::mappings::Remesh::touchVertexLastTime(
      peanoclaw::Vertex&         fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridH,
      peanoclaw::Vertex * const  coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&           coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfVertex
) {
  logTraceInWith6Arguments( "touchVertexLastTime(...)", fineGridVertex, fineGridX, fineGridH, coarseGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfVertex );

  peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids adjacentSubgrids(
    fineGridVertex,
    _vertexPositionToIndexMap,
    fineGridX,
    coarseGridVerticesEnumerator.getLevel()+1
  );

  adjacentSubgrids.refineOnParallelAndAdaptiveBoundary();

  adjacentSubgrids.regainTwoIrregularity(
    coarseGridVertices,
    coarseGridVerticesEnumerator,
    fineGridPositionOfVertex
  );

  //Mark vertex as "old" (i.e. older than just created ;-))
  fineGridVertex.setWasCreatedInThisIteration(false);

  logTraceOutWith1Argument( "touchVertexLastTime(...)", fineGridVertex );
}


void peanoclaw::mappings::Remesh::enterCell(
      peanoclaw::Cell&                 fineGridCell,
      peanoclaw::Vertex * const        fineGridVertices,
      const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell
) {
  logTraceInWith5Arguments( "enterCell(...)", fineGridCell, fineGridVerticesEnumerator.toString(), coarseGridCell, coarseGridVerticesEnumerator.toString(), fineGridPositionOfCell );

  Patch patch(
    fineGridCell
  );

  assertion(patch.isLeaf() || !patch.isLeaf());

  #ifdef Parallel
  assertionEquals4(patch.getLevel(),
    fineGridVerticesEnumerator.getLevel(),
    patch,
    fineGridVerticesEnumerator.getVertexPosition(0),
    fineGridVerticesEnumerator.getCellSize(),
    tarch::parallel::Node::getInstance().getRank()
  );
  #endif

  _gridLevelTransfer->stepDown(
    coarseGridCell.isRoot() ? -1 : coarseGridCell.getCellDescriptionIndex(),
    patch,
    fineGridVertices,
    fineGridVerticesEnumerator,
    _isInitializing
  );

  #ifdef Asserts
  if(patch.isLeaf() && !fineGridCell.isLeaf()) {
    bool isRefining = false;
    for(int i = 0; i < TWO_POWER_D; i++) {
      if(fineGridVertices[fineGridVerticesEnumerator(i)].getRefinementControl() == peanoclaw::records::Vertex::Refining) {
        isRefining = true;
      }
    }
    assertion1(isRefining, patch);
  }
  #endif
  logTraceOutWith2Arguments( "enterCell(...)", fineGridCell, patch );
}


void peanoclaw::mappings::Remesh::leaveCell(
      peanoclaw::Cell&           fineGridCell,
      peanoclaw::Vertex * const  fineGridVertices,
      const peano::grid::VertexEnumerator&          fineGridVerticesEnumerator,
      peanoclaw::Vertex * const  coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&           coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfCell
) {
  logTraceInWith4Arguments( "leaveCell(...)", fineGridCell, fineGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfCell );

  assertion(fineGridCell.isInside());
  assertion(fineGridCell.getCellDescriptionIndex() != -1);
  assertion(coarseGridCell.getCellDescriptionIndex() != -1);

  Patch finePatch(
    fineGridCell
  );

  _gridLevelTransfer->stepUp(
    coarseGridCell.getCellDescriptionIndex(),
    finePatch,
    fineGridCell.isLeaf(),
    fineGridVertices,
    fineGridVerticesEnumerator
  );

  assertionEquals1(finePatch.isLeaf(), fineGridCell.isLeaf(), finePatch);
  assertionEquals1(finePatch.getLevel(), fineGridVerticesEnumerator.getLevel(), finePatch.toString());

  if(!fineGridCell.isAssignedToRemoteRank()) {
    //Count number of adjacent subgrids
    ParallelSubgrid parallelSubgrid(fineGridCell.getCellDescriptionIndex());
    parallelSubgrid.countNumberOfAdjacentParallelSubgrids(
      fineGridVertices,
      fineGridVerticesEnumerator
    );
  }

  logTraceOutWith1Argument( "leaveCell(...)", fineGridCell );
}


void peanoclaw::mappings::Remesh::beginIteration(
  peanoclaw::State&  solverState
) {
  logTraceInWith1Argument( "beginIteration(State)", solverState );

  _unknownsPerSubcell       = solverState.getUnknownsPerSubcell();
  _auxiliarFieldsPerSubcell = solverState.getAuxiliarFieldsPerSubcell();
  _defaultSubdivisionFactor = solverState.getDefaultSubdivisionFactor();
  _defaultGhostLayerWidth   = solverState.getDefaultGhostLayerWidth();
  _initialTimestepSize      = solverState.getInitialTimestepSize();
  _numerics                 = solverState.getNumerics();
  _domainOffset             = solverState.getDomainOffset();
  _domainSize               = solverState.getDomainSize();

  _gridLevelTransfer = new peanoclaw::interSubgridCommunication::GridLevelTransfer(
                              solverState.useDimensionalSplittingOptimization(),
                              *_numerics
                           );

  _initialMinimalMeshWidth = solverState.getInitialMaximalSubgridSize();
  _isInitializing = solverState.getIsInitializing();
  _useDimensionalSplittingOptimization = solverState.useDimensionalSplittingOptimization();
  _parallelStatistics = peanoclaw::statistics::ParallelStatistics("Iteration");
  _state = &solverState;

  //Reset touched for all hanging vertex descriptions
  std::map<tarch::la::Vector<DIMENSIONS_PLUS_ONE,double> , VertexDescription, tarch::la::VectorCompare<DIMENSIONS_PLUS_ONE> >::iterator i = _vertexPositionToIndexMap.begin();
  while(i != _vertexPositionToIndexMap.end()) {
    if(i->second.getTouched()) {
      i->second.setTouched(false);
      i++;
    } else {
      _vertexPositionToIndexMap.erase(i++);
    }
  }

  #ifdef Parallel
  DataHeap::getInstance().startToSendSynchronousData();
  DataHeap::getInstance().startToSendBoundaryData(solverState.isTraversalInverted());
  CellDescriptionHeap::getInstance().startToSendSynchronousData();
  CellDescriptionHeap::getInstance().startToSendBoundaryData(solverState.isTraversalInverted());

  if(tarch::parallel::Node::getInstance().isGlobalMaster()) {
    solverState.resetLocalHeightOfWorkerTree();

    logDebug("beginIteration(State)", "Height of worker tree in last grid iteration was " << solverState.getGlobalHeightOfWorkerTreeDuringLastIteration());
  } else {
    solverState.increaseLocalHeightOfWorkerTree();
  }
  #endif

  logTraceOutWith1Argument( "beginIteration(State)", solverState);
}


void peanoclaw::mappings::Remesh::endIteration(
  peanoclaw::State&  solverState
) {
  logTraceInWith1Argument( "endIteration(State)", solverState );

  delete _gridLevelTransfer;

  _parallelStatistics.logStatistics();

  DataHeap::getInstance().finishedToSendBoundaryData(solverState.isTraversalInverted());
  CellDescriptionHeap::getInstance().finishedToSendBoundaryData(solverState.isTraversalInverted());

  if(tarch::parallel::Node::getInstance().isGlobalMaster()) {
    CellDescriptionHeap::getInstance().finishedToSendSynchronousData();
    DataHeap::getInstance().finishedToSendSynchronousData();
  }

  logTraceOutWith1Argument( "endIteration(State)", solverState);
}


void peanoclaw::mappings::Remesh::descend(
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


void peanoclaw::mappings::Remesh::ascend(
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
