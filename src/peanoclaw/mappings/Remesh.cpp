#include "peanoclaw/mappings/Remesh.h"

#include "peanoclaw/interSubgridCommunication/GridLevelTransfer.h"
#include "peanoclaw/Numerics.h"
#include "peanoclaw/parallel/NeighbourCommunicator.h"
#include "peanoclaw/parallel/MasterWorkerAndForkJoinCommunicator.h"

#include "peano/heap/Heap.h"

#include "peano/grid/aspects/VertexStateAnalysis.h"

std::map<tarch::la::Vector<DIMENSIONS_PLUS_ONE,double> , peanoclaw::mappings::Remesh::VertexDescription, tarch::la::VectorCompare<DIMENSIONS_PLUS_ONE> >   peanoclaw::mappings::Remesh::_vertexPositionToIndexMap;

/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::Remesh::touchVertexLastTimeSpecification() {
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::Serial,false);
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::Remesh::touchVertexFirstTimeSpecification() { 
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::Serial,false);
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::Remesh::enterCellSpecification() {
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::Serial,false);
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::Remesh::leaveCellSpecification() {
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::Serial,false);
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::Remesh::ascendSpecification() {
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::Serial,false);
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::Remesh::descendSpecification() {
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::Serial,false);
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
  _sentNeighborData(0),
  _receivedNeighborData(0),
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
peanoclaw::mappings::Remesh::Remesh(const Remesh&  masterThread) {
  logTraceIn( "Remesh(Remesh)" );
  // @todo Insert your code here
  logTraceOut( "Remesh(Remesh)" );
}


void peanoclaw::mappings::Remesh::mergeWithWorkerThread(const Remesh& workerThread) {
  logTraceIn( "mergeWithWorkerThread(Remesh)" );
  // @todo Insert your code here
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

//    std::cout << "Creating cell on rank "
//        #ifdef Parallel
//        << tarch::parallel::Node::getInstance().getRank() << ": "
//        #endif
//        << fineGridVerticesEnumerator.getVertexPosition(0) << ", "
//        << fineGridVerticesEnumerator.getCellSize()
//        << ", index=" << fineGridCell.getCellDescriptionIndex()
//        << ", level=" << fineGridVerticesEnumerator.getLevel()
//        << std::endl;
 
  //Initialise new Patch
  Patch fineGridPatch = Patch(
    fineGridVerticesEnumerator.getVertexPosition(),
    fineGridVerticesEnumerator.getCellSize(),
    _unknownsPerSubcell,
    _auxiliarFieldsPerSubcell,
    _defaultSubdivisionFactor,
    _defaultGhostLayerWidth,
    _initialTimestepSize,
    fineGridVerticesEnumerator.getLevel()
  );
  fineGridCell.setCellDescriptionIndex(fineGridPatch.getCellDescriptionIndex());

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
    assertion1(coarseGridPatch.getTimestepSize() >= 0.0 || coarseGridPatch.isVirtual(), coarseGridPatch);

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
		CellDescription& coarseCellDescription = peano::heap::Heap<CellDescription>::getInstance().getData(coarseGridCell.getCellDescriptionIndex()).at(0);
		assertion1(tarch::la::greaterEquals(coarseCellDescription.getTimestepSize(), 0.0), finePatch);

		//Fix timestep size
		coarseCellDescription.setTimestepSize(std::max(0.0, coarseCellDescription.getTimestepSize()));

		//Set indices on coarse adjacent vertices and fill adjacent ghostlayers
		for(int i = 0; i < TWO_POWER_D; i++) {
		  fineGridVertices[fineGridVerticesEnumerator(i)].setAdjacentCellDescriptionIndex(i, coarseGridCell.getCellDescriptionIndex());
		}

		//Skip update for coarse patch in next grid iteration
		coarseCellDescription.setSkipGridIterations(1);

		//Set demanded mesh width for coarse cell to coarse cell size. Otherwise
		//the coarse patch might get refined immediately.
		coarseCellDescription.setDemandedMeshWidth(coarseGridVerticesEnumerator.getCellSize()(0) / coarseCellDescription.getSubdivisionFactor()(0));
	  } else {
		for(int i = 0; i < TWO_POWER_D; i++) {
			fineGridVertices[fineGridVerticesEnumerator(i)].setAdjacentCellDescriptionIndex(i, -1);
		}
	  }

	  finePatch.deleteData();
  } else if(fineGridCell.isAssignedToRemoteRank()) {
    //Patch got moved to other rank, check whether it is now adjacent to the local domain.
    bool adjacentToLocalDomain = !coarseGridCell.isAssignedToRemoteRank();
    #ifdef Parallel
    for(int i = 0; i < TWO_POWER_D; i++) {
      adjacentToLocalDomain |= fineGridVertices[fineGridVerticesEnumerator(i)].isAdjacentToDomainOf(
          tarch::parallel::Node::getInstance().getRank()
        );
    }
    #endif

    //If it is adjacent -> Now remote
    //If not -> Delete it
    if(adjacentToLocalDomain) {
      #ifdef Parallel
      finePatch.setIsRemote(true);
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

  if(!tarch::parallel::Node::getInstance().isGlobalMaster() && fromRank != 0) {
    assertionEquals(vertex.isInside(), neighbour.isInside());
    assertionEquals(vertex.isBoundary(), neighbour.isBoundary());

    tarch::la::Vector<TWO_POWER_D, int> neighbourVertexRanks = neighbour.getAdjacentRanks();

    for(int i = TWO_POWER_D-1; i >= 0; i--) {
      tarch::la::Vector<DIMENSIONS,double> patchPosition = fineGridX + tarch::la::multiplyComponents(fineGridH, peano::utils::dDelinearised(i, 2).convertScalar<double>() - 1.0);
      peanoclaw::parallel::NeighbourCommunicator communicator(fromRank, patchPosition, level);
      int localAdjacentCellDescriptionIndex = vertex.getAdjacentCellDescriptionIndexInPeanoOrder(i);
      int remoteAdjacentCellDescriptionIndex = neighbour.getAdjacentCellDescriptionIndexInPeanoOrder(i);

      assertion4(
        localAdjacentCellDescriptionIndex != -1
        || vertex.isAdjacentToRemoteRank(),
        localAdjacentCellDescriptionIndex,
        remoteAdjacentCellDescriptionIndex,
        patchPosition,
        level
      );

      if(neighbourVertexRanks(i) == fromRank && remoteAdjacentCellDescriptionIndex != -1) {
        if(localAdjacentCellDescriptionIndex == -1) {
          //Create outside patch
          Patch outsidePatch(
            patchPosition,
            fineGridH,
            _unknownsPerSubcell,
            _auxiliarFieldsPerSubcell,
            _defaultSubdivisionFactor,
            _defaultGhostLayerWidth,
            _initialTimestepSize,
            level
          );
          localAdjacentCellDescriptionIndex = outsidePatch.getCellDescriptionIndex();
          vertex.setAdjacentCellDescriptionIndexInPeanoOrder(i, localAdjacentCellDescriptionIndex);
        }
        assertion(localAdjacentCellDescriptionIndex != -1);

        communicator.receivePatch(localAdjacentCellDescriptionIndex);
      } else {
        //Receive dummy message
        communicator.receivePaddingPatch();
      }

      _receivedNeighborData++;
    }
  }

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

  if(!tarch::parallel::Node::getInstance().isGlobalMaster() && toRank != 0) {
    //TODO unterweg debug
//    logInfo("", "Sending to neighbor " << tarch::parallel::Node::getInstance().getRank()
//      << " to " << toRank
//      << ", position:" << x
//      << ", level:" << level);

    for(int i = 0; i < TWO_POWER_D; i++) {
      #ifdef Asserts
      tarch::la::Vector<DIMENSIONS,double> patchPosition = x + tarch::la::multiplyComponents(h, peano::utils::dDelinearised(i, 2).convertScalar<double>() - 1.0);
      #else
      tarch::la::Vector<DIMENSIONS,double> patchPosition(0.0);
      #endif
      peanoclaw::parallel::NeighbourCommunicator communicator(toRank, patchPosition, level);
      int adjacentCellDescriptionIndex = vertex.getAdjacentCellDescriptionIndexInPeanoOrder(i);
      communicator.sendPatch(adjacentCellDescriptionIndex);

      _sentNeighborData++;
    }
  }

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

//  std::cout << "Copying vertex to remote from " << tarch::parallel::Node::getInstance().getRank()
//      << " to " << toRank
//      << " at " << (x)
//      << " on level " << level
//      << " adjacentRanks=" << localVertex.getAdjacentRanks()
//      << std::endl;

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
//  std::cout << "[prepareCopyToRemoteNode] sending to " << toRank << " next item @ " << cellCentre << " on level " << level << std::endl;

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

    //TODO unterweg debug
//    std::cout << "Copying to remote from " << tarch::parallel::Node::getInstance().getRank() << " to " << toRank << " "
//        << patch
//        << " isInside " << localCell.isInside()
//        << std::endl;

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
  logInfo("", "[mergeWithRemoteDataDueToForkOrJoin] receiving next item @ " << cellCentre << " on level " << level << " localCell.isRemote: " << localCell.isRemote(*_state, false, false));

  assertion3(localCell.isAssignedToRemoteRank() || localCell.getCellDescriptionIndex() != -2, localCell.toString(), cellCentre, cellSize);

  //TODO unterweg debug
  #ifdef Parallel
  std::cout << "Merging with remote data localIndex=" << localCell.getCellDescriptionIndex()
      << " on rank=" << tarch::parallel::Node::getInstance().getRank()
      << " remoteIndex=" << masterOrWorkerCell.getCellDescriptionIndex()
      << " position=" << (cellCentre-0.5*cellSize) << " size=" << cellSize
      << " isInside=" << localCell.isInside() << " isAssignedToRemoteRank=" << masterOrWorkerCell.isAssignedToRemoteRank()
      << std::endl;
  #endif

  if(localCell.isInside() && !masterOrWorkerCell.isAssignedToRemoteRank()) {
    peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator communicator(fromRank, cellCentre, level, true);

    if(localCell.isRemote(*_state, false, false)) {
      if(tarch::parallel::NodePool::getInstance().getMasterRank() != 0) {
        assertionEquals2(localCell.getCellDescriptionIndex(), -2, cellCentre, level);
      }
      Patch temporaryPatch(
        cellCentre - cellSize * 0.5,
        cellSize,
        _unknownsPerSubcell,
        _auxiliarFieldsPerSubcell,
        _defaultSubdivisionFactor,
        _defaultGhostLayerWidth,
        _initialTimestepSize,
        level
      );

      communicator.receivePatch(temporaryPatch.getCellDescriptionIndex());
      temporaryPatch.reloadCellDescription();

      if(temporaryPatch.isLeaf()) {
        temporaryPatch.switchToVirtual();
      }
      if(temporaryPatch.isVirtual()) {
        temporaryPatch.switchToNonVirtual();
      }
      peano::heap::Heap<CellDescription>::getInstance().deleteData(temporaryPatch.getCellDescriptionIndex());
    } else {
      assertion2(localCell.getCellDescriptionIndex() != -1, cellCentre, level);

      Patch localPatch(localCell);

      assertion2(
        (!localPatch.isLeaf() && !localPatch.isVirtual())
        || localPatch.getUNewIndex() >= 0,
        cellCentre,
        level
      );

      communicator.receivePatch(localPatch.getCellDescriptionIndex());
      localPatch.loadCellDescription(localCell.getCellDescriptionIndex());

      assertion1(!localPatch.isRemote(), localPatch);

      //TODO unterweg dissertation: Wenn auf dem neuen Knoten die Adjazenzinformationen auf den
      // Vertices noch nicht richtig gesetzt sind können wir nicht gleich voranschreiten.
      // U.U. brauchen wir sogar 2 Iterationen ohne Aktion... (Wegen hin- und herlaufen).
      localPatch.setSkipNextGridIteration(2);

      //TODO unterweg debug
      localPatch = Patch(localCell);
//      std::cout << "Merged during fork on rank " << tarch::parallel::Node::getInstance().getRank() << ": " << localPatch
//          << std::endl;

      assertionEquals1(localPatch.getLevel(), level, localPatch);
    }
  }

  logTraceOut( "mergeWithRemoteDataDueToForkOrJoin(...)" );
}

void peanoclaw::mappings::Remesh::prepareSendToWorker(
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
 
//  std::cout << "[prepareSendToWorker] sending to worker " << worker << ": next item @ " << fineGridVerticesEnumerator.getCellCenter() << " on level " << fineGridVerticesEnumerator.getLevel() << " size " << fineGridVerticesEnumerator.getCellSize() << " inside:" << fineGridCell.isInside() << std::endl;

  assertion4(
    peano::heap::Heap<CellDescription>::getInstance().isValidIndex(fineGridCell.getCellDescriptionIndex()),
    fineGridVerticesEnumerator.getVertexPosition(0),
    fineGridVerticesEnumerator.getCellSize(),
    fineGridCell.getCellDescriptionIndex(),
    worker
  );
//  std::cout << "Sending " << peano::heap::Heap<CellDescription>::getInstance().getData(fineGridCell.getCellDescriptionIndex()).at(0).toString() << std::endl;

  if(fineGridCell.isInside()){
    peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator communicator(worker, fineGridVerticesEnumerator.getCellCenter(), fineGridVerticesEnumerator.getLevel(), false);
    communicator.sendPatch(fineGridCell.getCellDescriptionIndex());
  }

  logTraceOut( "prepareSendToWorker(...)" );
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
  
//  std::cout << "[prepareSendToMaster] sending to master on rank " << tarch::parallel::Node::getInstance().getRank() << ": next item @ " << verticesEnumerator.getCellCenter() << " on level " << verticesEnumerator.getLevel() << std::endl;

  int toRank = tarch::parallel::NodePool::getInstance().getMasterRank();
  if(localCell.isInside()){
    peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator communicator(toRank, verticesEnumerator.getCellCenter(), verticesEnumerator.getLevel(), false);
    communicator.sendPatch(localCell.getCellDescriptionIndex());
  }

  Patch localPatch(
    localCell
  );

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
    tarch::parallel::NodePool::getInstance().getMasterRank(),
    fineGridVerticesEnumerator.getCellCenter(),
    fineGridVerticesEnumerator.getLevel(),
    false
  );

  communicator.mergeWorkerStateIntoMasterState(workerState, masterState);

  if(fineGridCell.isInside()) {
    communicator.receivePatch(fineGridCell.getCellDescriptionIndex());

    assertionEquals1(
      fineGridCell.getCellDescriptionIndex(),
      peano::heap::Heap<CellDescription>::getInstance().getData(fineGridCell.getCellDescriptionIndex()).at(0).getCellDescriptionIndex(),
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

    int temporaryCellDescriptionIndex = peano::heap::Heap<CellDescription>::getInstance().createData();
    CellDescription temporaryCellDescription;
    temporaryCellDescription.setUNewIndex(-1);
    temporaryCellDescription.setUOldIndex(-1);
    temporaryCellDescription.setAuxIndex(-1);
    temporaryCellDescription.setPosition(
      receivedVerticesEnumerator.getVertexPosition(tarch::la::Vector<DIMENSIONS, int>(0))
    );
    temporaryCellDescription.setSize(receivedVerticesEnumerator.getCellSize());
    temporaryCellDescription.setLevel(receivedVerticesEnumerator.getLevel());
    peano::heap::Heap<CellDescription>::getInstance().getData(temporaryCellDescriptionIndex).push_back(temporaryCellDescription);
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
  CellDescription& localCellDescription = peano::heap::Heap<CellDescription>::getInstance().getData(localCell.getCellDescriptionIndex()).at(0);
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
 
  //Regain 2-irregularity if necessary
  if(fineGridVertex.getRefinementControl() == peanoclaw::Vertex::Records::Refined
      || fineGridVertex.getRefinementControl() == peanoclaw::Vertex::Records::Refining) {
    tarch::la::Vector<DIMENSIONS, int> coarseGridPositionOfVertex(0);
    for(int d = 0; d < DIMENSIONS; d++) {
      if(fineGridPositionOfVertex(d) > 1) {
        coarseGridPositionOfVertex(d) = 1;
      }
    }

    peanoclaw::Vertex& coarseVertex = coarseGridVertices[coarseGridVerticesEnumerator(coarseGridPositionOfVertex)];
    if(coarseVertex.getRefinementControl() == peanoclaw::Vertex::Records::Unrefined
        && !coarseVertex.isHangingNode()) {
      coarseVertex.refine();
    }
  }

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

  #ifdef Parallel
  assertionEquals4(patch.getLevel(),
    fineGridVerticesEnumerator.getLevel(),
    patch,
    fineGridVerticesEnumerator.getVertexPosition(0),
    fineGridVerticesEnumerator.getCellSize(),
    tarch::parallel::Node::getInstance().getRank()
  );
  #endif

  _gridLevelTransfer->updatePatchStateBeforeStepDown(
    patch,
    fineGridVertices,
    fineGridVerticesEnumerator,
    _isInitializing,
    _state->isInvolvedInJoinOrFork()
  );

  _gridLevelTransfer->stepDown(
    coarseGridCell.isRoot() ? -1 : coarseGridCell.getCellDescriptionIndex(),
    patch,
    fineGridVertices,
    fineGridVerticesEnumerator
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

  _gridLevelTransfer->updatePatchStateAfterStepUp(
    finePatch,
    fineGridVertices,
    fineGridVerticesEnumerator,
    fineGridCell.isLeaf()
  );

  assertionEquals1(finePatch.isLeaf(), fineGridCell.isLeaf(), finePatch);
  assertionEquals1(finePatch.getLevel(), fineGridVerticesEnumerator.getLevel(), finePatch.toString());

  //TODO unterweg: Braucht man das wirklich nicht mehr?
//  for(int i = 0; i < TWO_POWER_D; i++) {
//    fineGridVertices[fineGridVerticesEnumerator(i)].setAdjacentCellDescriptionIndex(
//      i,
//      fineGridCell.getCellDescriptionIndex()
//    );
//  }

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
  _numerics                   = solverState.getNumerics();
  _domainOffset             = solverState.getDomainOffset();
  _domainSize               = solverState.getDomainSize();

//  if(_iterationParity == peanoclaw::records::VertexDescription::EVEN) {
//    _iterationParity = peanoclaw::records::VertexDescription::ODD;
//  } else {
//    _iterationParity = peanoclaw::records::VertexDescription::EVEN;
//  }

  _gridLevelTransfer = new peanoclaw::interSubgridCommunication::GridLevelTransfer(
                              solverState.useDimensionalSplittingOptimization(),
                              *_numerics
                           );

  _initialMinimalMeshWidth = solverState.getInitialMinimalMeshWidth();
  _isInitializing = solverState.getIsInitializing();
  _useDimensionalSplittingOptimization = solverState.useDimensionalSplittingOptimization();
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
  peano::heap::Heap<peanoclaw::records::Data>::getInstance().startToSendOrReceiveHeapData(solverState.isTraversalInverted());
  peano::heap::Heap<CellDescription>::getInstance().startToSendOrReceiveHeapData(solverState.isTraversalInverted());
  #endif

  logTraceOutWith1Argument( "beginIteration(State)", solverState);
}


void peanoclaw::mappings::Remesh::endIteration(
  peanoclaw::State&  solverState
) {
  logTraceInWith1Argument( "endIteration(State)", solverState );

  delete _gridLevelTransfer;

  logInfo("endIteration(State)", "Sent neighbor data: " << _sentNeighborData << " Received neighbor data: " << _receivedNeighborData);

  peano::heap::Heap<peanoclaw::records::Data>::getInstance().finishedToSendOrReceiveHeapData();
  peano::heap::Heap<peanoclaw::records::CellDescription>::getInstance().finishedToSendOrReceiveHeapData();

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
