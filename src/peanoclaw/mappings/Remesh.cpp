#include "peanoclaw/mappings/Remesh.h"

#include "peanoclaw/Heap.h"
#include "peanoclaw/Numerics.h"
#include "peanoclaw/ParallelSubgrid.h"
//#include "peanoclaw/grid/SubgridLevelContainer.h"
#include "peanoclaw/interSubgridCommunication/GridLevelTransfer.h"
#include "peanoclaw/parallel/NeighbourCommunicator.h"
#include "peanoclaw/parallel/MasterWorkerAndForkJoinCommunicator.h"

#include "peano/grid/aspects/VertexStateAnalysis.h"

#include "tarch/multicore/Lock.h"
#include "tarch/parallel/Node.h"

/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::CommunicationSpecification   peanoclaw::mappings::Remesh::communicationSpecification() {
  return peano::CommunicationSpecification(
            peano::CommunicationSpecification::SendDataAndStateBeforeFirstTouchVertexFirstTime,
            peano::CommunicationSpecification::SendDataAndStateAfterLastTouchVertexLastTime
         );
}

peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids::VertexMap peanoclaw::mappings::Remesh::_vertexPositionToIndexMap;
peanoclaw::parallel::NeighbourCommunicator::RemoteSubgridMap               peanoclaw::mappings::Remesh::_remoteSubgridMap;

tarch::timing::Watch peanoclaw::mappings::Remesh::_spacetreeCommunicationWaitingTimeWatch("", "", false);

/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::Remesh::touchVertexLastTimeSpecification() {
  return peano::MappingSpecification(
    peano::MappingSpecification::WholeTree,
    peano::MappingSpecification::AvoidCoarseGridRaces
  );
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::Remesh::touchVertexFirstTimeSpecification() { 
  return peano::MappingSpecification(
    peano::MappingSpecification::WholeTree,
    peano::MappingSpecification::AvoidCoarseGridRaces
  );
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::Remesh::enterCellSpecification() {
  return peano::MappingSpecification(
    peano::MappingSpecification::WholeTree,
    peano::MappingSpecification::AvoidCoarseGridRaces
  );
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::Remesh::leaveCellSpecification() {
  return peano::MappingSpecification(
    peano::MappingSpecification::WholeTree,
    peano::MappingSpecification::AvoidFineGridRaces
  );
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::Remesh::ascendSpecification() {
  return peano::MappingSpecification(
    peano::MappingSpecification::WholeTree,
    peano::MappingSpecification::AvoidCoarseGridRaces
  );
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::Remesh::descendSpecification() {
  return peano::MappingSpecification(
    peano::MappingSpecification::WholeTree,
    peano::MappingSpecification::AvoidCoarseGridRaces
  );
}

tarch::logging::Log                peanoclaw::mappings::Remesh::_log( "peanoclaw::mappings::Remesh" );

peanoclaw::mappings::Remesh::Remesh()
: _unknownsPerSubcell(-1),
  _parametersWithoutGhostlayerPerSubcell(-1),
  _parametersWithGhostlayerPerSubcell(-1),
  _defaultSubdivisionFactor(-1),
  _defaultGhostLayerWidth(-1),
  _initialTimestepSize(0.0),
  _process(),
  _numerics(0),
  _gridLevelTransfer(),
  _isInitializing(false),
  _useDimensionalSplittingExtrapolation(false),
  _parallelStatistics(""),
  _totalParallelStatistics("Simulation"),
  _state(),
  _iterationNumber(0),
  _iterationWatch("", "", false) {
  logTraceIn( "Remesh()" );

  _spacetreeCommunicationWaitingTimeWatch.stopTimer();

//  _subgridLevelContainer = new peanoclaw::grid::SubgridLevelContainer;

  logTraceOut( "Remesh()" );
}


peanoclaw::mappings::Remesh::~Remesh() {
  logTraceIn( "~Remesh()" );

  _totalParallelStatistics.logTotalStatistics();
//  delete _subgridLevelContainer;

  logTraceOut( "~Remesh()" );
}


#if defined(SharedMemoryParallelisation)
peanoclaw::mappings::Remesh::Remesh(const Remesh&  masterThread)
: _unknownsPerSubcell(masterThread._unknownsPerSubcell),
  _parametersWithoutGhostlayerPerSubcell(masterThread._parametersWithoutGhostlayerPerSubcell),
  _parametersWithGhostlayerPerSubcell(masterThread._parametersWithGhostlayerPerSubcell),
  _defaultSubdivisionFactor(masterThread._defaultSubdivisionFactor),
  _defaultGhostLayerWidth(masterThread._defaultGhostLayerWidth),
  _initialTimestepSize(masterThread._initialTimestepSize),
  _numerics(masterThread._numerics),
  _domainOffset(masterThread._domainOffset),
  _domainSize(masterThread._domainSize),
  _gridLevelTransfer(masterThread._gridLevelTransfer),
  _initialMinimalMeshWidth(masterThread._initialMinimalMeshWidth),
  _isInitializing(masterThread._isInitializing),
  _useDimensionalSplittingExtrapolation(masterThread._useDimensionalSplittingExtrapolation),
  _parallelStatistics(""),
  _totalParallelStatistics("Simulation"),
  _state(masterThread._state),
  _iterationWatch("", "", false)
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
  fineGridVertex.setAllSubcellEraseVetos();

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

  //TODO unterweg debug
  //TODO unterweg dissertation: Kein haengender Vertex muss remote-Subgitter als Nachbarn halten. Durch das Verbot,
  //  dass keine abgewinkelte Verfeinerungsgrenzen durch parallele Grenzen laufen dürfen, muss immer auch über einen
  //  persistenten Vertex ausgetauscht werden können.
  #ifdef Parallel
  for(int i = 0; i < TWO_POWER_D; i++) {
    if(vertex.getAdjacentCellDescriptionIndex(i) != -1) {
      Patch subgrid(vertex.getAdjacentCellDescriptionIndex(i));
      assertion5(!subgrid.isRemote() || subgrid.getLevel() <= coarseGridVerticesEnumerator.getLevel(), i, subgrid, vertex, _domainOffset, _domainSize);
    }
  }
  #endif

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
  fineGridVertex.setAllSubcellEraseVetos();

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
  fineGridVertex.setAllSubcellEraseVetos();

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

  //TODO unterweg debug
//  std::cout << "Creating cell at " << fineGridVerticesEnumerator.getVertexPosition() << " on level " << fineGridVerticesEnumerator.getLevel() << " on rank " << tarch::parallel::Node::getInstance().getRank() << std::endl;

  //Initialise new Patch
  Patch fineGridPatch(
    fineGridVerticesEnumerator.getVertexPosition(0),
    fineGridVerticesEnumerator.getCellSize(),
    _unknownsPerSubcell,
    _parametersWithoutGhostlayerPerSubcell,
    _parametersWithGhostlayerPerSubcell,
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

  bool isCreatingDueToForkOrJoin = peano::grid::aspects::VertexStateAnalysis::doesNoVertexCarryRefinementFlag(
                                     coarseGridVertices, coarseGridVerticesEnumerator, peanoclaw::Vertex::Records::Refining
                                   );

  //Transfer data from coarse to fine patch
  if(!coarseGridCell.isRoot() && !isCreatingDueToForkOrJoin) {
    assertion4(coarseGridCell.getCellDescriptionIndex() > -1, coarseGridCell.getCellDescriptionIndex(), fineGridVerticesEnumerator.getCellSize(), fineGridVerticesEnumerator.getLevel(), fineGridVerticesEnumerator.getVertexPosition());
    Patch coarseGridPatch(
      coarseGridCell
    );
    assertion1(tarch::la::greaterEquals(coarseGridPatch.getTimeIntervals().getTimestepSize(), 0.0) || coarseGridPatch.isVirtual(), coarseGridPatch);

    if(!_isInitializing && (coarseGridPatch.isVirtual() || coarseGridPatch.isLeaf())) {
      //TODO unterweg dissertation: The grid is skipped directly after the creation in enterCell.
      //Therefore, we need to skip at least two iterations to ensure that all ghostlayers have been set.
      fineGridPatch.setSkipNextGridIteration(2);

      fineGridPatch.getTimeIntervals().setCurrentTime(coarseGridPatch.getTimeIntervals().getCurrentTime());
      fineGridPatch.getTimeIntervals().setTimestepSize(coarseGridPatch.getTimeIntervals().getTimestepSize());
      fineGridPatch.getTimeIntervals().setEstimatedNextTimestepSize(coarseGridPatch.getTimeIntervals().getEstimatedNextTimestepSize());
      fineGridPatch.getTimeIntervals().updateMinimalNeighborTimeConstraint(
        coarseGridPatch.getTimeIntervals().getMinimalNeighborTimeConstraint(),
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
        _numerics->update(fineGridPatch);
        _numerics->interpolateSolution(
          fineGridPatch.getSubdivisionFactor(),
          0,
          coarseGridPatch,
          fineGridPatch,
          false,
          false,
          false
        );
        _numerics->interpolateSolution(
          fineGridPatch.getSubdivisionFactor(),
          0,
          coarseGridPatch,
          fineGridPatch,
          true,
          true,
          false
        );
      }

      fineGridPatch.setDemandedMeshWidth(_numerics->getDemandedMeshWidth(fineGridPatch, _isInitializing));
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

//  std::cout << "Creating cell on rank "
//      #ifdef Parallel
//      << tarch::parallel::Node::getInstance().getRank() << ": "
//      #endif
//      << fineGridVerticesEnumerator.getVertexPosition(0) << ", "
//      << fineGridVerticesEnumerator.getCellSize()
//      << ", index=" << fineGridCell.getCellDescriptionIndex()
//      << ", level=" << fineGridVerticesEnumerator.getLevel()
//      << ", _isInitializing=" << _isInitializing
//      << std::endl << fineGridPatch.toStringUNew()
//      << std::endl;

  #if defined(AssertForPositiveValues)
  assertion4(_isInitializing || !fineGridPatch.containsNonPositiveNumberInUnknownInUNew(0),
              tarch::parallel::Node::getInstance().getRank(),
              fineGridPatch,
              fineGridPatch.toStringUNew(),
              _isInitializing
  );
  #endif

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

  //TODO unterweg debug
//  std::cout << "Destroy cell at " << fineGridVerticesEnumerator.getVertexPosition() << " on level " << fineGridVerticesEnumerator.getLevel() << " on rank " << tarch::parallel::Node::getInstance().getRank()
//      << " copying: " << fineGridCell.isAssignedToRemoteRank() << std::endl;

  assertion5(
    fineGridCell.getCellDescriptionIndex() != -2,
    fineGridCell.toString(),
    fineGridVerticesEnumerator.getVertexPosition(0),
    fineGridVerticesEnumerator.getCellSize(),
    fineGridVerticesEnumerator.getLevel(),
    tarch::parallel::Node::getInstance().getRank()
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

  vertex.mergeWithNeighbor(neighbour);

  #ifdef Asserts
  for(int i = 0; i < TWO_POWER_D; i++) {
    if(vertex.getAdjacentCellDescriptionIndex(i) != -1) {
      Patch subgrid(vertex.getAdjacentCellDescriptionIndex(i));
      assertion1(!tarch::la::smaller(subgrid.getTimeIntervals().getTimestepSize(), 0.0) || !subgrid.isLeaf(), subgrid);
    }
  }
  #endif
 
  /*{
      Serialization::ReceiveBuffer& recvbuffer = peano::parallel::SerializationMap::getInstance().getReceiveBuffer(fromRank);
      assertion1(recvbuffer.isBlockAvailable(), "cannot read heap data from Serialization Buffer - not enough blocks"); 

      Serialization::Block block = recvbuffer.nextBlock();

      assertion2(block.size() == 1337, "wrong block size!", block.size())
  }*/

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
  communicator.sendSubgridsForVertex(vertex, x, h, level, *_state);
 
  logTraceOut( "prepareSendToNeighbour(...)" );
}

void peanoclaw::mappings::Remesh::prepareCopyToRemoteNode(
  peanoclaw::Vertex&  localVertex,
  int                                           toRank,
  const tarch::la::Vector<DIMENSIONS,double>&   x,
  const tarch::la::Vector<DIMENSIONS,double>&   h,
  int                                           level
) {
  logTraceInWith5Arguments( "prepareCopyToRemoteNode(...)", localVertex, toRank, x, h, level );
  // @todo Insert your code here
  logTraceOut( "prepareCopyToRemoteNode(...)" );
}

void peanoclaw::mappings::Remesh::prepareCopyToRemoteNode(
  peanoclaw::Cell&  localCell,
      int                                           toRank,
      const tarch::la::Vector<DIMENSIONS,double>&   cellCentre,
      const tarch::la::Vector<DIMENSIONS,double>&   cellSize,
      int                                           level
) {
  logTraceInWith5Arguments( "prepareCopyToRemoteNode(...)", localCell, toRank, cellCentre, cellSize, level );

  logDebug("prepareCopyToRemoteNode", "Copying data from rank " << tarch::parallel::Node::getInstance().getRank() << " to " << toRank
      << " position:" << (cellCentre - cellSize / 2.0) << ", level:" << level
      << ", isInside:" << localCell.isInside()
      << ", assignedToRemoteRank:" << localCell.isAssignedToRemoteRank()
      << ", assignedRank:" << localCell.getRankOfRemoteNode()
      << ", isRemote:" << localCell.isRemote(*_state, false, false)
      << ", sending:" << (localCell.isInside() && localCell.getRankOfRemoteNode() == toRank && !localCell.isRemote(*_state, false, false))
      << ", index:" << localCell.getCellDescriptionIndex()
      << ", valid: " << CellDescriptionHeap::getInstance().isValidIndex(localCell.getCellDescriptionIndex())
      << ", iteration:" << _iterationNumber
  );

  peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator communicator(toRank, cellCentre - cellSize * 0.5, level, true);
  communicator.sendCellDuringForkOrJoin(
    localCell,
    (cellCentre - cellSize*0.5),
    cellSize,
    *_state
  );

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

//  logInfo("mergeWithRemoteDataDueToForkOrJoin", "Merging data from rank " << fromRank << " on " << tarch::parallel::Node::getInstance().getRank()
//      << " position:" << (cellCentre - cellSize / 2.0) << ", level:" << level
//      << ", isInside(local):" << localCell.isInside()
//      << ", isInside(remote):" << masterOrWorkerCell.isInside()
//      << ", assignedToRemoteRank(local):" << localCell.isAssignedToRemoteRank()
//      << ", assignedToRemoteRank(remote):" << masterOrWorkerCell.isAssignedToRemoteRank()
//      << ", assignedRank(local):" << localCell.getRankOfRemoteNode()
//      << ", assignedRank(remote):" << masterOrWorkerCell.getRankOfRemoteNode()
//      << ", isRemote(local):" << localCell.isRemote(*_state, false, false)
//      << ", isRemote(remote):" << masterOrWorkerCell.isRemote(*_state, false, false)
//      << ", iteration:" << _iterationNumber
//  );

  assertion3(localCell.isAssignedToRemoteRank() || localCell.getCellDescriptionIndex() != -2, localCell.toString(), cellCentre, cellSize);

  peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator communicator(fromRank, cellCentre - cellSize * 0.5, level, true);
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
  bool requiresReduction = false;

  if(fineGridCell.isInside()){

    assertion4(
      CellDescriptionHeap::getInstance().isValidIndex(fineGridCell.getCellDescriptionIndex()),
      fineGridVerticesEnumerator.getVertexPosition(0),
      fineGridVerticesEnumerator.getCellSize(),
      fineGridCell.getCellDescriptionIndex(),
      worker
    );

    Patch subgrid(fineGridCell);

    peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator communicator(
      worker,
      fineGridVerticesEnumerator.getVertexPosition(0),
      fineGridVerticesEnumerator.getLevel(),
      false
    );
    communicator.sendSubgridBetweenMasterAndWorker(subgrid);

    //TODO unterweg dissertation: Subgitter m��ssen auch auf virtuell geschaltet werden, wenn sie
    //von einer Ghostlayer ��berlappt werden und mit einem Worker geshared sind.
    requiresReduction = subgrid.isVirtual();
  }

  //Change priority
  _process.setToLowPriority();

  logTraceOut( "prepareSendToWorker(...)" );
  return requiresReduction;
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

//  logInfo("prepareSendToMaster", "Sending data from rank " << tarch::parallel::Node::getInstance().getRank() << " to master " << tarch::parallel::NodePool::getInstance().getMasterRank()
//      << " position:" << (coarseGridVerticesEnumerator.getVertexPosition() + tarch::la::multiplyComponents(fineGridPositionOfCell.convertScalar<double>(), coarseGridVerticesEnumerator.getCellSize()/3.0))
//      << ", level:" << (coarseGridVerticesEnumerator.getLevel() + 1)
//      << ", isInside:" << localCell.isInside()
//      << ", assignedToRemoteRank:" << localCell.isAssignedToRemoteRank()
//      << ", assignedRank:" << localCell.getRankOfRemoteNode()
//      << ", isRemote:" << localCell.isRemote(*_state, false, false)
//      << ", sending:" << (localCell.isInside() && localCell.getRankOfRemoteNode() == tarch::parallel::NodePool::getInstance().getMasterRank() && !localCell.isRemote(*_state, false, false))
//      << ", index:" << localCell.getCellDescriptionIndex()
//      << ", valid: " << CellDescriptionHeap::getInstance().isValidIndex(localCell.getCellDescriptionIndex())
//      << ", iteration:" << _iterationNumber
//  );

  int toRank = tarch::parallel::NodePool::getInstance().getMasterRank();
  if(localCell.isInside()){
    Patch subgrid(localCell);
    peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator communicator(
      toRank,
      verticesEnumerator.getVertexPosition(0),
      verticesEnumerator.getLevel(),
      false
    );
    communicator.sendSubgridBetweenMasterAndWorker(subgrid);
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

  _process.setToNormalPriority();

  peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator communicator(
    worker,
    fineGridVerticesEnumerator.getVertexPosition(0),
    fineGridVerticesEnumerator.getLevel(),
    false
  );
 
  communicator.mergeWorkerStateIntoMasterState(workerState, masterState);

  if(fineGridCell.isInside()) {

    tarch::timing::Watch masterWorkerSubgridCommunicationWatch("", "", false);
    communicator.receivePatch(fineGridCell.getCellDescriptionIndex());
    masterWorkerSubgridCommunicationWatch.stopTimer();
    _parallelStatistics.addWaitingTimeForMasterWorkerSubgridCommunication(masterWorkerSubgridCommunicationWatch.getCalendarTime());

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
      receivedVerticesEnumerator.getVertexPosition(),
      receivedVerticesEnumerator.getLevel(),
      false
    );

    int temporaryCellDescriptionIndex = CellDescriptionHeap::getInstance().createData();
    CellDescription temporaryCellDescription;
    temporaryCellDescription.setUIndex(-1);
    temporaryCellDescription.setPosition(
      receivedVerticesEnumerator.getVertexPosition(tarch::la::Vector<DIMENSIONS, int>(0))
    );
    temporaryCellDescription.setSize(receivedVerticesEnumerator.getCellSize());

    //TODO unterweg debug
//    assertionNumericalEquals8(communicator._position, communicator._subgridCommunicator._position,
//        &communicator,
//        &(communicator._position),
//        sizeof(communicator),
//        &temporaryCellDescription,
//        &(communicator._subgridCommunicator),
//        &(communicator._subgridCommunicator._position),
//        &(communicator._subgridCommunicator._level),
//        sizeof(temporaryCellDescription));

    temporaryCellDescription.setLevel(receivedVerticesEnumerator.getLevel());
    CellDescriptionHeap::getInstance().getData(temporaryCellDescriptionIndex).push_back(temporaryCellDescription);
    receivedCell.setCellDescriptionIndex(temporaryCellDescriptionIndex);

    tarch::timing::Watch masterWorkerSubgridCommunicationWatch("", "", false);
    communicator.receivePatch(temporaryCellDescriptionIndex);
    masterWorkerSubgridCommunicationWatch.stopTimer();
    _parallelStatistics.addWaitingTimeForMasterWorkerSubgridCommunication(masterWorkerSubgridCommunicationWatch.getCalendarTime());
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

  peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids adjacentSubgrids(
    fineGridVertex,
    _vertexPositionToIndexMap,
    fineGridX,
    coarseGridVerticesEnumerator.getLevel()+1
  );
  adjacentSubgrids.checkForChangesInAdjacentRanks();
  //adjacentSubgrids.setOverlapOfRemoteGhostlayers();

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

  //TODO unterweg debug
//  for(int i = 0; i < TWO_POWER_D; i++) {
//    assertionEquals(fineGridVertex.getAdjacentRanks()(i), 0);
//  }

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

  fineGridVertex.increaseAgeInGridIterations();
 
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

  assertion6(CellDescriptionHeap::getInstance().isValidIndex(fineGridCell.getCellDescriptionIndex()),
    fineGridCell.getCellDescriptionIndex(),
    fineGridVerticesEnumerator.getVertexPosition(0),
    fineGridVerticesEnumerator.getCellSize(),
    fineGridVerticesEnumerator.getLevel(),
    fineGridCell.toString(),
    _iterationNumber
  );

  //Prepare subgrid on first level
//  if(coarseGridCell.isRoot()) {
//    _subgridLevelContainer->addFirstLevel(fineGridCell, fineGridVertices, fineGridVerticesEnumerator);
//  }

//  Patch& subgrid = fineGridCell.getSubgrid();
  Patch subgrid(fineGridCell);

  assertion(subgrid.isLeaf() || !subgrid.isLeaf());

  #ifdef Parallel
  if(!_isInitializing) {
    fineGridCell.setCellIsAForkCandidate(true);
  }

  assertionEquals4(subgrid.getLevel(),
    fineGridVerticesEnumerator.getLevel(),
    subgrid,
    fineGridVerticesEnumerator.getVertexPosition(0),
    fineGridVerticesEnumerator.getCellSize(),
    tarch::parallel::Node::getInstance().getRank()
  );
  #endif

  Patch coarseSubgrid;
  if(!coarseGridCell.isRoot()) {
    coarseSubgrid = Patch(coarseGridCell);
  }

  _gridLevelTransfer->stepDown(
    //coarseGridCell.isRoot() ? 0 : &(coarseGridCell.getSubgrid()),
    coarseGridCell.isRoot() ? 0 : &coarseSubgrid,
    subgrid,
    fineGridVertices,
    fineGridVerticesEnumerator,
    _isInitializing,
    fineGridCell.isLeaf()
  );

  #ifdef Asserts
  if(subgrid.isLeaf() && !fineGridCell.isLeaf()) {
    bool isRefining = false;
    for(int i = 0; i < TWO_POWER_D; i++) {
      if(fineGridVertices[fineGridVerticesEnumerator(i)].getRefinementControl() == peanoclaw::records::Vertex::Refining) {
        isRefining = true;
      }
    }
    assertion6(isRefining, subgrid, fineGridCell,
        fineGridVertices[fineGridVerticesEnumerator(0)],
        fineGridVertices[fineGridVerticesEnumerator(1)],
        fineGridVertices[fineGridVerticesEnumerator(2)],
        fineGridVertices[fineGridVerticesEnumerator(3)]);
  }
  #endif
  logTraceOutWith2Arguments( "enterCell(...)", fineGridCell, subgrid );
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

  Patch fineSubgrid(
    fineGridCell
  );
//  Patch& fineSubgrid = fineGridCell.getSubgrid();
  ParallelSubgrid fineParallelSubgrid(
    fineGridCell
  );

  Patch coarseSubgrid;
  if(coarseGridCell.holdsSubgrid()) {
    coarseSubgrid = Patch(coarseGridCell);
  }

  _gridLevelTransfer->stepUp(
    //coarseGridCell.getCellDescriptionIndex(),
    //coarseGridCell.holdsSubgrid() ? &coarseGridCell.getSubgrid() : 0,
    coarseGridCell.holdsSubgrid() ? &coarseSubgrid : 0,
    fineSubgrid,
    fineParallelSubgrid,
    fineGridCell.isLeaf(),
    fineGridVertices,
    fineGridVerticesEnumerator
  );

  assertionEquals1(fineSubgrid.isLeaf(), fineGridCell.isLeaf(), fineSubgrid);
  assertionEquals1(fineSubgrid.getLevel(), fineGridVerticesEnumerator.getLevel(), fineSubgrid.toString());

  if(!fineGridCell.isAssignedToRemoteRank()) {
    //Count number of adjacent subgrids
    ParallelSubgrid parallelSubgrid(fineGridCell.getCellDescriptionIndex());
    parallelSubgrid.countNumberOfAdjacentParallelSubgrids(
      fineGridVertices,
      fineGridVerticesEnumerator
    );

    //Set remote overlaps
    for(int i = 0; i < TWO_POWER_D; i++) {
      peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids adjacentSubgrids(
        fineGridVertices[fineGridVerticesEnumerator(i)],
        _vertexPositionToIndexMap,
        fineGridVerticesEnumerator.getVertexPosition(i),
        fineGridVerticesEnumerator.getLevel()
      );

      adjacentSubgrids.setOverlapOfRemoteGhostlayers(i);
    }
  }

  //Avoid erasing when one coarse grid vertex is refining.
  if(peano::grid::aspects::VertexStateAnalysis::doesOneVertexCarryRefinementFlag(
       coarseGridVertices,
       coarseGridVerticesEnumerator,
       peanoclaw::Vertex::Records::RefinementTriggered
     )
     ||
     peano::grid::aspects::VertexStateAnalysis::doesOneVertexCarryRefinementFlag(
       coarseGridVertices,
       coarseGridVerticesEnumerator,
       peanoclaw::Vertex::Records::Refining
     )) {
    for(int i = 0; i < TWO_POWER_D; i++) {
      coarseGridVertices[coarseGridVerticesEnumerator(i)].setSubcellEraseVeto(i);
    }
  }

  //Remove subgrid on first level
//  if(coarseGridCell.isRoot()) {
//    _subgridLevelContainer->removeFirstLevel();
//  }

  logTraceOutWith1Argument( "leaveCell(...)", fineGridCell );
}


void peanoclaw::mappings::Remesh::beginIteration(
  peanoclaw::State&  solverState
) {
  logTraceInWith1Argument( "beginIteration(State)", solverState );

  _spacetreeCommunicationWaitingTimeWatch.stopTimer();
  _parallelStatistics = peanoclaw::statistics::ParallelStatistics("Iteration");
  if(_iterationNumber > 0) {
    _parallelStatistics.addWaitingTimeForMasterWorkerSpacetreeCommunication(_spacetreeCommunicationWaitingTimeWatch.getCalendarTime());
  }

  //TODO unterweg debug
//  #ifdef Parallel
//  if(solverState.isJoinWithMasterTriggered()) {
//    logInfo("beginIteration", "Join triggered: "
//        << tarch::parallel::Node::getInstance().getRank() << "+" << tarch::parallel::NodePool::getInstance().getMasterRank()
//        << "->" << tarch::parallel::NodePool::getInstance().getMasterRank());
//  }
//  if(solverState.isJoinWithMasterTriggered()) {
//    logInfo("beginIteration", "Joining: "
//        << tarch::parallel::Node::getInstance().getRank() << "+" << tarch::parallel::NodePool::getInstance().getMasterRank()
//        << "->" << tarch::parallel::NodePool::getInstance().getMasterRank());
//  }
//  #endif

  _iterationNumber++;

  //TODO unterweg debug
//  logInfo("beginIteration", "Beginning Iteration " << _iterationNumber);

  _unknownsPerSubcell       = solverState.getUnknownsPerSubcell();
  _parametersWithoutGhostlayerPerSubcell = solverState.getNumberOfParametersWithoutGhostlayerPerSubcell();
  _parametersWithGhostlayerPerSubcell = solverState.getNumberOfParametersWithGhostlayerPerSubcell();
  _defaultSubdivisionFactor = solverState.getDefaultSubdivisionFactor();
  _defaultGhostLayerWidth   = solverState.getDefaultGhostLayerWidth();
  _initialTimestepSize      = solverState.getInitialTimestepSize();
  _numerics                 = solverState.getNumerics();
  _domainOffset             = solverState.getDomainOffset();
  _domainSize               = solverState.getDomainSize();

  _initialMinimalMeshWidth = solverState.getInitialMaximalSubgridSize();
  _isInitializing = solverState.getIsInitializing();
  _useDimensionalSplittingExtrapolation = solverState.useDimensionalSplittingExtrapolation();
  _state = &solverState;

  _gridLevelTransfer = new peanoclaw::interSubgridCommunication::GridLevelTransfer(
                              _useDimensionalSplittingExtrapolation,
                              *_numerics
                           );

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
  tarch::timing::Watch neighborSubgridCommunicationWatch("", "", false);
  DataHeap::getInstance().startToSendSynchronousData();
  DataHeap::getInstance().startToSendBoundaryData(solverState.isTraversalInverted());
  CellDescriptionHeap::getInstance().startToSendSynchronousData();
  CellDescriptionHeap::getInstance().startToSendBoundaryData(solverState.isTraversalInverted());
  neighborSubgridCommunicationWatch.stopTimer();
  _parallelStatistics.addWaitingTimeForNeighborSubgridCommunication(neighborSubgridCommunicationWatch.getCalendarTime());

//  if(tarch::parallel::Node::getInstance().isGlobalMaster()) {
//    solverState.resetLocalHeightOfWorkerTree();
//
//    logDebug("beginIteration(State)", "Height of worker tree in last grid iteration was " << solverState.getGlobalHeightOfWorkerTreeDuringLastIteration());
//  } else {
//    solverState.increaseLocalHeightOfWorkerTree();
//  }
  #endif

  _iterationWatch.startTimer();
  logTraceOutWith1Argument( "beginIteration(State)", solverState);
}


void peanoclaw::mappings::Remesh::endIteration(
  peanoclaw::State&  solverState
) {
  logTraceInWith1Argument( "endIteration(State)", solverState );
  _iterationWatch.stopTimer();
  logInfo("logStatistics()", "Waiting time for iteration: "
      << _iterationWatch.getCalendarTime() << " (total), "
      << _iterationWatch.getCalendarTime() << " (average) "
      << 1 << " samples");

  delete _gridLevelTransfer;

  _parallelStatistics.logIterationStatistics();
  _totalParallelStatistics.merge(_parallelStatistics);

  DataHeap::getInstance().finishedToSendBoundaryData(solverState.isTraversalInverted());
  CellDescriptionHeap::getInstance().finishedToSendBoundaryData(solverState.isTraversalInverted());

  #ifdef Parallel
  if(tarch::parallel::Node::getInstance().isGlobalMaster() || _state->isJoiningWithMaster()) {
    CellDescriptionHeap::getInstance().finishedToSendSynchronousData();
    DataHeap::getInstance().finishedToSendSynchronousData();
  }
  #endif

  _spacetreeCommunicationWaitingTimeWatch.startTimer();
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

//  _subgridLevelContainer->addNewLevel(
//    fineGridCells,
//    fineGridVertices,
//    fineGridVerticesEnumerator
//  );
//
//  #ifdef Asserts
//  for(int i = 0; i < FOUR_POWER_D; i++) {
//    for(int j = 0; j < TWO_POWER_D; j++) {
//      int adjacentCellDescriptionIndex = fineGridVertices[fineGridVerticesEnumerator(i)].getAdjacentCellDescriptionIndex(j);
//      assertion3(
//        adjacentCellDescriptionIndex == -1
//          || adjacentCellDescriptionIndex == fineGridVertices[fineGridVerticesEnumerator(i)].getAdjacentSubgrid(j).getCellDescriptionIndex(),
//        i,
//        j,
//        adjacentCellDescriptionIndex
//      );
//    }
//  }
//  #endif

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

  //TODO unterweg dissertation
  //Oscillation warning: If a subgrid got refined the new fine subgrids get annotated with a
  //certain demanded mesh width by the application. If this is so large for all $2^d$ fine
  //subgrids that they could be immediately coarsened again but the coarse subgrid has a
  //demanded mesh width so it has to be refined oscillating refinement may occur.
  if(coarseGridCell.getCellDescriptionIndex() > -1) {
    //Patch& coarseSubgrid = coarseGridCell.getSubgrid();
    Patch coarseSubgrid(coarseGridCell);
    bool printOscillationWarning = tarch::la::oneGreater(coarseSubgrid.getSubcellSize(), coarseSubgrid.getDemandedMeshWidth());
    //for(int i = 0; i < THREE_POWER_D; i++) {
    dfor3(cellIndex)
      peanoclaw::Cell& fineCell = fineGridCells[fineGridVerticesEnumerator.cell(cellIndex)];
      if(fineCell.getCellDescriptionIndex() > -1) {
        //Patch& fineSubgrid = fineCell.getSubgrid();
        Patch fineSubgrid(fineCell);

        //Only relevant if all fine subgrids have just been created.
        if(fineSubgrid.getAge() > 1) {
          printOscillationWarning = false;
          break;
        }

        if(!tarch::la::allGreaterEquals(fineSubgrid.getDemandedMeshWidth(), fineSubgrid.getSubcellSize() * 3.0)) {
          printOscillationWarning = false;
          break;
        }
      }
    enddforx
    if(printOscillationWarning) {
      logWarning("ascend(...)", "Oscillating refinement may occur. Check refinement criterion... Coarse subgrid: " << coarseSubgrid);
    }
  }

  //Remove level from SubgridLevelContainer
//  _subgridLevelContainer->removeCurrentLevel();

  logTraceOut( "ascend(...)" );
}
