#include "peanoclaw/mappings/ValidateGrid.h"

#include "peanoclaw/Patch.h"

#include "peano/utils/Loop.h"

/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::ValidateGrid::touchVertexLastTimeSpecification() {
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::AvoidFineGridRaces,false);
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::ValidateGrid::touchVertexFirstTimeSpecification() { 
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::AvoidFineGridRaces,false);
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::ValidateGrid::enterCellSpecification() {
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::AvoidFineGridRaces,false);
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::ValidateGrid::leaveCellSpecification() {
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::AvoidFineGridRaces,false);
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::ValidateGrid::ascendSpecification() {
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::AvoidFineGridRaces,false);
}


/**
 * @todo Please tailor the parameters to your mapping's properties.
 */
peano::MappingSpecification   peanoclaw::mappings::ValidateGrid::descendSpecification() {
  return peano::MappingSpecification(peano::MappingSpecification::WholeTree,peano::MappingSpecification::AvoidFineGridRaces,false);
}


tarch::logging::Log                peanoclaw::mappings::ValidateGrid::_log( "peanoclaw::mappings::ValidateGrid" ); 

peanoclaw::mappings::ValidateGrid::ValidateGrid()
 : _heap(peano::heap::Heap<PatchDescription>::getInstance()),
   _validator(0, 0, false)
{
  logTraceIn( "ValidateGrid()" );

  _patchDescriptionsIndex = peano::heap::Heap<PatchDescription>::getInstance().createData();

  logTraceOut( "ValidateGrid()" );
}


peanoclaw::mappings::ValidateGrid::~ValidateGrid() {
  logTraceIn( "~ValidateGrid()" );
  // @todo Insert your code here
  logTraceOut( "~ValidateGrid()" );
}


#if defined(SharedMemoryParallelisation)
peanoclaw::mappings::ValidateGrid::ValidateGrid(const ValidateGrid&  masterThread) {
  logTraceIn( "ValidateGrid(ValidateGrid)" );
  // @todo Insert your code here
  logTraceOut( "ValidateGrid(ValidateGrid)" );
}


void peanoclaw::mappings::ValidateGrid::mergeWithWorkerThread(const ValidateGrid& workerThread) {
  logTraceIn( "mergeWithWorkerThread(ValidateGrid)" );
  // @todo Insert your code here
  logTraceOut( "mergeWithWorkerThread(ValidateGrid)" );
}
#endif


void peanoclaw::mappings::ValidateGrid::createHangingVertex(
      peanoclaw::Vertex&     fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                fineGridH,
      peanoclaw::Vertex * const   coarseGridVertices,
      const peano::grid::VertexEnumerator&      coarseGridVerticesEnumerator,
      peanoclaw::Cell&       coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                   fineGridPositionOfVertex
) {
  logTraceInWith6Arguments( "createHangingVertex(...)", fineGridVertex, fineGridX, fineGridH, coarseGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfVertex );

  //TODO unterweg debug
//  std::cout<< "Create hanging vertex " << fineGridX << ", "
//      << fineGridH
//      #ifdef Parallel
//      << ", rank=" << tarch::parallel::Node::getInstance().getRank()
//      #endif
//      << std::endl;

  logTraceOutWith1Argument( "createHangingVertex(...)", fineGridVertex );
}


void peanoclaw::mappings::ValidateGrid::destroyHangingVertex(
      const peanoclaw::Vertex&   fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridH,
      peanoclaw::Vertex * const  coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&           coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfVertex
) {
  logTraceInWith6Arguments( "destroyHangingVertex(...)", fineGridVertex, fineGridX, fineGridH, coarseGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfVertex );

  //TODO unterweg debug
//  std::cout<< "Destroy hanging vertex " << fineGridX << ", "
//       << fineGridH
//        << std::endl;

  if(tarch::la::allGreaterEquals(fineGridX, _domainOffset)
    && tarch::la::allGreaterEquals(_domainOffset+_domainSize, fineGridX)) {
    _validator.findAdjacentPatches(
      fineGridVertex,
      fineGridX,
      coarseGridVerticesEnumerator.getLevel() + 1,
      #ifdef Parallel
      tarch::parallel::Node::getInstance().getRank()
      #else
      0
      #endif
    );
  }

  logTraceOutWith1Argument( "destroyHangingVertex(...)", fineGridVertex );
}


void peanoclaw::mappings::ValidateGrid::createInnerVertex(
      peanoclaw::Vertex&               fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridH,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfVertex
) {
  logTraceInWith6Arguments( "createInnerVertex(...)", fineGridVertex, fineGridX, fineGridH, coarseGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfVertex );

  //TODO unterweg debug
//  std::cout<< "Create inner vertex " << fineGridX << ", "
//      << fineGridH
//      #ifdef Parallel
//      << ", rank=" << tarch::parallel::Node::getInstance().getRank()
//      #endif
//      << std::endl;

  logTraceOutWith1Argument( "createInnerVertex(...)", fineGridVertex );
}


void peanoclaw::mappings::ValidateGrid::createBoundaryVertex(
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


void peanoclaw::mappings::ValidateGrid::destroyVertex(
      const peanoclaw::Vertex&   fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridH,
      peanoclaw::Vertex * const  coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&           coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfVertex
) {
  logTraceInWith6Arguments( "destroyVertex(...)", fineGridVertex, fineGridX, fineGridH, coarseGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfVertex );

  //TODO unterweg debug
//  std::cout<< "Destroying vertex " << fineGridX << ", "
//      << fineGridH
//      << ", level=" << (coarseGridVerticesEnumerator.getLevel()+1)
//      #ifdef Parallel
//      << ", rank=" << tarch::parallel::Node::getInstance().getRank()
//      #endif
//      << std::endl;

  _validator.deleteNonRemoteAdjacentPatches(
    fineGridVertex,
    fineGridX,
    coarseGridVerticesEnumerator.getLevel() + 1,
    #ifdef Parallel
    tarch::parallel::Node::getInstance().getRank()
    #else
    0
    #endif
  );

  logTraceOutWith1Argument( "destroyVertex(...)", fineGridVertex );
}


void peanoclaw::mappings::ValidateGrid::createCell(
      peanoclaw::Cell&                 fineGridCell,
      peanoclaw::Vertex * const        fineGridVertices,
      const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell
) {
  logTraceInWith4Arguments( "createCell(...)", fineGridCell, fineGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfCell );

  //TODO unterweg debug
//  std::cout<< "Creating cell " << fineGridVerticesEnumerator.getVertexPosition(0) << ", "
//      << fineGridVerticesEnumerator.getCellSize()
//      << ", index=" << fineGridCell.getCellDescriptionIndex()
//      #ifdef Parallel
//      << ", rank=" << tarch::parallel::Node::getInstance().getRank()
//      #endif
//      << std::endl;

  logTraceOutWith1Argument( "createCell(...)", fineGridCell );
}


void peanoclaw::mappings::ValidateGrid::destroyCell(
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
//  std::cout<< "Destroying cell " << fineGridVerticesEnumerator.getVertexPosition(0) << ", "
//      << fineGridVerticesEnumerator.getCellSize()
//      << ", index=" << fineGridCell.getCellDescriptionIndex()
//      #ifdef Parallel
//      << ", rank=" << tarch::parallel::Node::getInstance().getRank()
//      #endif
//      << std::endl;

  logTraceOutWith1Argument( "destroyCell(...)", fineGridCell );
}

#ifdef Parallel
void peanoclaw::mappings::ValidateGrid::mergeWithNeighbour(
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

void peanoclaw::mappings::ValidateGrid::prepareSendToNeighbour(
  peanoclaw::Vertex&  vertex,
      int                                           toRank,
      const tarch::la::Vector<DIMENSIONS,double>&   x,
      const tarch::la::Vector<DIMENSIONS,double>&   h,
      int                                           level
) {
  logTraceInWith3Arguments( "prepareSendToNeighbour(...)", vertex, toRank, level );

  logTraceOut( "prepareSendToNeighbour(...)" );
}

void peanoclaw::mappings::ValidateGrid::prepareCopyToRemoteNode(
  peanoclaw::Vertex&  localVertex,
      int                                           toRank,
      const tarch::la::Vector<DIMENSIONS,double>&   x,
      const tarch::la::Vector<DIMENSIONS,double>&   h,
      int                                           level
) {
  logTraceInWith5Arguments( "prepareCopyToRemoteNode(...)", localVertex, toRank, x, h, level );

  _validator.findAdjacentPatches(
    localVertex,
    x,
    level,
    toRank
  );

  logTraceOut( "prepareCopyToRemoteNode(...)" );
}

void peanoclaw::mappings::ValidateGrid::prepareCopyToRemoteNode(
  peanoclaw::Cell&  localCell,
      int                                           toRank,
      const tarch::la::Vector<DIMENSIONS,double>&   cellCentre,
      const tarch::la::Vector<DIMENSIONS,double>&   cellSize,
      int                                           level
) {
  logTraceInWith5Arguments( "prepareCopyToRemoteNode(...)", localCell, toRank, cellCentre, cellSize, level );

  //TODO unterweg debug
//  std::cout<< "Copying cell " << (cellCentre-0.5*cellSize) << ", "
//      << cellSize
//      << ", index=" << localCell.getCellDescriptionIndex()
//      #ifdef Parallel
//      << ", from=" << tarch::parallel::Node::getInstance().getRank()
//      << ", to=" << toRank
//      #endif
//      << std::endl;

  logTraceOut( "prepareCopyToRemoteNode(...)" );
}

void peanoclaw::mappings::ValidateGrid::mergeWithRemoteDataDueToForkOrJoin(
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

void peanoclaw::mappings::ValidateGrid::mergeWithRemoteDataDueToForkOrJoin(
  peanoclaw::Cell&  localCell,
  const peanoclaw::Cell&  masterOrWorkerCell,
  int                                       fromRank,
  const tarch::la::Vector<DIMENSIONS,double>&  cellCentre,
  const tarch::la::Vector<DIMENSIONS,double>&  cellSize,
  int                                       level
) {
  logTraceInWith3Arguments( "mergeWithRemoteDataDueToForkOrJoin(...)", localCell, masterOrWorkerCell, fromRank );
  // @todo Insert your code here
  logTraceOut( "mergeWithRemoteDataDueToForkOrJoin(...)" );
}

void peanoclaw::mappings::ValidateGrid::prepareSendToWorker(
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
  // @todo Insert your code here
  logTraceOut( "prepareSendToWorker(...)" );
}

void peanoclaw::mappings::ValidateGrid::prepareSendToMaster(
  peanoclaw::Cell&                       localCell,
  peanoclaw::Vertex *                    vertices,
  const peano::grid::VertexEnumerator&       verticesEnumerator, 
  const peanoclaw::Vertex * const        coarseGridVertices,
  const peano::grid::VertexEnumerator&       coarseGridVerticesEnumerator,
  const peanoclaw::Cell&                 coarseGridCell,
  const tarch::la::Vector<DIMENSIONS,int>&   fineGridPositionOfCell
) {
  logTraceInWith2Arguments( "prepareSendToMaster(...)", localCell, verticesEnumerator.toString() );

  //Assemble vector to send to master
  std::vector<PatchDescription>& workerAndLocalData = _heap.getData(_patchDescriptionsIndex);
  std::vector<PatchDescription> localData = _validator.getAllPatches();
  for(int i = 0; i < localData.size(); i++) {
    workerAndLocalData.push_back(localData[i]);
  }

  //Add non-referenced patches
  int index = 0;
  int numberOfEntries = 0;
  while(numberOfEntries < peano::heap::Heap<CellDescription>::getInstance().getNumberOfAllocatedEntries()) {
    if(peano::heap::Heap<CellDescription>::getInstance().isValidIndex(index)) {
//      if(_descriptions.find(index) == _descriptions.end()) {
//        //Found non-referenced patch
//        CellDescription& cellDescription = peano::heap::Heap<CellDescription>::getInstance().getData(index).at(0);
//        PatchDescription patchDescription;
//        patchDescription.setPosition(cellDescription.getPosition());
//        patchDescription.setSize(cellDescription.getSize());
//        patchDescription.setLevel(cellDescription.getLevel());
//        patchDescription.setIsRemote(cellDescription.getIsRemote());
//        patchDescription.setIsReferenced(false);
//        patchDescription.setCellDescriptionIndex(index);
//        patchDescription.setRank(tarch::parallel::Node::getInstance().getRank());
//        descriptionVector.push_back(patchDescription);
//      }
      numberOfEntries++;
    }
    index++;
  }

  //Send
  _heap.sendData(
    _patchDescriptionsIndex,
    tarch::parallel::NodePool::getInstance().getMasterRank(),
    verticesEnumerator.getVertexPosition(0),
    verticesEnumerator.getLevel(),
    peano::heap::MasterWorkerCommunication
  );

  //TODO unterweg debug
  std::cout << "ValidateGrid Sent " << _heap.getData(_patchDescriptionsIndex).size() << " Data on rank " << tarch::parallel::Node::getInstance().getRank() << std::endl;

  logTraceOut( "prepareSendToMaster(...)" );
}


void peanoclaw::mappings::ValidateGrid::mergeWithMaster(
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

  std::vector<PatchDescription> remoteData = _heap.receiveData(
    worker,
    fineGridVerticesEnumerator.getVertexPosition(0),
    fineGridVerticesEnumerator.getLevel(),
    peano::heap::MasterWorkerCommunication
  );

  std::vector<PatchDescription>& localData = _heap.getData(_patchDescriptionsIndex);
  for(int i = 0; i < (int)remoteData.size(); i++) {
    localData.push_back(remoteData[i]);
  }

  logInfo("mergeWithMaster(...)", "Merged " << remoteData.size() << " patches: " << _heap.getData(_patchDescriptionsIndex).size() << " entries.");

  logTraceOut( "mergeWithMaster(...)" );
}


void peanoclaw::mappings::ValidateGrid::receiveDataFromMaster(
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
  logTraceIn( "receiveDataFromMaster(...)" );
  // @todo Insert your code here
  logTraceOut( "receiveDataFromMaster(...)" );
}


void peanoclaw::mappings::ValidateGrid::mergeWithWorker(
  peanoclaw::Cell&           localCell, 
  const peanoclaw::Cell&     receivedMasterCell,
  const tarch::la::Vector<DIMENSIONS,double>&  cellCentre,
  const tarch::la::Vector<DIMENSIONS,double>&  cellSize,
  int                                          level
) {
  logTraceInWith2Arguments( "mergeWithWorker(...)", localCell.toString(), receivedMasterCell.toString() );
  // @todo Insert your code here

  std::cout << "ValidateGrid Receive on rank " << tarch::parallel::Node::getInstance().getRank() << std::endl;

  logTraceOutWith1Argument( "mergeWithWorker(...)", localCell.toString() );
}


void peanoclaw::mappings::ValidateGrid::mergeWithWorker(
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

void peanoclaw::mappings::ValidateGrid::touchVertexFirstTime(
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


void peanoclaw::mappings::ValidateGrid::touchVertexLastTime(
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
//  std::cout<< "Touching vertex " << fineGridX << ", "
//      << fineGridH
//      #ifdef Parallel
//      << ", rank=" << tarch::parallel::Node::getInstance().getRank()
//      #endif
//      << std::endl;

  _validator.findAdjacentPatches(
    fineGridVertex,
    fineGridX,
    coarseGridVerticesEnumerator.getLevel() + 1,
    #ifdef Parallel
    tarch::parallel::Node::getInstance().getRank()
    #else
    0
    #endif
  );

  logTraceOutWith1Argument( "touchVertexLastTime(...)", fineGridVertex );
}


void peanoclaw::mappings::ValidateGrid::enterCell(
      peanoclaw::Cell&                 fineGridCell,
      peanoclaw::Vertex * const        fineGridVertices,
      const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell
) {
  logTraceInWith4Arguments( "enterCell(...)", fineGridCell, fineGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfCell );
  // @todo Insert your code here
  logTraceOutWith1Argument( "enterCell(...)", fineGridCell );
}


void peanoclaw::mappings::ValidateGrid::leaveCell(
      peanoclaw::Cell&           fineGridCell,
      peanoclaw::Vertex * const  fineGridVertices,
      const peano::grid::VertexEnumerator&          fineGridVerticesEnumerator,
      peanoclaw::Vertex * const  coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&           coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfCell
) {
  logTraceInWith4Arguments( "leaveCell(...)", fineGridCell, fineGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfCell );
  // @todo Insert your code here
  logTraceOutWith1Argument( "leaveCell(...)", fineGridCell );
}


void peanoclaw::mappings::ValidateGrid::beginIteration(
  peanoclaw::State&  solverState
) {
  logTraceInWith1Argument( "beginIteration(State)", solverState );

  _validator = peanoclaw::statistics::ParallelGridValidator(
    solverState.getDomainOffset(),
    solverState.getDomainSize(),
    solverState.useDimensionalSplittingOptimization()
  );
  assertionEquals(_validator.getAllPatches().size(), 0);
  _domainOffset = solverState.getDomainOffset();
  _domainSize = solverState.getDomainSize();
  _heap.getData(_patchDescriptionsIndex).clear();

  logTraceOutWith1Argument( "beginIteration(State)", solverState);
}


void peanoclaw::mappings::ValidateGrid::endIteration(
  peanoclaw::State&  solverState
) {
  logTraceInWith1Argument( "endIteration(State)", solverState );

  if(
    #ifdef Parallel
    tarch::parallel::Node::getInstance().isGlobalMaster()
    #else
    true
    #endif
  ) {

    //TODO unterweg debug
    std::cout << "END ITERATION!"
         << std::endl;

    //Copy collected patches
    std::vector<PatchDescription>& receivedDescriptions = _heap.getData(_patchDescriptionsIndex);
    logInfo("endIteration(State)", "Received patches: " << receivedDescriptions.size());
    std::vector<PatchDescription> localDescriptions = _validator.getAllPatches();
    for(std::vector<PatchDescription>::iterator i = receivedDescriptions.begin();
        i != receivedDescriptions.end();
        i++) {
      localDescriptions.push_back(*i);
    }
    logInfo("endIteration(State)", "Total patches: " << localDescriptions.size());

    //Construct database
    peanoclaw::statistics::PatchDescriptionDatabase database;
    for(int i = 0; i < (int)localDescriptions.size(); i++) {
      database.insertPatch(localDescriptions[i]);
    }

    //Check for unreferenced patches
    for(int i = 0; i < (int)localDescriptions.size(); i++) {
      if(!localDescriptions[i].getIsReferenced()) {
        logError("endIteration", "Unreferenced Patch found: " << localDescriptions[i].toString() <<
            ", isUnreferenced=" << database.containsPatch(localDescriptions[i].getPosition(), localDescriptions[i].getLevel(), localDescriptions[i].getRank()));
        assertionFail("");
      }
    }

    _validator.validatePatches();

    logInfo("endIteration", "Validated " << localDescriptions.size() << " subgrids.");
  }

  logTraceOutWith1Argument( "endIteration(State)", solverState);
}



void peanoclaw::mappings::ValidateGrid::descend(
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


void peanoclaw::mappings::ValidateGrid::ascend(
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
