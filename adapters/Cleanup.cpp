#include "adapters/Cleanup.h"



peano::MappingSpecification   peanoclaw::adapters::Cleanup::touchVertexLastTimeSpecification() {
  return peano::MappingSpecification::getMostGeneralSpecification()
   & peanoclaw::mappings::Cleanup::touchVertexLastTimeSpecification()


;
}


peano::MappingSpecification   peanoclaw::adapters::Cleanup::touchVertexFirstTimeSpecification() { 
  return peano::MappingSpecification::getMostGeneralSpecification()
   & peanoclaw::mappings::Cleanup::touchVertexFirstTimeSpecification()


;
}


peano::MappingSpecification   peanoclaw::adapters::Cleanup::enterCellSpecification() {
  return peano::MappingSpecification::getMostGeneralSpecification()
   & peanoclaw::mappings::Cleanup::enterCellSpecification()


;
}


peano::MappingSpecification   peanoclaw::adapters::Cleanup::leaveCellSpecification() {
  return peano::MappingSpecification::getMostGeneralSpecification()
   & peanoclaw::mappings::Cleanup::leaveCellSpecification()


;
}


peano::MappingSpecification   peanoclaw::adapters::Cleanup::ascendSpecification() {
  return peano::MappingSpecification::getMostGeneralSpecification()
   & peanoclaw::mappings::Cleanup::ascendSpecification()


;
}


peano::MappingSpecification   peanoclaw::adapters::Cleanup::descendSpecification() {
  return peano::MappingSpecification::getMostGeneralSpecification()
   & peanoclaw::mappings::Cleanup::descendSpecification()


;
}


peanoclaw::adapters::Cleanup::Cleanup() {
}


peanoclaw::adapters::Cleanup::~Cleanup() {
}


#if defined(SharedMemoryParallelisation)
peanoclaw::adapters::Cleanup::Cleanup(const Cleanup&  masterThread):
  _map2Cleanup(masterThread._map2Cleanup) 

 

{
}


void peanoclaw::adapters::Cleanup::mergeWithWorkerThread(const Cleanup& workerThread) {

  _map2Cleanup.mergeWithWorkerThread(workerThread._map2Cleanup);


}
#endif


void peanoclaw::adapters::Cleanup::createHangingVertex(
      peanoclaw::Vertex&     fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                fineGridH,
      peanoclaw::Vertex * const   coarseGridVertices,
      const peano::grid::VertexEnumerator&      coarseGridVerticesEnumerator,
      peanoclaw::Cell&       coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                   fineGridPositionOfVertex
) {

  _map2Cleanup.createHangingVertex(fineGridVertex, fineGridX, fineGridH, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfVertex );


}


void peanoclaw::adapters::Cleanup::destroyHangingVertex(
      const peanoclaw::Vertex&   fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridH,
      peanoclaw::Vertex * const  coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&           coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfVertex
) {

  _map2Cleanup.destroyHangingVertex(fineGridVertex, fineGridX, fineGridH, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfVertex );


}


void peanoclaw::adapters::Cleanup::createInnerVertex(
      peanoclaw::Vertex&               fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridH,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfVertex
) {

  _map2Cleanup.createInnerVertex(fineGridVertex, fineGridX, fineGridH, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfVertex );


}


void peanoclaw::adapters::Cleanup::createBoundaryVertex(
      peanoclaw::Vertex&               fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridH,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfVertex
) {

  _map2Cleanup.createBoundaryVertex( fineGridVertex, fineGridX, fineGridH, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfVertex );


}


void peanoclaw::adapters::Cleanup::destroyVertex(
      const peanoclaw::Vertex&   fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridH,
      peanoclaw::Vertex * const  coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&           coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfVertex
) {

  _map2Cleanup.destroyVertex( fineGridVertex, fineGridX, fineGridH, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfVertex );


}


void peanoclaw::adapters::Cleanup::createCell(
      peanoclaw::Cell&                 fineGridCell,
      peanoclaw::Vertex * const        fineGridVertices,
      const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell
) {

  _map2Cleanup.createCell( fineGridCell, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfCell );


}


void peanoclaw::adapters::Cleanup::destroyCell(
      const peanoclaw::Cell&           fineGridCell,
      peanoclaw::Vertex * const        fineGridVertices,
      const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell
) {

  _map2Cleanup.destroyCell( fineGridCell, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfCell );


}

#ifdef Parallel
void peanoclaw::adapters::Cleanup::mergeWithNeighbour(
  peanoclaw::Vertex&  vertex,
  const peanoclaw::Vertex&  neighbour,
  int                                           fromRank,
  const tarch::la::Vector<DIMENSIONS,double>&   fineGridX,
  const tarch::la::Vector<DIMENSIONS,double>&   fineGridH,
  int                                           level
) {

   _map2Cleanup.mergeWithNeighbour( vertex, neighbour, fromRank, fineGridX, fineGridH, level );


}

void peanoclaw::adapters::Cleanup::prepareSendToNeighbour(
  peanoclaw::Vertex&  vertex,
  int  toRank,
  int  level
) {

   _map2Cleanup.prepareSendToNeighbour( vertex, toRank, level );


}

void peanoclaw::adapters::Cleanup::prepareCopyToRemoteNode(
  peanoclaw::Vertex&  localVertex,
  int  toRank
) {

   _map2Cleanup.prepareCopyToRemoteNode( localVertex, toRank );


}

void peanoclaw::adapters::Cleanup::prepareCopyToRemoteNode(
  peanoclaw::Cell&  localCell,
  int  toRank
) {

   _map2Cleanup.prepareCopyToRemoteNode( localCell, toRank );


}

void peanoclaw::adapters::Cleanup::mergeWithRemoteDataDueToForkOrJoin(
  peanoclaw::Vertex&  localVertex,
  const peanoclaw::Vertex&  masterOrWorkerVertex,
  int                                       fromRank,
  const tarch::la::Vector<DIMENSIONS,double>&  x,
  const tarch::la::Vector<DIMENSIONS,double>&  h,
  int                                       level
) {

   _map2Cleanup.mergeWithRemoteDataDueToForkOrJoin( localVertex, masterOrWorkerVertex, fromRank, x, h, level );


}

void peanoclaw::adapters::Cleanup::mergeWithRemoteDataDueToForkOrJoin(
  peanoclaw::Cell&  localCell,
  const peanoclaw::Cell&  masterOrWorkerCell,
  int                                       fromRank,
  const tarch::la::Vector<DIMENSIONS,double>&  x,
  const tarch::la::Vector<DIMENSIONS,double>&  h,
  int                                       level
) {

   _map2Cleanup.mergeWithRemoteDataDueToForkOrJoin( localCell, masterOrWorkerCell, fromRank, x, h, level );


}

void peanoclaw::adapters::Cleanup::prepareSendToWorker(
  peanoclaw::Cell&                 fineGridCell,
  peanoclaw::Vertex * const        fineGridVertices,
  const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
  peanoclaw::Vertex * const        coarseGridVertices,
  const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
  peanoclaw::Cell&                 coarseGridCell,
  const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell,
  int                                                                  worker
) {

   _map2Cleanup.prepareSendToWorker( fineGridCell, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfCell, worker );


}

void peanoclaw::adapters::Cleanup::prepareSendToMaster(
  peanoclaw::Cell&     localCell,
  peanoclaw::Vertex *  vertices,
  const peano::grid::VertexEnumerator&  verticesEnumerator
) {

   _map2Cleanup.prepareSendToMaster( localCell, vertices, verticesEnumerator );


}

void peanoclaw::adapters::Cleanup::mergeWithMaster(
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

   _map2Cleanup.mergeWithMaster( workerGridCell, workerGridVertices, workerEnumerator, fineGridCell, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfCell, worker, workerState, masterState );


}

void peanoclaw::adapters::Cleanup::receiveDataFromMaster(
  peanoclaw::Cell&                    receivedCell, 
  peanoclaw::Vertex *                 receivedVertices,
  const peano::grid::VertexEnumerator&    verticesEnumerator
) {

   _map2Cleanup.receiveDataFromMaster( receivedCell, receivedVertices, verticesEnumerator );


}


void peanoclaw::adapters::Cleanup::mergeWithWorker(
  peanoclaw::Cell&           localCell, 
  const peanoclaw::Cell&     receivedMasterCell
) {

   _map2Cleanup.mergeWithWorker( localCell, receivedMasterCell );


}

void peanoclaw::adapters::Cleanup::mergeWithWorker(
  peanoclaw::Vertex&        localVertex,
  const peanoclaw::Vertex&  receivedMasterVertex
) {

   _map2Cleanup.mergeWithWorker( localVertex, receivedMasterVertex );


}
#endif

void peanoclaw::adapters::Cleanup::touchVertexFirstTime(
      peanoclaw::Vertex&               fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridH,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfVertex
) {

  _map2Cleanup.touchVertexFirstTime( fineGridVertex, fineGridX, fineGridH, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfVertex );


}


void peanoclaw::adapters::Cleanup::touchVertexLastTime(
      peanoclaw::Vertex&         fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridH,
      peanoclaw::Vertex * const  coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&           coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfVertex
) {

  _map2Cleanup.touchVertexLastTime( fineGridVertex, fineGridX, fineGridH, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfVertex );


}


void peanoclaw::adapters::Cleanup::enterCell(
      peanoclaw::Cell&                 fineGridCell,
      peanoclaw::Vertex * const        fineGridVertices,
      const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell
) {

  _map2Cleanup.enterCell( fineGridCell, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfCell );


}


void peanoclaw::adapters::Cleanup::leaveCell(
      peanoclaw::Cell&           fineGridCell,
      peanoclaw::Vertex * const  fineGridVertices,
      const peano::grid::VertexEnumerator&          fineGridVerticesEnumerator,
      peanoclaw::Vertex * const  coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&           coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfCell
) {

  _map2Cleanup.leaveCell( fineGridCell, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfCell );


}


void peanoclaw::adapters::Cleanup::beginIteration(
  peanoclaw::State&  solverState
) {

  _map2Cleanup.beginIteration( solverState );


}


void peanoclaw::adapters::Cleanup::endIteration(
  peanoclaw::State&  solverState
) {

  _map2Cleanup.endIteration( solverState );


}




void peanoclaw::adapters::Cleanup::descend(
  peanoclaw::Cell * const          fineGridCells,
  peanoclaw::Vertex * const        fineGridVertices,
  const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
  peanoclaw::Vertex * const        coarseGridVertices,
  const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
  peanoclaw::Cell&                 coarseGridCell
) {

  _map2Cleanup.descend( fineGridCells, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell );


}


void peanoclaw::adapters::Cleanup::ascend(
  peanoclaw::Cell * const    fineGridCells,
  peanoclaw::Vertex * const  fineGridVertices,
  const peano::grid::VertexEnumerator&          fineGridVerticesEnumerator,
  peanoclaw::Vertex * const  coarseGridVertices,
  const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
  peanoclaw::Cell&           coarseGridCell
) {

  _map2Cleanup.ascend( fineGridCells, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell );


}
