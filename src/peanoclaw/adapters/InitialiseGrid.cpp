#include "peanoclaw/adapters/InitialiseGrid.h"



peano::MappingSpecification   peanoclaw::adapters::InitialiseGrid::touchVertexLastTimeSpecification() {
  return peano::MappingSpecification::getMostGeneralSpecification()
   & peanoclaw::mappings::Remesh::touchVertexLastTimeSpecification()
   & peanoclaw::mappings::InitialiseGrid::touchVertexLastTimeSpecification()


;
}


peano::MappingSpecification   peanoclaw::adapters::InitialiseGrid::touchVertexFirstTimeSpecification() { 
  return peano::MappingSpecification::getMostGeneralSpecification()
   & peanoclaw::mappings::Remesh::touchVertexFirstTimeSpecification()
   & peanoclaw::mappings::InitialiseGrid::touchVertexFirstTimeSpecification()


;
}


peano::MappingSpecification   peanoclaw::adapters::InitialiseGrid::enterCellSpecification() {
  return peano::MappingSpecification::getMostGeneralSpecification()
   & peanoclaw::mappings::Remesh::enterCellSpecification()
   & peanoclaw::mappings::InitialiseGrid::enterCellSpecification()


;
}


peano::MappingSpecification   peanoclaw::adapters::InitialiseGrid::leaveCellSpecification() {
  return peano::MappingSpecification::getMostGeneralSpecification()
   & peanoclaw::mappings::Remesh::leaveCellSpecification()
   & peanoclaw::mappings::InitialiseGrid::leaveCellSpecification()


;
}


peano::MappingSpecification   peanoclaw::adapters::InitialiseGrid::ascendSpecification() {
  return peano::MappingSpecification::getMostGeneralSpecification()
   & peanoclaw::mappings::Remesh::ascendSpecification()
   & peanoclaw::mappings::InitialiseGrid::ascendSpecification()


;
}


peano::MappingSpecification   peanoclaw::adapters::InitialiseGrid::descendSpecification() {
  return peano::MappingSpecification::getMostGeneralSpecification()
   & peanoclaw::mappings::Remesh::descendSpecification()
   & peanoclaw::mappings::InitialiseGrid::descendSpecification()


;
}


peanoclaw::adapters::InitialiseGrid::InitialiseGrid() {
}


peanoclaw::adapters::InitialiseGrid::~InitialiseGrid() {
}


#if defined(SharedMemoryParallelisation)
peanoclaw::adapters::InitialiseGrid::InitialiseGrid(const InitialiseGrid&  masterThread):
  _map2Remesh(masterThread._map2Remesh) , 
  _map2InitialiseGrid(masterThread._map2InitialiseGrid) 

 

{
}


void peanoclaw::adapters::InitialiseGrid::mergeWithWorkerThread(const InitialiseGrid& workerThread) {

  _map2Remesh.mergeWithWorkerThread(workerThread._map2Remesh);
  _map2InitialiseGrid.mergeWithWorkerThread(workerThread._map2InitialiseGrid);


}
#endif


void peanoclaw::adapters::InitialiseGrid::createHangingVertex(
      peanoclaw::Vertex&     fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                fineGridH,
      peanoclaw::Vertex * const   coarseGridVertices,
      const peano::grid::VertexEnumerator&      coarseGridVerticesEnumerator,
      peanoclaw::Cell&       coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                   fineGridPositionOfVertex
) {


  _map2Remesh.createHangingVertex(fineGridVertex, fineGridX, fineGridH, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfVertex );
  _map2InitialiseGrid.createHangingVertex(fineGridVertex, fineGridX, fineGridH, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfVertex );


}


void peanoclaw::adapters::InitialiseGrid::destroyHangingVertex(
      const peanoclaw::Vertex&   fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridH,
      peanoclaw::Vertex * const  coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&           coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfVertex
) {
  _map2Remesh.destroyHangingVertex(fineGridVertex, fineGridX, fineGridH, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfVertex );
  _map2InitialiseGrid.destroyHangingVertex(fineGridVertex, fineGridX, fineGridH, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfVertex );



}


void peanoclaw::adapters::InitialiseGrid::createInnerVertex(
      peanoclaw::Vertex&               fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridH,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfVertex
) {


  _map2Remesh.createInnerVertex(fineGridVertex, fineGridX, fineGridH, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfVertex );
  _map2InitialiseGrid.createInnerVertex(fineGridVertex, fineGridX, fineGridH, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfVertex );

}


void peanoclaw::adapters::InitialiseGrid::createBoundaryVertex(
      peanoclaw::Vertex&               fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridH,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfVertex
) {


  _map2Remesh.createBoundaryVertex( fineGridVertex, fineGridX, fineGridH, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfVertex );
  _map2InitialiseGrid.createBoundaryVertex( fineGridVertex, fineGridX, fineGridH, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfVertex );

}


void peanoclaw::adapters::InitialiseGrid::destroyVertex(
      const peanoclaw::Vertex&   fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridH,
      peanoclaw::Vertex * const  coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&           coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfVertex
) {

  _map2Remesh.destroyVertex( fineGridVertex, fineGridX, fineGridH, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfVertex );
  _map2InitialiseGrid.destroyVertex( fineGridVertex, fineGridX, fineGridH, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfVertex );


}


void peanoclaw::adapters::InitialiseGrid::createCell(
      peanoclaw::Cell&                 fineGridCell,
      peanoclaw::Vertex * const        fineGridVertices,
      const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell
) {


  _map2Remesh.createCell( fineGridCell, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfCell );
  _map2InitialiseGrid.createCell( fineGridCell, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfCell );

}


void peanoclaw::adapters::InitialiseGrid::destroyCell(
      const peanoclaw::Cell&           fineGridCell,
      peanoclaw::Vertex * const        fineGridVertices,
      const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell
) {

  _map2Remesh.destroyCell( fineGridCell, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfCell );
  _map2InitialiseGrid.destroyCell( fineGridCell, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfCell );


}

#ifdef Parallel
void peanoclaw::adapters::InitialiseGrid::mergeWithNeighbour(
  peanoclaw::Vertex&  vertex,
  const peanoclaw::Vertex&  neighbour,
  int                                           fromRank,
  const tarch::la::Vector<DIMENSIONS,double>&   fineGridX,
  const tarch::la::Vector<DIMENSIONS,double>&   fineGridH,
  int                                           level
) {


   _map2Remesh.mergeWithNeighbour( vertex, neighbour, fromRank, fineGridX, fineGridH, level );
   _map2InitialiseGrid.mergeWithNeighbour( vertex, neighbour, fromRank, fineGridX, fineGridH, level );

}

void peanoclaw::adapters::InitialiseGrid::prepareSendToNeighbour(
  peanoclaw::Vertex&  vertex,
  int                                           toRank,
  const tarch::la::Vector<DIMENSIONS,double>&   x,
  const tarch::la::Vector<DIMENSIONS,double>&   h,
  int                                           level
) {
   _map2Remesh.prepareSendToNeighbour( vertex, toRank, x, h, level );
   _map2InitialiseGrid.prepareSendToNeighbour( vertex, toRank, x, h, level );


}

void peanoclaw::adapters::InitialiseGrid::prepareCopyToRemoteNode(
  peanoclaw::Vertex&  localVertex,
  int                                           toRank,
  const tarch::la::Vector<DIMENSIONS,double>&   x,
  const tarch::la::Vector<DIMENSIONS,double>&   h,
  int                                           level
) {
   _map2Remesh.prepareCopyToRemoteNode( localVertex, toRank, x, h, level );
   _map2InitialiseGrid.prepareCopyToRemoteNode( localVertex, toRank, x, h, level );


}

void peanoclaw::adapters::InitialiseGrid::prepareCopyToRemoteNode(
  peanoclaw::Cell&  localCell,
      int                                           toRank,
      const tarch::la::Vector<DIMENSIONS,double>&   x,
      const tarch::la::Vector<DIMENSIONS,double>&   h,
      int                                           level
) {
   _map2Remesh.prepareCopyToRemoteNode( localCell, toRank, x, h, level );
   _map2InitialiseGrid.prepareCopyToRemoteNode( localCell, toRank, x, h, level );


}

void peanoclaw::adapters::InitialiseGrid::mergeWithRemoteDataDueToForkOrJoin(
  peanoclaw::Vertex&  localVertex,
  const peanoclaw::Vertex&  masterOrWorkerVertex,
  int                                       fromRank,
  const tarch::la::Vector<DIMENSIONS,double>&  x,
  const tarch::la::Vector<DIMENSIONS,double>&  h,
  int                                       level
) {


   _map2Remesh.mergeWithRemoteDataDueToForkOrJoin( localVertex, masterOrWorkerVertex, fromRank, x, h, level );
   _map2InitialiseGrid.mergeWithRemoteDataDueToForkOrJoin( localVertex, masterOrWorkerVertex, fromRank, x, h, level );

}

void peanoclaw::adapters::InitialiseGrid::mergeWithRemoteDataDueToForkOrJoin(
  peanoclaw::Cell&  localCell,
  const peanoclaw::Cell&  masterOrWorkerCell,
  int                                       fromRank,
  const tarch::la::Vector<DIMENSIONS,double>&  x,
  const tarch::la::Vector<DIMENSIONS,double>&  h,
  int                                       level
) {


   _map2Remesh.mergeWithRemoteDataDueToForkOrJoin( localCell, masterOrWorkerCell, fromRank, x, h, level );
   _map2InitialiseGrid.mergeWithRemoteDataDueToForkOrJoin( localCell, masterOrWorkerCell, fromRank, x, h, level );

}

bool peanoclaw::adapters::InitialiseGrid::prepareSendToWorker(
  peanoclaw::Cell&                 fineGridCell,
  peanoclaw::Vertex * const        fineGridVertices,
  const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
  peanoclaw::Vertex * const        coarseGridVertices,
  const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
  peanoclaw::Cell&                 coarseGridCell,
  const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell,
  int                                                                  worker
) {
  bool result = false;
   result |= _map2Remesh.prepareSendToWorker( fineGridCell, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfCell, worker );
   result |= _map2InitialiseGrid.prepareSendToWorker( fineGridCell, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfCell, worker );


  return result;
}

void peanoclaw::adapters::InitialiseGrid::prepareSendToMaster(
  peanoclaw::Cell&                       localCell,
  peanoclaw::Vertex *                    vertices,
  const peano::grid::VertexEnumerator&       verticesEnumerator, 
  const peanoclaw::Vertex * const        coarseGridVertices,
  const peano::grid::VertexEnumerator&       coarseGridVerticesEnumerator,
  const peanoclaw::Cell&                 coarseGridCell,
  const tarch::la::Vector<DIMENSIONS,int>&   fineGridPositionOfCell
) {

   _map2Remesh.prepareSendToMaster( localCell, vertices, verticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfCell );
   _map2InitialiseGrid.prepareSendToMaster( localCell, vertices, verticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfCell );


}

void peanoclaw::adapters::InitialiseGrid::mergeWithMaster(
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


   _map2Remesh.mergeWithMaster( workerGridCell, workerGridVertices, workerEnumerator, fineGridCell, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfCell, worker, workerState, masterState );
   _map2InitialiseGrid.mergeWithMaster( workerGridCell, workerGridVertices, workerEnumerator, fineGridCell, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfCell, worker, workerState, masterState );

}

void peanoclaw::adapters::InitialiseGrid::receiveDataFromMaster(
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


   _map2Remesh.receiveDataFromMaster( receivedCell, receivedVertices, receivedVerticesEnumerator, receivedCoarseGridVertices, receivedCoarseGridVerticesEnumerator, receivedCoarseGridCell, workersCoarseGridVertices, workersCoarseGridVerticesEnumerator, workersCoarseGridCell, fineGridPositionOfCell );
   _map2InitialiseGrid.receiveDataFromMaster( receivedCell, receivedVertices, receivedVerticesEnumerator, receivedCoarseGridVertices, receivedCoarseGridVerticesEnumerator, receivedCoarseGridCell, workersCoarseGridVertices, workersCoarseGridVerticesEnumerator, workersCoarseGridCell, fineGridPositionOfCell );

}


void peanoclaw::adapters::InitialiseGrid::mergeWithWorker(
  peanoclaw::Cell&           localCell, 
  const peanoclaw::Cell&     receivedMasterCell,
  const tarch::la::Vector<DIMENSIONS,double>&  cellCentre,
  const tarch::la::Vector<DIMENSIONS,double>&  cellSize,
  int                                          level
) {


   _map2Remesh.mergeWithWorker( localCell, receivedMasterCell, cellCentre, cellSize, level );
   _map2InitialiseGrid.mergeWithWorker( localCell, receivedMasterCell, cellCentre, cellSize, level );

}

void peanoclaw::adapters::InitialiseGrid::mergeWithWorker(
  peanoclaw::Vertex&        localVertex,
  const peanoclaw::Vertex&  receivedMasterVertex,
  const tarch::la::Vector<DIMENSIONS,double>&   x,
  const tarch::la::Vector<DIMENSIONS,double>&   h,
  int                                           level
) {

   _map2Remesh.mergeWithWorker( localVertex, receivedMasterVertex, x, h, level );
   _map2InitialiseGrid.mergeWithWorker( localVertex, receivedMasterVertex, x, h, level );


}
#endif

void peanoclaw::adapters::InitialiseGrid::touchVertexFirstTime(
      peanoclaw::Vertex&               fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridH,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfVertex
) {


  _map2Remesh.touchVertexFirstTime( fineGridVertex, fineGridX, fineGridH, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfVertex );
  _map2InitialiseGrid.touchVertexFirstTime( fineGridVertex, fineGridX, fineGridH, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfVertex );

}


void peanoclaw::adapters::InitialiseGrid::touchVertexLastTime(
      peanoclaw::Vertex&         fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridH,
      peanoclaw::Vertex * const  coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&           coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfVertex
) {

  _map2Remesh.touchVertexLastTime( fineGridVertex, fineGridX, fineGridH, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfVertex );
  _map2InitialiseGrid.touchVertexLastTime( fineGridVertex, fineGridX, fineGridH, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfVertex );


}


void peanoclaw::adapters::InitialiseGrid::enterCell(
      peanoclaw::Cell&                 fineGridCell,
      peanoclaw::Vertex * const        fineGridVertices,
      const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell
) {


  _map2Remesh.enterCell( fineGridCell, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfCell );
  _map2InitialiseGrid.enterCell( fineGridCell, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfCell );

}


void peanoclaw::adapters::InitialiseGrid::leaveCell(
      peanoclaw::Cell&           fineGridCell,
      peanoclaw::Vertex * const  fineGridVertices,
      const peano::grid::VertexEnumerator&          fineGridVerticesEnumerator,
      peanoclaw::Vertex * const  coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&           coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfCell
) {

  _map2Remesh.leaveCell( fineGridCell, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfCell );
  _map2InitialiseGrid.leaveCell( fineGridCell, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell, fineGridPositionOfCell );


}


void peanoclaw::adapters::InitialiseGrid::beginIteration(
  peanoclaw::State&  solverState
) {


  _map2Remesh.beginIteration( solverState );
  _map2InitialiseGrid.beginIteration( solverState );

}


void peanoclaw::adapters::InitialiseGrid::endIteration(
  peanoclaw::State&  solverState
) {

  _map2Remesh.endIteration( solverState );
  _map2InitialiseGrid.endIteration( solverState );


}




void peanoclaw::adapters::InitialiseGrid::descend(
  peanoclaw::Cell * const          fineGridCells,
  peanoclaw::Vertex * const        fineGridVertices,
  const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
  peanoclaw::Vertex * const        coarseGridVertices,
  const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
  peanoclaw::Cell&                 coarseGridCell
) {


  _map2Remesh.descend( fineGridCells, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell );
  _map2InitialiseGrid.descend( fineGridCells, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell );

}


void peanoclaw::adapters::InitialiseGrid::ascend(
  peanoclaw::Cell * const    fineGridCells,
  peanoclaw::Vertex * const  fineGridVertices,
  const peano::grid::VertexEnumerator&          fineGridVerticesEnumerator,
  peanoclaw::Vertex * const  coarseGridVertices,
  const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
  peanoclaw::Cell&           coarseGridCell
) {

  _map2Remesh.ascend( fineGridCells, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell );
  _map2InitialiseGrid.ascend( fineGridCells, fineGridVertices, fineGridVerticesEnumerator, coarseGridVertices, coarseGridVerticesEnumerator, coarseGridCell );


}
