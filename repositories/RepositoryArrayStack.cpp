#include "peanoclaw/repositories/RepositoryArrayStack.h"

#include "tarch/Assertions.h"
#include "tarch/timing/Watch.h"

#ifdef Parallel
#include "tarch/parallel/Node.h"
#include "tarch/parallel/NodePool.h"
#include "peano/parallel/SendReceiveBufferPool.h"
#include "peano/parallel/loadbalancing/Oracle.h"
#endif

#include "peano/datatraversal/autotuning/Oracle.h"


tarch::logging::Log peanoclaw::repositories::RepositoryArrayStack::_log( "peanoclaw::repositories::RepositoryArrayStack" );


peanoclaw::repositories::RepositoryArrayStack::RepositoryArrayStack(
  peano::geometry::Geometry&                   geometry,
  const tarch::la::Vector<DIMENSIONS,double>&  domainSize,
  const tarch::la::Vector<DIMENSIONS,double>&  domainOffset,
  int                                          maximumSizeOfCellInOutStack,
  int                                          maximumSizeOfVertexInOutStack,
  int                                          maximumSizeOfVertexTemporaryStack
):
  _geometry(geometry),
  _cellStack(maximumSizeOfCellInOutStack),
  _vertexStack(maximumSizeOfVertexInOutStack, maximumSizeOfVertexTemporaryStack),
  _solverState(),
  _gridWithInitialiseGrid(_vertexStack,_cellStack,_geometry,_solverState,domainSize,domainOffset,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithPlot(_vertexStack,_cellStack,_geometry,_solverState,domainSize,domainOffset,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithRemesh(_vertexStack,_cellStack,_geometry,_solverState,domainSize,domainOffset,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithSolveTimestep(_vertexStack,_cellStack,_geometry,_solverState,domainSize,domainOffset,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithSolveTimestepAndPlot(_vertexStack,_cellStack,_geometry,_solverState,domainSize,domainOffset,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithGatherCurrentSolution(_vertexStack,_cellStack,_geometry,_solverState,domainSize,domainOffset,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithCleanup(_vertexStack,_cellStack,_geometry,_solverState,domainSize,domainOffset,_regularGridContainer,_traversalOrderOnTopLevel),

  _repositoryState() {
  logTraceIn( "RepositoryArrayStack(...)" );
  
  _repositoryState.setAction( peanoclaw::records::RepositoryState::Terminate );

  peano::datatraversal::autotuning::Oracle::getInstance().setNumberOfOracles(7 +3);
  #ifdef Parallel
  peano::parallel::loadbalancing::Oracle::getInstance().setNumberOfOracles(7 +3 );
  #endif
  
  logTraceOut( "RepositoryArrayStack(...)" );
}



peanoclaw::repositories::RepositoryArrayStack::RepositoryArrayStack(
  peano::geometry::Geometry&                   geometry,
  int                                          maximumSizeOfCellInOutStack,
  int                                          maximumSizeOfVertexInOutStack,
  int                                          maximumSizeOfVertexTemporaryStack
):
  _geometry(geometry),
  _cellStack(maximumSizeOfCellInOutStack),
  _vertexStack(maximumSizeOfVertexInOutStack,maximumSizeOfVertexTemporaryStack),
  _solverState(),
  _gridWithInitialiseGrid(_vertexStack,_cellStack,_geometry,_solverState,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithPlot(_vertexStack,_cellStack,_geometry,_solverState,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithRemesh(_vertexStack,_cellStack,_geometry,_solverState,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithSolveTimestep(_vertexStack,_cellStack,_geometry,_solverState,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithSolveTimestepAndPlot(_vertexStack,_cellStack,_geometry,_solverState,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithGatherCurrentSolution(_vertexStack,_cellStack,_geometry,_solverState,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithCleanup(_vertexStack,_cellStack,_geometry,_solverState,_regularGridContainer,_traversalOrderOnTopLevel),

  _repositoryState() {
  logTraceIn( "RepositoryArrayStack(Geometry&)" );
  
  _repositoryState.setAction( peanoclaw::records::RepositoryState::Terminate );

  peano::datatraversal::autotuning::Oracle::getInstance().setNumberOfOracles(7 +3);
  #ifdef Parallel
  peano::parallel::loadbalancing::Oracle::getInstance().setNumberOfOracles(7 +3 );
  #endif
  
  logTraceOut( "RepositoryArrayStack(Geometry&)" );
}
    
   
peanoclaw::repositories::RepositoryArrayStack::~RepositoryArrayStack() {
  assertion( _repositoryState.getAction() == peanoclaw::records::RepositoryState::Terminate );
}


void peanoclaw::repositories::RepositoryArrayStack::restart(
  const tarch::la::Vector<DIMENSIONS,double>&  domainSize,
  const tarch::la::Vector<DIMENSIONS,double>&  domainOffset,
  int                                          domainLevel
) {
  logTraceInWith3Arguments( "restart(...)", domainSize, domainOffset, domainLevel );
  #ifdef Parallel
  assertion( !tarch::parallel::Node::getInstance().isGlobalMaster());
  #endif
  
  logInfo( "restart(...)", "start node for subdomain " << domainOffset << "x" << domainSize << " on level " << domainLevel );
  
  assertion( _repositoryState.getAction() == peanoclaw::records::RepositoryState::Terminate );

  _vertexStack.clear();
  _cellStack.clear();

  _gridWithInitialiseGrid.restart(domainSize,domainOffset,domainLevel);
  _gridWithPlot.restart(domainSize,domainOffset,domainLevel);
  _gridWithRemesh.restart(domainSize,domainOffset,domainLevel);
  _gridWithSolveTimestep.restart(domainSize,domainOffset,domainLevel);
  _gridWithSolveTimestepAndPlot.restart(domainSize,domainOffset,domainLevel);
  _gridWithGatherCurrentSolution.restart(domainSize,domainOffset,domainLevel);
  _gridWithCleanup.restart(domainSize,domainOffset,domainLevel);

 
   _solverState.restart();
 
  logTraceOut( "restart(...)" );
}


void peanoclaw::repositories::RepositoryArrayStack::terminate() {
  logTraceIn( "terminate()" );
  
  _repositoryState.setAction( peanoclaw::records::RepositoryState::Terminate );
  
  #ifdef Parallel
  if (tarch::parallel::Node::getInstance().isGlobalMaster()) {
    tarch::parallel::NodePool::getInstance().broadcastToWorkingNodes(
      _repositoryState,
      peano::parallel::SendReceiveBufferPool::getInstance().getIterationManagementTag()
    );
  }
  peano::parallel::SendReceiveBufferPool::getInstance().terminate();
  #endif
  
  _gridWithInitialiseGrid.terminate();
  _gridWithPlot.terminate();
  _gridWithRemesh.terminate();
  _gridWithSolveTimestep.terminate();
  _gridWithSolveTimestepAndPlot.terminate();
  _gridWithGatherCurrentSolution.terminate();
  _gridWithCleanup.terminate();

  logTraceOut( "terminate()" );
}


peanoclaw::State& peanoclaw::repositories::RepositoryArrayStack::getState() {
  logInfo( "terminate()", "rank has terminated" );
  return _solverState;
}

   
void peanoclaw::repositories::RepositoryArrayStack::iterate(bool reduceState) {
  tarch::timing::Watch watch( "peanoclaw::repositories::RepositoryArrayStack", "iterate(bool)", false);
  
  #ifdef Parallel
  if (tarch::parallel::Node::getInstance().isGlobalMaster()) {
    _repositoryState.setReduceState(reduceState);
    tarch::parallel::NodePool::getInstance().broadcastToWorkingNodes(
      _repositoryState,
      peano::parallel::SendReceiveBufferPool::getInstance().getIterationManagementTag()
    );
  }
  else {
    reduceState = _repositoryState.getReduceState();
  }
  #endif

  peano::datatraversal::autotuning::Oracle::getInstance().switchToOracle(_repositoryState.getAction());
  #ifdef Parallel
  peano::parallel::loadbalancing::Oracle::getInstance().switchToOracle(_repositoryState.getAction());
  #endif

  switch ( _repositoryState.getAction()) {
    case peanoclaw::records::RepositoryState::UseAdapterInitialiseGrid: watch.startTimer(); _gridWithInitialiseGrid.iterate(reduceState); watch.stopTimer(); _measureInitialiseGridCPUTime.setValue( watch.getCPUTime() ); _measureInitialiseGridCalendarTime.setValue( watch.getCalendarTime() ); break;
    case peanoclaw::records::RepositoryState::UseAdapterPlot: watch.startTimer(); _gridWithPlot.iterate(reduceState); watch.stopTimer(); _measurePlotCPUTime.setValue( watch.getCPUTime() ); _measurePlotCalendarTime.setValue( watch.getCalendarTime() ); break;
    case peanoclaw::records::RepositoryState::UseAdapterRemesh: watch.startTimer(); _gridWithRemesh.iterate(reduceState); watch.stopTimer(); _measureRemeshCPUTime.setValue( watch.getCPUTime() ); _measureRemeshCalendarTime.setValue( watch.getCalendarTime() ); break;
    case peanoclaw::records::RepositoryState::UseAdapterSolveTimestep: watch.startTimer(); _gridWithSolveTimestep.iterate(reduceState); watch.stopTimer(); _measureSolveTimestepCPUTime.setValue( watch.getCPUTime() ); _measureSolveTimestepCalendarTime.setValue( watch.getCalendarTime() ); break;
    case peanoclaw::records::RepositoryState::UseAdapterSolveTimestepAndPlot: watch.startTimer(); _gridWithSolveTimestepAndPlot.iterate(reduceState); watch.stopTimer(); _measureSolveTimestepAndPlotCPUTime.setValue( watch.getCPUTime() ); _measureSolveTimestepAndPlotCalendarTime.setValue( watch.getCalendarTime() ); break;
    case peanoclaw::records::RepositoryState::UseAdapterGatherCurrentSolution: watch.startTimer(); _gridWithGatherCurrentSolution.iterate(reduceState); watch.stopTimer(); _measureGatherCurrentSolutionCPUTime.setValue( watch.getCPUTime() ); _measureGatherCurrentSolutionCalendarTime.setValue( watch.getCalendarTime() ); break;
    case peanoclaw::records::RepositoryState::UseAdapterCleanup: watch.startTimer(); _gridWithCleanup.iterate(reduceState); watch.stopTimer(); _measureCleanupCPUTime.setValue( watch.getCPUTime() ); _measureCleanupCalendarTime.setValue( watch.getCalendarTime() ); break;

    case peanoclaw::records::RepositoryState::Terminate:
      assertionMsg( false, "this branch/state should never be reached" ); 
      break;
    case peanoclaw::records::RepositoryState::ReadCheckpoint:
      assertionMsg( false, "not implemented yet" );
      break;
    case peanoclaw::records::RepositoryState::WriteCheckpoint:
      assertionMsg( false, "not implemented yet" );
      break;
  }
  
  #ifdef Parallel
  if (_solverState.isJoiningWithMaster()) {
    _repositoryState.setAction( peanoclaw::records::RepositoryState::Terminate );
  }
  #endif
}

 void peanoclaw::repositories::RepositoryArrayStack::switchToInitialiseGrid() { _repositoryState.setAction(peanoclaw::records::RepositoryState::UseAdapterInitialiseGrid); }
 void peanoclaw::repositories::RepositoryArrayStack::switchToPlot() { _repositoryState.setAction(peanoclaw::records::RepositoryState::UseAdapterPlot); }
 void peanoclaw::repositories::RepositoryArrayStack::switchToRemesh() { _repositoryState.setAction(peanoclaw::records::RepositoryState::UseAdapterRemesh); }
 void peanoclaw::repositories::RepositoryArrayStack::switchToSolveTimestep() { _repositoryState.setAction(peanoclaw::records::RepositoryState::UseAdapterSolveTimestep); }
 void peanoclaw::repositories::RepositoryArrayStack::switchToSolveTimestepAndPlot() { _repositoryState.setAction(peanoclaw::records::RepositoryState::UseAdapterSolveTimestepAndPlot); }
 void peanoclaw::repositories::RepositoryArrayStack::switchToGatherCurrentSolution() { _repositoryState.setAction(peanoclaw::records::RepositoryState::UseAdapterGatherCurrentSolution); }
 void peanoclaw::repositories::RepositoryArrayStack::switchToCleanup() { _repositoryState.setAction(peanoclaw::records::RepositoryState::UseAdapterCleanup); }



 bool peanoclaw::repositories::RepositoryArrayStack::isActiveAdapterInitialiseGrid() const { return _repositoryState.getAction() == peanoclaw::records::RepositoryState::UseAdapterInitialiseGrid; }
 bool peanoclaw::repositories::RepositoryArrayStack::isActiveAdapterPlot() const { return _repositoryState.getAction() == peanoclaw::records::RepositoryState::UseAdapterPlot; }
 bool peanoclaw::repositories::RepositoryArrayStack::isActiveAdapterRemesh() const { return _repositoryState.getAction() == peanoclaw::records::RepositoryState::UseAdapterRemesh; }
 bool peanoclaw::repositories::RepositoryArrayStack::isActiveAdapterSolveTimestep() const { return _repositoryState.getAction() == peanoclaw::records::RepositoryState::UseAdapterSolveTimestep; }
 bool peanoclaw::repositories::RepositoryArrayStack::isActiveAdapterSolveTimestepAndPlot() const { return _repositoryState.getAction() == peanoclaw::records::RepositoryState::UseAdapterSolveTimestepAndPlot; }
 bool peanoclaw::repositories::RepositoryArrayStack::isActiveAdapterGatherCurrentSolution() const { return _repositoryState.getAction() == peanoclaw::records::RepositoryState::UseAdapterGatherCurrentSolution; }
 bool peanoclaw::repositories::RepositoryArrayStack::isActiveAdapterCleanup() const { return _repositoryState.getAction() == peanoclaw::records::RepositoryState::UseAdapterCleanup; }



peano::grid::Checkpoint<peanoclaw::Vertex, peanoclaw::Cell>* peanoclaw::repositories::RepositoryArrayStack::createEmptyCheckpoint() {
  return new peano::grid::Checkpoint<peanoclaw::Vertex, peanoclaw::Cell>();
} 


void peanoclaw::repositories::RepositoryArrayStack::writeCheckpoint(peano::grid::Checkpoint<peanoclaw::Vertex, peanoclaw::Cell> * const checkpoint) {
  _solverState.writeToCheckpoint( *checkpoint );
  _vertexStack.writeToCheckpoint( *checkpoint );
  _cellStack.writeToCheckpoint( *checkpoint );
} 


void peanoclaw::repositories::RepositoryArrayStack::setMaximumMemoryFootprintForTemporaryRegularGrids(double value) {
  _regularGridContainer.setMaximumMemoryFootprintForTemporaryRegularGrids(value);
}


void peanoclaw::repositories::RepositoryArrayStack::readCheckpoint( peano::grid::Checkpoint<peanoclaw::Vertex, peanoclaw::Cell> const * const checkpoint ) {
  assertionMsg( checkpoint->isValid(), "checkpoint has to be valid if you call this operation" );

  _solverState.readFromCheckpoint( *checkpoint );
  _vertexStack.readFromCheckpoint( *checkpoint );
  _cellStack.readFromCheckpoint( *checkpoint );
}


#ifdef Parallel
bool peanoclaw::repositories::RepositoryArrayStack::continueToIterate() {
  logTraceIn( "continueToIterate()" );

  assertion( !tarch::parallel::Node::getInstance().isGlobalMaster());

  bool result;
  if ( _solverState.hasJoinedWithMaster() ) {
    result = false;
  }
  else {
    int masterNode = tarch::parallel::Node::getInstance().getGlobalMasterRank();
    assertion( masterNode != -1 );

    _repositoryState.receive( masterNode, peano::parallel::SendReceiveBufferPool::getInstance().getIterationManagementTag(), true );

    result = _repositoryState.getAction()!=peanoclaw::records::RepositoryState::Terminate;
  }
   
  logTraceOutWith1Argument( "continueToIterate()", result );
  return result;
}
#endif


void peanoclaw::repositories::RepositoryArrayStack::logIterationStatistics() const {
  logInfo( "logIterationStatistics()", "|| adapter name \t || iterations \t || total CPU time [t]=s \t || average CPU time [t]=s \t || total user time [t]=s \t || average user time [t]=s  || CPU time properties  || user time properties " );  
   logInfo( "logIterationStatistics()", "| InitialiseGrid \t |  " << _measureInitialiseGridCPUTime.getNumberOfMeasurements() << " \t |  " << _measureInitialiseGridCPUTime.getAccumulatedValue() << " \t |  " << _measureInitialiseGridCPUTime.getValue()  << " \t |  " << _measureInitialiseGridCalendarTime.getAccumulatedValue() << " \t |  " << _measureInitialiseGridCalendarTime.getValue() << " \t |  " << _measureInitialiseGridCPUTime.toString() << " \t |  " << _measureInitialiseGridCalendarTime.toString() );
   logInfo( "logIterationStatistics()", "| Plot \t |  " << _measurePlotCPUTime.getNumberOfMeasurements() << " \t |  " << _measurePlotCPUTime.getAccumulatedValue() << " \t |  " << _measurePlotCPUTime.getValue()  << " \t |  " << _measurePlotCalendarTime.getAccumulatedValue() << " \t |  " << _measurePlotCalendarTime.getValue() << " \t |  " << _measurePlotCPUTime.toString() << " \t |  " << _measurePlotCalendarTime.toString() );
   logInfo( "logIterationStatistics()", "| Remesh \t |  " << _measureRemeshCPUTime.getNumberOfMeasurements() << " \t |  " << _measureRemeshCPUTime.getAccumulatedValue() << " \t |  " << _measureRemeshCPUTime.getValue()  << " \t |  " << _measureRemeshCalendarTime.getAccumulatedValue() << " \t |  " << _measureRemeshCalendarTime.getValue() << " \t |  " << _measureRemeshCPUTime.toString() << " \t |  " << _measureRemeshCalendarTime.toString() );
   logInfo( "logIterationStatistics()", "| SolveTimestep \t |  " << _measureSolveTimestepCPUTime.getNumberOfMeasurements() << " \t |  " << _measureSolveTimestepCPUTime.getAccumulatedValue() << " \t |  " << _measureSolveTimestepCPUTime.getValue()  << " \t |  " << _measureSolveTimestepCalendarTime.getAccumulatedValue() << " \t |  " << _measureSolveTimestepCalendarTime.getValue() << " \t |  " << _measureSolveTimestepCPUTime.toString() << " \t |  " << _measureSolveTimestepCalendarTime.toString() );
   logInfo( "logIterationStatistics()", "| SolveTimestepAndPlot \t |  " << _measureSolveTimestepAndPlotCPUTime.getNumberOfMeasurements() << " \t |  " << _measureSolveTimestepAndPlotCPUTime.getAccumulatedValue() << " \t |  " << _measureSolveTimestepAndPlotCPUTime.getValue()  << " \t |  " << _measureSolveTimestepAndPlotCalendarTime.getAccumulatedValue() << " \t |  " << _measureSolveTimestepAndPlotCalendarTime.getValue() << " \t |  " << _measureSolveTimestepAndPlotCPUTime.toString() << " \t |  " << _measureSolveTimestepAndPlotCalendarTime.toString() );
   logInfo( "logIterationStatistics()", "| GatherCurrentSolution \t |  " << _measureGatherCurrentSolutionCPUTime.getNumberOfMeasurements() << " \t |  " << _measureGatherCurrentSolutionCPUTime.getAccumulatedValue() << " \t |  " << _measureGatherCurrentSolutionCPUTime.getValue()  << " \t |  " << _measureGatherCurrentSolutionCalendarTime.getAccumulatedValue() << " \t |  " << _measureGatherCurrentSolutionCalendarTime.getValue() << " \t |  " << _measureGatherCurrentSolutionCPUTime.toString() << " \t |  " << _measureGatherCurrentSolutionCalendarTime.toString() );
   logInfo( "logIterationStatistics()", "| Cleanup \t |  " << _measureCleanupCPUTime.getNumberOfMeasurements() << " \t |  " << _measureCleanupCPUTime.getAccumulatedValue() << " \t |  " << _measureCleanupCPUTime.getValue()  << " \t |  " << _measureCleanupCalendarTime.getAccumulatedValue() << " \t |  " << _measureCleanupCalendarTime.getValue() << " \t |  " << _measureCleanupCPUTime.toString() << " \t |  " << _measureCleanupCalendarTime.toString() );

}


void peanoclaw::repositories::RepositoryArrayStack::clearIterationStatistics() {
   _measureInitialiseGridCPUTime.erase();
   _measurePlotCPUTime.erase();
   _measureRemeshCPUTime.erase();
   _measureSolveTimestepCPUTime.erase();
   _measureSolveTimestepAndPlotCPUTime.erase();
   _measureGatherCurrentSolutionCPUTime.erase();
   _measureCleanupCPUTime.erase();

   _measureInitialiseGridCalendarTime.erase();
   _measurePlotCalendarTime.erase();
   _measureRemeshCalendarTime.erase();
   _measureSolveTimestepCalendarTime.erase();
   _measureSolveTimestepAndPlotCalendarTime.erase();
   _measureGatherCurrentSolutionCalendarTime.erase();
   _measureCleanupCalendarTime.erase();

}
