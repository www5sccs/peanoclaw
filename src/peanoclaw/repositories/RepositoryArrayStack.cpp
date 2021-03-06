#include "peanoclaw/repositories/RepositoryArrayStack.h"

#include "tarch/Assertions.h"
#include "tarch/timing/Watch.h"

#include "tarch/compiler/CompilerSpecificSettings.h"

#ifdef Parallel
#include "tarch/parallel/Node.h"
#include "tarch/parallel/NodePool.h"
#include "peano/parallel/SendReceiveBufferPool.h"
#include "peano/parallel/loadbalancing/Oracle.h"
#endif

#include "peano/datatraversal/autotuning/Oracle.h"

#include "tarch/compiler/CompilerSpecificSettings.h"

#if !defined(CompilerICC)
#include "peano/grid/Grid.cpph"
#endif


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
  _gridWithInitialiseAndValidateGrid(_vertexStack,_cellStack,_geometry,_solverState,domainSize,domainOffset,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithPlot(_vertexStack,_cellStack,_geometry,_solverState,domainSize,domainOffset,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithPlotAndValidateGrid(_vertexStack,_cellStack,_geometry,_solverState,domainSize,domainOffset,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithRemesh(_vertexStack,_cellStack,_geometry,_solverState,domainSize,domainOffset,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithSolveTimestep(_vertexStack,_cellStack,_geometry,_solverState,domainSize,domainOffset,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithSolveTimestepAndValidateGrid(_vertexStack,_cellStack,_geometry,_solverState,domainSize,domainOffset,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithSolveTimestepAndPlot(_vertexStack,_cellStack,_geometry,_solverState,domainSize,domainOffset,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithSolveTimestepAndPlotAndValidateGrid(_vertexStack,_cellStack,_geometry,_solverState,domainSize,domainOffset,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithGatherCurrentSolution(_vertexStack,_cellStack,_geometry,_solverState,domainSize,domainOffset,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithGatherCurrentSolutionAndValidateGrid(_vertexStack,_cellStack,_geometry,_solverState,domainSize,domainOffset,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithCleanup(_vertexStack,_cellStack,_geometry,_solverState,domainSize,domainOffset,_regularGridContainer,_traversalOrderOnTopLevel),

  _repositoryState() {
  logTraceIn( "RepositoryArrayStack(...)" );
  
  _repositoryState.setAction( peanoclaw::records::RepositoryState::Terminate );

  peano::datatraversal::autotuning::Oracle::getInstance().setNumberOfOracles(peanoclaw::records::RepositoryState::NumberOfAdapters);
  #ifdef Parallel
  peano::parallel::loadbalancing::Oracle::getInstance().setNumberOfOracles(peanoclaw::records::RepositoryState::NumberOfAdapters);
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
  _gridWithInitialiseAndValidateGrid(_vertexStack,_cellStack,_geometry,_solverState,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithPlot(_vertexStack,_cellStack,_geometry,_solverState,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithPlotAndValidateGrid(_vertexStack,_cellStack,_geometry,_solverState,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithRemesh(_vertexStack,_cellStack,_geometry,_solverState,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithSolveTimestep(_vertexStack,_cellStack,_geometry,_solverState,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithSolveTimestepAndValidateGrid(_vertexStack,_cellStack,_geometry,_solverState,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithSolveTimestepAndPlot(_vertexStack,_cellStack,_geometry,_solverState,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithSolveTimestepAndPlotAndValidateGrid(_vertexStack,_cellStack,_geometry,_solverState,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithGatherCurrentSolution(_vertexStack,_cellStack,_geometry,_solverState,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithGatherCurrentSolutionAndValidateGrid(_vertexStack,_cellStack,_geometry,_solverState,_regularGridContainer,_traversalOrderOnTopLevel),
  _gridWithCleanup(_vertexStack,_cellStack,_geometry,_solverState,_regularGridContainer,_traversalOrderOnTopLevel),

  _repositoryState() {
  logTraceIn( "RepositoryArrayStack(Geometry&)" );
  
  _repositoryState.setAction( peanoclaw::records::RepositoryState::Terminate );

  peano::datatraversal::autotuning::Oracle::getInstance().setNumberOfOracles(peanoclaw::records::RepositoryState::NumberOfAdapters);
  #ifdef Parallel
  peano::parallel::loadbalancing::Oracle::getInstance().setNumberOfOracles(peanoclaw::records::RepositoryState::NumberOfAdapters);
  #endif
  
  logTraceOut( "RepositoryArrayStack(Geometry&)" );
}
    
   
peanoclaw::repositories::RepositoryArrayStack::~RepositoryArrayStack() {
  assertion( _repositoryState.getAction() == peanoclaw::records::RepositoryState::Terminate );
}


void peanoclaw::repositories::RepositoryArrayStack::restart(
  const tarch::la::Vector<DIMENSIONS,double>&  domainSize,
  const tarch::la::Vector<DIMENSIONS,double>&  domainOffset,
  int                                          domainLevel,
  const tarch::la::Vector<DIMENSIONS,int>&     positionOfCentralElementWithRespectToCoarserRemoteLevel
) {
  logTraceInWith4Arguments( "restart(...)", domainSize, domainOffset, domainLevel, positionOfCentralElementWithRespectToCoarserRemoteLevel );
  #ifdef Parallel
  assertion( !tarch::parallel::Node::getInstance().isGlobalMaster());
  #endif
  
  logInfo( "restart(...)", "start node for subdomain " << domainOffset << "x" << domainSize << " on level " << domainLevel );
  
  assertion( _repositoryState.getAction() == peanoclaw::records::RepositoryState::Terminate );

  _vertexStack.clear();
  _cellStack.clear();

  _gridWithInitialiseGrid.restart(domainSize,domainOffset,domainLevel,positionOfCentralElementWithRespectToCoarserRemoteLevel);
  _gridWithInitialiseAndValidateGrid.restart(domainSize,domainOffset,domainLevel,positionOfCentralElementWithRespectToCoarserRemoteLevel);
  _gridWithPlot.restart(domainSize,domainOffset,domainLevel,positionOfCentralElementWithRespectToCoarserRemoteLevel);
  _gridWithPlotAndValidateGrid.restart(domainSize,domainOffset,domainLevel,positionOfCentralElementWithRespectToCoarserRemoteLevel);
  _gridWithRemesh.restart(domainSize,domainOffset,domainLevel,positionOfCentralElementWithRespectToCoarserRemoteLevel);
  _gridWithSolveTimestep.restart(domainSize,domainOffset,domainLevel,positionOfCentralElementWithRespectToCoarserRemoteLevel);
  _gridWithSolveTimestepAndValidateGrid.restart(domainSize,domainOffset,domainLevel,positionOfCentralElementWithRespectToCoarserRemoteLevel);
  _gridWithSolveTimestepAndPlot.restart(domainSize,domainOffset,domainLevel,positionOfCentralElementWithRespectToCoarserRemoteLevel);
  _gridWithSolveTimestepAndPlotAndValidateGrid.restart(domainSize,domainOffset,domainLevel,positionOfCentralElementWithRespectToCoarserRemoteLevel);
  _gridWithGatherCurrentSolution.restart(domainSize,domainOffset,domainLevel,positionOfCentralElementWithRespectToCoarserRemoteLevel);
  _gridWithGatherCurrentSolutionAndValidateGrid.restart(domainSize,domainOffset,domainLevel,positionOfCentralElementWithRespectToCoarserRemoteLevel);
  _gridWithCleanup.restart(domainSize,domainOffset,domainLevel,positionOfCentralElementWithRespectToCoarserRemoteLevel);

 
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
  _gridWithInitialiseAndValidateGrid.terminate();
  _gridWithPlot.terminate();
  _gridWithPlotAndValidateGrid.terminate();
  _gridWithRemesh.terminate();
  _gridWithSolveTimestep.terminate();
  _gridWithSolveTimestepAndValidateGrid.terminate();
  _gridWithSolveTimestepAndPlot.terminate();
  _gridWithSolveTimestepAndPlotAndValidateGrid.terminate();
  _gridWithGatherCurrentSolution.terminate();
  _gridWithGatherCurrentSolutionAndValidateGrid.terminate();
  _gridWithCleanup.terminate();

 
  logTraceOut( "terminate()" );
}


peanoclaw::State& peanoclaw::repositories::RepositoryArrayStack::getState() {
  return _solverState;
}


const peanoclaw::State& peanoclaw::repositories::RepositoryArrayStack::getState() const {
  return _solverState;
}

   
void peanoclaw::repositories::RepositoryArrayStack::iterate(int numberOfIterations) {
  tarch::timing::Watch watch( "peanoclaw::repositories::RepositoryArrayStack", "iterate(bool)", false);
  
  #ifdef Parallel
  if (tarch::parallel::Node::getInstance().isGlobalMaster()) {
    _repositoryState.setNumberOfIterations(numberOfIterations);
    tarch::parallel::NodePool::getInstance().broadcastToWorkingNodes(
      _repositoryState,
      peano::parallel::SendReceiveBufferPool::getInstance().getIterationManagementTag()
    );
  }
  else {
    assertionEquals( numberOfIterations, 1 );
    numberOfIterations = _repositoryState.getNumberOfIterations();
  }

  if ( numberOfIterations > 1 && ( peano::parallel::loadbalancing::Oracle::getInstance().isLoadBalancingActivated() || _solverState.isInvolvedInJoinOrFork() )) {
    logWarning( "iterate()", "iterate invoked for multiple traversals though load balancing is switched on or grid is not balanced globally. Use activateLoadBalancing(false) to deactivate the load balancing before" );
  }

  peano::datatraversal::autotuning::Oracle::getInstance().switchToOracle(_repositoryState.getAction());

  peano::parallel::loadbalancing::Oracle::getInstance().switchToOracle(_repositoryState.getAction());
  peano::parallel::loadbalancing::Oracle::getInstance().activateLoadBalancing(_repositoryState.getNumberOfIterations()==1);  
  
  _solverState.currentlyRunsMultipleIterations(_repositoryState.getNumberOfIterations()>1);
  #else
  peano::datatraversal::autotuning::Oracle::getInstance().switchToOracle(_repositoryState.getAction());
  #endif
  
  for (int i=0; i<numberOfIterations; i++) {
    switch ( _repositoryState.getAction()) {
      case peanoclaw::records::RepositoryState::UseAdapterInitialiseGrid: watch.startTimer(); _gridWithInitialiseGrid.iterate(); watch.stopTimer(); _measureInitialiseGridCPUTime.setValue( watch.getCPUTime() ); _measureInitialiseGridCalendarTime.setValue( watch.getCalendarTime() ); break;
      case peanoclaw::records::RepositoryState::UseAdapterInitialiseAndValidateGrid: watch.startTimer(); _gridWithInitialiseAndValidateGrid.iterate(); watch.stopTimer(); _measureInitialiseAndValidateGridCPUTime.setValue( watch.getCPUTime() ); _measureInitialiseAndValidateGridCalendarTime.setValue( watch.getCalendarTime() ); break;
      case peanoclaw::records::RepositoryState::UseAdapterPlot: watch.startTimer(); _gridWithPlot.iterate(); watch.stopTimer(); _measurePlotCPUTime.setValue( watch.getCPUTime() ); _measurePlotCalendarTime.setValue( watch.getCalendarTime() ); break;
      case peanoclaw::records::RepositoryState::UseAdapterPlotAndValidateGrid: watch.startTimer(); _gridWithPlotAndValidateGrid.iterate(); watch.stopTimer(); _measurePlotAndValidateGridCPUTime.setValue( watch.getCPUTime() ); _measurePlotAndValidateGridCalendarTime.setValue( watch.getCalendarTime() ); break;
      case peanoclaw::records::RepositoryState::UseAdapterRemesh: watch.startTimer(); _gridWithRemesh.iterate(); watch.stopTimer(); _measureRemeshCPUTime.setValue( watch.getCPUTime() ); _measureRemeshCalendarTime.setValue( watch.getCalendarTime() ); break;
      case peanoclaw::records::RepositoryState::UseAdapterSolveTimestep: watch.startTimer(); _gridWithSolveTimestep.iterate(); watch.stopTimer(); _measureSolveTimestepCPUTime.setValue( watch.getCPUTime() ); _measureSolveTimestepCalendarTime.setValue( watch.getCalendarTime() ); break;
      case peanoclaw::records::RepositoryState::UseAdapterSolveTimestepAndValidateGrid: watch.startTimer(); _gridWithSolveTimestepAndValidateGrid.iterate(); watch.stopTimer(); _measureSolveTimestepAndValidateGridCPUTime.setValue( watch.getCPUTime() ); _measureSolveTimestepAndValidateGridCalendarTime.setValue( watch.getCalendarTime() ); break;
      case peanoclaw::records::RepositoryState::UseAdapterSolveTimestepAndPlot: watch.startTimer(); _gridWithSolveTimestepAndPlot.iterate(); watch.stopTimer(); _measureSolveTimestepAndPlotCPUTime.setValue( watch.getCPUTime() ); _measureSolveTimestepAndPlotCalendarTime.setValue( watch.getCalendarTime() ); break;
      case peanoclaw::records::RepositoryState::UseAdapterSolveTimestepAndPlotAndValidateGrid: watch.startTimer(); _gridWithSolveTimestepAndPlotAndValidateGrid.iterate(); watch.stopTimer(); _measureSolveTimestepAndPlotAndValidateGridCPUTime.setValue( watch.getCPUTime() ); _measureSolveTimestepAndPlotAndValidateGridCalendarTime.setValue( watch.getCalendarTime() ); break;
      case peanoclaw::records::RepositoryState::UseAdapterGatherCurrentSolution: watch.startTimer(); _gridWithGatherCurrentSolution.iterate(); watch.stopTimer(); _measureGatherCurrentSolutionCPUTime.setValue( watch.getCPUTime() ); _measureGatherCurrentSolutionCalendarTime.setValue( watch.getCalendarTime() ); break;
      case peanoclaw::records::RepositoryState::UseAdapterGatherCurrentSolutionAndValidateGrid: watch.startTimer(); _gridWithGatherCurrentSolutionAndValidateGrid.iterate(); watch.stopTimer(); _measureGatherCurrentSolutionAndValidateGridCPUTime.setValue( watch.getCPUTime() ); _measureGatherCurrentSolutionAndValidateGridCalendarTime.setValue( watch.getCalendarTime() ); break;
      case peanoclaw::records::RepositoryState::UseAdapterCleanup: watch.startTimer(); _gridWithCleanup.iterate(); watch.stopTimer(); _measureCleanupCPUTime.setValue( watch.getCPUTime() ); _measureCleanupCalendarTime.setValue( watch.getCalendarTime() ); break;

      case peanoclaw::records::RepositoryState::Terminate:
        assertionMsg( false, "this branch/state should never be reached" ); 
        break;
      case peanoclaw::records::RepositoryState::NumberOfAdapters:
        assertionMsg( false, "this branch/state should never be reached" ); 
        break;
      case peanoclaw::records::RepositoryState::RunOnAllNodes:
        assertionMsg( false, "this branch/state should never be reached" ); 
        break;
      case peanoclaw::records::RepositoryState::ReadCheckpoint:
        assertionMsg( false, "not implemented yet" );
        break;
      case peanoclaw::records::RepositoryState::WriteCheckpoint:
        assertionMsg( false, "not implemented yet" );
        break;
    }
  }
    
  #ifdef Parallel
  if (_solverState.isJoiningWithMaster()) {
    _repositoryState.setAction( peanoclaw::records::RepositoryState::Terminate );
  }
  #endif
}

 void peanoclaw::repositories::RepositoryArrayStack::switchToInitialiseGrid() { _repositoryState.setAction(peanoclaw::records::RepositoryState::UseAdapterInitialiseGrid); }
 void peanoclaw::repositories::RepositoryArrayStack::switchToInitialiseAndValidateGrid() { _repositoryState.setAction(peanoclaw::records::RepositoryState::UseAdapterInitialiseAndValidateGrid); }
 void peanoclaw::repositories::RepositoryArrayStack::switchToPlot() { _repositoryState.setAction(peanoclaw::records::RepositoryState::UseAdapterPlot); }
 void peanoclaw::repositories::RepositoryArrayStack::switchToPlotAndValidateGrid() { _repositoryState.setAction(peanoclaw::records::RepositoryState::UseAdapterPlotAndValidateGrid); }
 void peanoclaw::repositories::RepositoryArrayStack::switchToRemesh() { _repositoryState.setAction(peanoclaw::records::RepositoryState::UseAdapterRemesh); }
 void peanoclaw::repositories::RepositoryArrayStack::switchToSolveTimestep() { _repositoryState.setAction(peanoclaw::records::RepositoryState::UseAdapterSolveTimestep); }
 void peanoclaw::repositories::RepositoryArrayStack::switchToSolveTimestepAndValidateGrid() { _repositoryState.setAction(peanoclaw::records::RepositoryState::UseAdapterSolveTimestepAndValidateGrid); }
 void peanoclaw::repositories::RepositoryArrayStack::switchToSolveTimestepAndPlot() { _repositoryState.setAction(peanoclaw::records::RepositoryState::UseAdapterSolveTimestepAndPlot); }
 void peanoclaw::repositories::RepositoryArrayStack::switchToSolveTimestepAndPlotAndValidateGrid() { _repositoryState.setAction(peanoclaw::records::RepositoryState::UseAdapterSolveTimestepAndPlotAndValidateGrid); }
 void peanoclaw::repositories::RepositoryArrayStack::switchToGatherCurrentSolution() { _repositoryState.setAction(peanoclaw::records::RepositoryState::UseAdapterGatherCurrentSolution); }
 void peanoclaw::repositories::RepositoryArrayStack::switchToGatherCurrentSolutionAndValidateGrid() { _repositoryState.setAction(peanoclaw::records::RepositoryState::UseAdapterGatherCurrentSolutionAndValidateGrid); }
 void peanoclaw::repositories::RepositoryArrayStack::switchToCleanup() { _repositoryState.setAction(peanoclaw::records::RepositoryState::UseAdapterCleanup); }



 bool peanoclaw::repositories::RepositoryArrayStack::isActiveAdapterInitialiseGrid() const { return _repositoryState.getAction() == peanoclaw::records::RepositoryState::UseAdapterInitialiseGrid; }
 bool peanoclaw::repositories::RepositoryArrayStack::isActiveAdapterInitialiseAndValidateGrid() const { return _repositoryState.getAction() == peanoclaw::records::RepositoryState::UseAdapterInitialiseAndValidateGrid; }
 bool peanoclaw::repositories::RepositoryArrayStack::isActiveAdapterPlot() const { return _repositoryState.getAction() == peanoclaw::records::RepositoryState::UseAdapterPlot; }
 bool peanoclaw::repositories::RepositoryArrayStack::isActiveAdapterPlotAndValidateGrid() const { return _repositoryState.getAction() == peanoclaw::records::RepositoryState::UseAdapterPlotAndValidateGrid; }
 bool peanoclaw::repositories::RepositoryArrayStack::isActiveAdapterRemesh() const { return _repositoryState.getAction() == peanoclaw::records::RepositoryState::UseAdapterRemesh; }
 bool peanoclaw::repositories::RepositoryArrayStack::isActiveAdapterSolveTimestep() const { return _repositoryState.getAction() == peanoclaw::records::RepositoryState::UseAdapterSolveTimestep; }
 bool peanoclaw::repositories::RepositoryArrayStack::isActiveAdapterSolveTimestepAndValidateGrid() const { return _repositoryState.getAction() == peanoclaw::records::RepositoryState::UseAdapterSolveTimestepAndValidateGrid; }
 bool peanoclaw::repositories::RepositoryArrayStack::isActiveAdapterSolveTimestepAndPlot() const { return _repositoryState.getAction() == peanoclaw::records::RepositoryState::UseAdapterSolveTimestepAndPlot; }
 bool peanoclaw::repositories::RepositoryArrayStack::isActiveAdapterSolveTimestepAndPlotAndValidateGrid() const { return _repositoryState.getAction() == peanoclaw::records::RepositoryState::UseAdapterSolveTimestepAndPlotAndValidateGrid; }
 bool peanoclaw::repositories::RepositoryArrayStack::isActiveAdapterGatherCurrentSolution() const { return _repositoryState.getAction() == peanoclaw::records::RepositoryState::UseAdapterGatherCurrentSolution; }
 bool peanoclaw::repositories::RepositoryArrayStack::isActiveAdapterGatherCurrentSolutionAndValidateGrid() const { return _repositoryState.getAction() == peanoclaw::records::RepositoryState::UseAdapterGatherCurrentSolutionAndValidateGrid; }
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
void peanoclaw::repositories::RepositoryArrayStack::runGlobalStep() {
  assertion(tarch::parallel::Node::getInstance().isGlobalMaster());

  peanoclaw::records::RepositoryState intermediateStateForWorkingNodes;
  intermediateStateForWorkingNodes.setAction( peanoclaw::records::RepositoryState::RunOnAllNodes );
  
  tarch::parallel::NodePool::getInstance().broadcastToWorkingNodes(
    intermediateStateForWorkingNodes,
    peano::parallel::SendReceiveBufferPool::getInstance().getIterationManagementTag()
  );
  tarch::parallel::NodePool::getInstance().activateIdleNodes();
}


peanoclaw::repositories::RepositoryArrayStack::ContinueCommand peanoclaw::repositories::RepositoryArrayStack::continueToIterate() {
  logTraceIn( "continueToIterate()" );

  assertion( !tarch::parallel::Node::getInstance().isGlobalMaster());

  ContinueCommand result;
  if ( _solverState.hasJoinedWithMaster() ) {
    result = Terminate;
  }
  else {
    int masterNode = tarch::parallel::Node::getInstance().getGlobalMasterRank();
    assertion( masterNode != -1 );

    _repositoryState.receive( masterNode, peano::parallel::SendReceiveBufferPool::getInstance().getIterationManagementTag(), true, ReceiveIterationControlMessagesBlocking );

    result = Continue;
    if (_repositoryState.getAction()==peanoclaw::records::RepositoryState::Terminate) {
      result = Terminate;
    } 
    if (_repositoryState.getAction()==peanoclaw::records::RepositoryState::RunOnAllNodes) {
      result = RunGlobalStep;
    } 
  }
   
  logTraceOutWith1Argument( "continueToIterate()", result );
  return result;
}
#endif


void peanoclaw::repositories::RepositoryArrayStack::logIterationStatistics() const {
  logInfo( "logIterationStatistics()", "|| adapter name \t || iterations \t || total CPU time [t]=s \t || average CPU time [t]=s \t || total user time [t]=s \t || average user time [t]=s  || CPU time properties  || user time properties " );  
   logInfo( "logIterationStatistics()", "| InitialiseGrid \t |  " << _measureInitialiseGridCPUTime.getNumberOfMeasurements() << " \t |  " << _measureInitialiseGridCPUTime.getAccumulatedValue() << " \t |  " << _measureInitialiseGridCPUTime.getValue()  << " \t |  " << _measureInitialiseGridCalendarTime.getAccumulatedValue() << " \t |  " << _measureInitialiseGridCalendarTime.getValue() << " \t |  " << _measureInitialiseGridCPUTime.toString() << " \t |  " << _measureInitialiseGridCalendarTime.toString() );
   logInfo( "logIterationStatistics()", "| InitialiseAndValidateGrid \t |  " << _measureInitialiseAndValidateGridCPUTime.getNumberOfMeasurements() << " \t |  " << _measureInitialiseAndValidateGridCPUTime.getAccumulatedValue() << " \t |  " << _measureInitialiseAndValidateGridCPUTime.getValue()  << " \t |  " << _measureInitialiseAndValidateGridCalendarTime.getAccumulatedValue() << " \t |  " << _measureInitialiseAndValidateGridCalendarTime.getValue() << " \t |  " << _measureInitialiseAndValidateGridCPUTime.toString() << " \t |  " << _measureInitialiseAndValidateGridCalendarTime.toString() );
   logInfo( "logIterationStatistics()", "| Plot \t |  " << _measurePlotCPUTime.getNumberOfMeasurements() << " \t |  " << _measurePlotCPUTime.getAccumulatedValue() << " \t |  " << _measurePlotCPUTime.getValue()  << " \t |  " << _measurePlotCalendarTime.getAccumulatedValue() << " \t |  " << _measurePlotCalendarTime.getValue() << " \t |  " << _measurePlotCPUTime.toString() << " \t |  " << _measurePlotCalendarTime.toString() );
   logInfo( "logIterationStatistics()", "| PlotAndValidateGrid \t |  " << _measurePlotAndValidateGridCPUTime.getNumberOfMeasurements() << " \t |  " << _measurePlotAndValidateGridCPUTime.getAccumulatedValue() << " \t |  " << _measurePlotAndValidateGridCPUTime.getValue()  << " \t |  " << _measurePlotAndValidateGridCalendarTime.getAccumulatedValue() << " \t |  " << _measurePlotAndValidateGridCalendarTime.getValue() << " \t |  " << _measurePlotAndValidateGridCPUTime.toString() << " \t |  " << _measurePlotAndValidateGridCalendarTime.toString() );
   logInfo( "logIterationStatistics()", "| Remesh \t |  " << _measureRemeshCPUTime.getNumberOfMeasurements() << " \t |  " << _measureRemeshCPUTime.getAccumulatedValue() << " \t |  " << _measureRemeshCPUTime.getValue()  << " \t |  " << _measureRemeshCalendarTime.getAccumulatedValue() << " \t |  " << _measureRemeshCalendarTime.getValue() << " \t |  " << _measureRemeshCPUTime.toString() << " \t |  " << _measureRemeshCalendarTime.toString() );
   logInfo( "logIterationStatistics()", "| SolveTimestep \t |  " << _measureSolveTimestepCPUTime.getNumberOfMeasurements() << " \t |  " << _measureSolveTimestepCPUTime.getAccumulatedValue() << " \t |  " << _measureSolveTimestepCPUTime.getValue()  << " \t |  " << _measureSolveTimestepCalendarTime.getAccumulatedValue() << " \t |  " << _measureSolveTimestepCalendarTime.getValue() << " \t |  " << _measureSolveTimestepCPUTime.toString() << " \t |  " << _measureSolveTimestepCalendarTime.toString() );
   logInfo( "logIterationStatistics()", "| SolveTimestepAndValidateGrid \t |  " << _measureSolveTimestepAndValidateGridCPUTime.getNumberOfMeasurements() << " \t |  " << _measureSolveTimestepAndValidateGridCPUTime.getAccumulatedValue() << " \t |  " << _measureSolveTimestepAndValidateGridCPUTime.getValue()  << " \t |  " << _measureSolveTimestepAndValidateGridCalendarTime.getAccumulatedValue() << " \t |  " << _measureSolveTimestepAndValidateGridCalendarTime.getValue() << " \t |  " << _measureSolveTimestepAndValidateGridCPUTime.toString() << " \t |  " << _measureSolveTimestepAndValidateGridCalendarTime.toString() );
   logInfo( "logIterationStatistics()", "| SolveTimestepAndPlot \t |  " << _measureSolveTimestepAndPlotCPUTime.getNumberOfMeasurements() << " \t |  " << _measureSolveTimestepAndPlotCPUTime.getAccumulatedValue() << " \t |  " << _measureSolveTimestepAndPlotCPUTime.getValue()  << " \t |  " << _measureSolveTimestepAndPlotCalendarTime.getAccumulatedValue() << " \t |  " << _measureSolveTimestepAndPlotCalendarTime.getValue() << " \t |  " << _measureSolveTimestepAndPlotCPUTime.toString() << " \t |  " << _measureSolveTimestepAndPlotCalendarTime.toString() );
   logInfo( "logIterationStatistics()", "| SolveTimestepAndPlotAndValidateGrid \t |  " << _measureSolveTimestepAndPlotAndValidateGridCPUTime.getNumberOfMeasurements() << " \t |  " << _measureSolveTimestepAndPlotAndValidateGridCPUTime.getAccumulatedValue() << " \t |  " << _measureSolveTimestepAndPlotAndValidateGridCPUTime.getValue()  << " \t |  " << _measureSolveTimestepAndPlotAndValidateGridCalendarTime.getAccumulatedValue() << " \t |  " << _measureSolveTimestepAndPlotAndValidateGridCalendarTime.getValue() << " \t |  " << _measureSolveTimestepAndPlotAndValidateGridCPUTime.toString() << " \t |  " << _measureSolveTimestepAndPlotAndValidateGridCalendarTime.toString() );
   logInfo( "logIterationStatistics()", "| GatherCurrentSolution \t |  " << _measureGatherCurrentSolutionCPUTime.getNumberOfMeasurements() << " \t |  " << _measureGatherCurrentSolutionCPUTime.getAccumulatedValue() << " \t |  " << _measureGatherCurrentSolutionCPUTime.getValue()  << " \t |  " << _measureGatherCurrentSolutionCalendarTime.getAccumulatedValue() << " \t |  " << _measureGatherCurrentSolutionCalendarTime.getValue() << " \t |  " << _measureGatherCurrentSolutionCPUTime.toString() << " \t |  " << _measureGatherCurrentSolutionCalendarTime.toString() );
   logInfo( "logIterationStatistics()", "| GatherCurrentSolutionAndValidateGrid \t |  " << _measureGatherCurrentSolutionAndValidateGridCPUTime.getNumberOfMeasurements() << " \t |  " << _measureGatherCurrentSolutionAndValidateGridCPUTime.getAccumulatedValue() << " \t |  " << _measureGatherCurrentSolutionAndValidateGridCPUTime.getValue()  << " \t |  " << _measureGatherCurrentSolutionAndValidateGridCalendarTime.getAccumulatedValue() << " \t |  " << _measureGatherCurrentSolutionAndValidateGridCalendarTime.getValue() << " \t |  " << _measureGatherCurrentSolutionAndValidateGridCPUTime.toString() << " \t |  " << _measureGatherCurrentSolutionAndValidateGridCalendarTime.toString() );
   logInfo( "logIterationStatistics()", "| Cleanup \t |  " << _measureCleanupCPUTime.getNumberOfMeasurements() << " \t |  " << _measureCleanupCPUTime.getAccumulatedValue() << " \t |  " << _measureCleanupCPUTime.getValue()  << " \t |  " << _measureCleanupCalendarTime.getAccumulatedValue() << " \t |  " << _measureCleanupCalendarTime.getValue() << " \t |  " << _measureCleanupCPUTime.toString() << " \t |  " << _measureCleanupCalendarTime.toString() );

}


void peanoclaw::repositories::RepositoryArrayStack::clearIterationStatistics() {
   _measureInitialiseGridCPUTime.erase();
   _measureInitialiseAndValidateGridCPUTime.erase();
   _measurePlotCPUTime.erase();
   _measurePlotAndValidateGridCPUTime.erase();
   _measureRemeshCPUTime.erase();
   _measureSolveTimestepCPUTime.erase();
   _measureSolveTimestepAndValidateGridCPUTime.erase();
   _measureSolveTimestepAndPlotCPUTime.erase();
   _measureSolveTimestepAndPlotAndValidateGridCPUTime.erase();
   _measureGatherCurrentSolutionCPUTime.erase();
   _measureGatherCurrentSolutionAndValidateGridCPUTime.erase();
   _measureCleanupCPUTime.erase();

   _measureInitialiseGridCalendarTime.erase();
   _measureInitialiseAndValidateGridCalendarTime.erase();
   _measurePlotCalendarTime.erase();
   _measurePlotAndValidateGridCalendarTime.erase();
   _measureRemeshCalendarTime.erase();
   _measureSolveTimestepCalendarTime.erase();
   _measureSolveTimestepAndValidateGridCalendarTime.erase();
   _measureSolveTimestepAndPlotCalendarTime.erase();
   _measureSolveTimestepAndPlotAndValidateGridCalendarTime.erase();
   _measureGatherCurrentSolutionCalendarTime.erase();
   _measureGatherCurrentSolutionAndValidateGridCalendarTime.erase();
   _measureCleanupCalendarTime.erase();

}
