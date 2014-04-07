#include "ControlLoopLoadBalancer/ControlLoopLoadBalancer.h"

using namespace mpibalancing::ControlLoopLoadBalancer;

tarch::logging::Log mpibalancing::ControlLoopLoadBalancer::ControlLoopLoadBalancer::_log("mpibalancing::ControlLoopLoadBalancer::ControlLoopLoadBalancer");

mpibalancing::ControlLoopLoadBalancer::ControlLoopLoadBalancer::ControlLoopLoadBalancer() {}
mpibalancing::ControlLoopLoadBalancer::ControlLoopLoadBalancer::~ControlLoopLoadBalancer() {}

void mpibalancing::ControlLoopLoadBalancer::ControlLoopLoadBalancer::setJoinsAllowed(bool flag) {
    getGridStateHistory().getCurrentItem().setJoinsAllowed(flag);
}

void mpibalancing::ControlLoopLoadBalancer::ControlLoopLoadBalancer::setForkFailed(bool flag) {
    getGridStateHistory().getCurrentItem().setForkFailed(flag);
}

void mpibalancing::ControlLoopLoadBalancer::ControlLoopLoadBalancer::receivedStartCommand( int commandFromMaster ) {
    // start a new time cycle!
    advanceInTime();

    // give implementations time to update their strategy configuration
    updateStrategies();

    // as this is the only information provided by the Oracle of our master
    // we have to set both commands.
    getMasterHistory().getCurrentItem().setDesiredLoadBalancingCommand(commandFromMaster);
    getMasterHistory().getCurrentItem().setActualLoadBalancingCommand(commandFromMaster);

}

int mpibalancing::ControlLoopLoadBalancer::ControlLoopLoadBalancer::getCommandForWorker( int workerRank, bool forkIsAllowed, bool joinIsAllowed ) {
    // first check if this is the first occurence of this worker
    // if it does not exist add a new instance
    // otherwise get previous instance
    
    // our default StdHistoryMap allows this in a very convenient way
    
    History<WorkerData>& workerHistory = getWorkerHistorySet().getHistory(workerRank);
    WorkerData& currentInformation =  workerHistory.getCurrentItem();

    // just in case that this worker is new
    currentInformation.setRank(workerRank);

    currentInformation.setForkAllowed(forkIsAllowed);
    currentInformation.setJoinAllowed(joinIsAllowed);

    int desiredLoadBalancingCommand = getStrategy().run(workerRank);
    currentInformation.setDesiredLoadBalancingCommand(desiredLoadBalancingCommand);

    int actualLoadBalancingCommand = getFilterStrategy().run(workerRank, desiredLoadBalancingCommand);
    currentInformation.setActualLoadBalancingCommand(actualLoadBalancingCommand);

    logInfo("getCommandForWorker()", "worker " << workerRank << " | desired = " << desiredLoadBalancingCommand << " | actual = " << actualLoadBalancingCommand);
    return actualLoadBalancingCommand;
}

void mpibalancing::ControlLoopLoadBalancer::ControlLoopLoadBalancer::receivedTerminateCommand(
  int     workerRank,
  double  waitedTime,
  double  workerNumberOfInnerVertices,
  double  workerNumberOfBoundaryVertices,
  double  workerNumberOfOuterVertices,
  double  workerNumberOfInnerCells,
  double  workerNumberOfOuterCells,
  int     workerMaxLevel,
  double  workerLocalWorkload,
  double  workerTotalWorkload,
  double  workerMaxWorkload,
  double  workerMinWorkload,
  int     currentLevel,
  double  parentCellLocalWorkload,
  const tarch::la::Vector<DIMENSIONS,double>& boundingBoxOffset,
  const tarch::la::Vector<DIMENSIONS,double>& boundingBoxSize,
  bool workerCouldNotEraseDueToDecomposition
) {
    History<WorkerData>& workerHistory = getWorkerHistorySet().getHistory(workerRank);
    WorkerData& currentInformation =  workerHistory.getCurrentItem();
    const WorkerData& pastInformation =  workerHistory.getPastItem(1);

 
    if (currentInformation.getActualLoadBalancingCommand() == peano::parallel::loadbalancing::Join) {
        // TODO: is it actually true that we are able to join worker after only one iteration?
 
        // This was the last answer of this worker -> delete its history
        getWorkerHistorySet().deleteHistory(workerRank);
    } else {
        currentInformation.setWaitedTime(waitedTime);
        currentInformation.setNumberOfInnerVertices(workerNumberOfInnerVertices);
        currentInformation.setNumberOfBoundaryVertices(workerNumberOfBoundaryVertices);
        currentInformation.setNumberOfOuterVertices(workerNumberOfOuterVertices);
        currentInformation.setNumberOfInnerCells(workerNumberOfInnerCells);
        currentInformation.setNumberOfOuterCells(workerNumberOfOuterCells);
        currentInformation.setMaxLevel(workerMaxLevel);
        currentInformation.setMaxWorkload(workerMaxWorkload);
        currentInformation.setMinWorkload(workerMinWorkload);
        currentInformation.setCurrentLevel(currentLevel);
        currentInformation.setLocalWorkload(workerLocalWorkload);
        currentInformation.setTotalWorkload(workerTotalWorkload);
        currentInformation.setParentCellLocalWorkload(parentCellLocalWorkload);
        currentInformation.setBoundingBoxOffset(boundingBoxOffset);
        currentInformation.setBoundingBoxSize(boundingBoxSize);
        currentInformation.setCouldNotEraseDueToDecompositionFlag(workerCouldNotEraseDueToDecomposition);
    }
}

void mpibalancing::ControlLoopLoadBalancer::ControlLoopLoadBalancer::updateStrategies(void) {}

void mpibalancing::ControlLoopLoadBalancer::ControlLoopLoadBalancer::advanceInTime(void) {
    getWorkerHistorySet().advanceInTime();
    getMasterHistory().advanceInTime();
    getGridStateHistory().advanceInTime();
}

void mpibalancing::ControlLoopLoadBalancer::ControlLoopLoadBalancer::reset(void) {
    getWorkerHistorySet().reset();
    getMasterHistory().reset();
    getGridStateHistory().reset();
}
