#include "peano/parallel/loadbalancing/OracleForOnePhase.h"

#include "ControlLoopLoadBalancer/FilterStrategy.h"


mpibalancing::ControlLoopLoadBalancer::FilterStrategy::FilterStrategy() {}
mpibalancing::ControlLoopLoadBalancer::FilterStrategy::~FilterStrategy() {}

//-------------------------------------------------------------

mpibalancing::ControlLoopLoadBalancer::PassThroughFilterStrategy::PassThroughFilterStrategy() {}
mpibalancing::ControlLoopLoadBalancer::PassThroughFilterStrategy::~PassThroughFilterStrategy() {}

int mpibalancing::ControlLoopLoadBalancer::PassThroughFilterStrategy::run( int workerRank, int desiredCommand ) {
    return desiredCommand;
}

//-------------------------------------------------------------

tarch::logging::Log mpibalancing::ControlLoopLoadBalancer::BasicFilterStrategy::_log("mpibalancing::ControlLoopLoadBalancer::BasicFilterStrategy");

mpibalancing::ControlLoopLoadBalancer::BasicFilterStrategy::BasicFilterStrategy(
        History<WorkerData>& masterHistory, 
        HistorySet<int, WorkerData>& workerHistorySet,
        History<GridStateData>& gridStateHistory
) : _masterHistory(masterHistory), 
    _workerHistorySet(workerHistorySet), 
    _gridStateHistory(gridStateHistory) 
{
}

mpibalancing::ControlLoopLoadBalancer::BasicFilterStrategy::~BasicFilterStrategy() {}

int mpibalancing::ControlLoopLoadBalancer::BasicFilterStrategy::run( int worker, int desiredCommand ) {
    int result = desiredCommand;

    const WorkerData& masterData = _masterHistory.getCurrentItem();
    const GridStateData& gridStateData = _gridStateHistory.getPastItem(1);
    const WorkerData& workerData = _workerHistorySet.getHistory(worker).getCurrentItem();
    const WorkerData& pastWorkerData = _workerHistorySet.getHistory(worker).getPastItem(1);

    const bool joinIsPossible = workerData.isJoinAllowed()
                                     && gridStateData.isGridBalanced() 
                                     && !gridStateData.subWorkerIsInvolvedInJoinOrFork()
                                     && gridStateData.areJoinsAllowed()
                                     ;
    const bool forkIsPossible = workerData.isForkAllowed() 
                                   && !gridStateData.hasForkFailed()
                                   && !gridStateData.subWorkerIsInvolvedInJoinOrFork()
                                   && gridStateData.isGridBalanced()
                                   && gridStateData.isGridStationary()
                                   && !gridStateData.couldNotEraseDueToDecomposition()
                                   && !workerData.getCouldNotEraseDueToDecompositionFlag()
                                   && masterData.getActualLoadBalancingCommand() != peano::parallel::loadbalancing::ContinueButTryToJoinWorkers
                                   ;

    if (!joinIsPossible && desiredCommand == peano::parallel::loadbalancing::Join)  {
        logInfo("run(worker,desiredcommand)", "could not join worker " << worker << " due to " 
                  << "(isJoinAllowed=" << workerData.isJoinAllowed() 
                  << ", areJoinsAllowed=" << gridStateData.areJoinsAllowed() 
                  << ")");
        result = peano::parallel::loadbalancing::ContinueButTryToJoinWorkers;
    }

    if (!forkIsPossible && desiredCommand >= peano::parallel::loadbalancing::ForkOnce) {
        logInfo("run(worker,desiredcommand)", "could not fork worker " << worker << " due to " 
                  << "(isForkAllowed=" << workerData.isForkAllowed() 
                  << ", hasForkFailed=" << gridStateData.hasForkFailed()
                  << ", isGridBalanced=" << gridStateData.isGridBalanced()
                  << ", ContinueButTryToJoin from master=" << (masterData.getActualLoadBalancingCommand() == peano::parallel::loadbalancing::ContinueButTryToJoinWorkers)
                  << ")" 
                );
        
        if (masterData.getActualLoadBalancingCommand() == peano::parallel::loadbalancing::ContinueButTryToJoinWorkers) {
           result = peano::parallel::loadbalancing::ContinueButTryToJoinWorkers;
        } else {
           result = peano::parallel::loadbalancing::Continue;
        }
    }

    //if (result == peano::parallel::loadbalancing::Join) {
    //    std::cout << "||||||||| would have joined! DENIED! |||||||||||||" << std::endl;
    //    result = peano::parallel::loadbalancing::ContinueButTryToJoinWorkers;
    //}
    return result;
}
