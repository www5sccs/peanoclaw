#include "tarch/parallel/Node.h"

#include "ControlLoopLoadBalancer/Strategy.h"
#include "ControlLoopLoadBalancer/Reductions.h"

mpibalancing::ControlLoopLoadBalancer::Strategy::Strategy() {}
mpibalancing::ControlLoopLoadBalancer::Strategy::~Strategy() {}

// -----------------------------------------------------------------

mpibalancing::ControlLoopLoadBalancer::strategies::ForkGreedyStrategy::ForkGreedyStrategy() {}
mpibalancing::ControlLoopLoadBalancer::strategies::ForkGreedyStrategy::~ForkGreedyStrategy() {}
int mpibalancing::ControlLoopLoadBalancer::strategies::ForkGreedyStrategy::run ( int worker ) {
    return peano::parallel::loadbalancing::ForkAllChildrenAndBecomeAdministrativeRank;
}

// -----------------------------------------------------------------

mpibalancing::ControlLoopLoadBalancer::strategies::MaximumForkGreedyStrategy::MaximumForkGreedyStrategy(HistorySet<int, WorkerData>& workerHistorySet) 
    : _workerHistorySet(workerHistorySet) {}

mpibalancing::ControlLoopLoadBalancer::strategies::MaximumForkGreedyStrategy::~MaximumForkGreedyStrategy() {}

int mpibalancing::ControlLoopLoadBalancer::strategies::MaximumForkGreedyStrategy::run ( int worker ) {
    int result = peano::parallel::loadbalancing::Continue;

    // get our own history
    History<WorkerData>& ownHistory = _workerHistorySet.getHistory(worker);
    const WorkerData& pastData = ownHistory.getPastItem(1);
    if (pastData.getActualLoadBalancingCommand() != peano::parallel::loadbalancing::UndefinedLoadBalancingFlag) {
        // collect information about the maximum total workload   
        mpibalancing::ControlLoopLoadBalancer::reductions::MaximumTotalWorkloadReduction reductionOperator;
        _workerHistorySet.reduce(reductionOperator);

        // this worker is already known
        if (reductionOperator.getMaximumTotalWorkload() == pastData.getTotalWorkload()) {
            result = peano::parallel::loadbalancing::ForkAllChildrenAndBecomeAdministrativeRank;
        } else {
            result = peano::parallel::loadbalancing::ContinueButTryToJoinWorkers;
        }
    } else {
        // this worker is new
        result = peano::parallel::loadbalancing::Continue;
    }

    return result;
}


// -----------------------------------------------------------------

mpibalancing::ControlLoopLoadBalancer::strategies::ContinueStrategy::ContinueStrategy(bool forksAllowed)
{
    allowForks(forksAllowed);
}

mpibalancing::ControlLoopLoadBalancer::strategies::ContinueStrategy::~ContinueStrategy() {}

int mpibalancing::ControlLoopLoadBalancer::strategies::ContinueStrategy::run( int worker ) {
    int result = peano::parallel::loadbalancing::Continue;
    if (_forksAllowed) {
        result = peano::parallel::loadbalancing::Continue;
    } else {
        result = peano::parallel::loadbalancing::ContinueButTryToJoinWorkers;
    }
    return result;
}

void mpibalancing::ControlLoopLoadBalancer::strategies::ContinueStrategy::allowForks(bool flag) {
    _forksAllowed = flag;
}

// -----------------------------------------------------------------

mpibalancing::ControlLoopLoadBalancer::strategies::RetryStrategy::RetryStrategy(HistorySet<int, WorkerData>& workerHistorySet)
    : _workerHistorySet(workerHistorySet)
{
}

mpibalancing::ControlLoopLoadBalancer::strategies::RetryStrategy::~RetryStrategy() {}

int mpibalancing::ControlLoopLoadBalancer::strategies::RetryStrategy::run( int worker ) {
    int result = peano::parallel::loadbalancing::Continue;
    const WorkerData& workerData = _workerHistorySet.getHistory(worker).getPastItem(1);
    
    if (workerData.getDesiredLoadBalancingCommand() != workerData.getActualLoadBalancingCommand()) {
        result = workerData.getDesiredLoadBalancingCommand();
    } else {
        result = peano::parallel::loadbalancing::Continue;
    }
    return result;
}


// -----------------------------------------------------------------


// -----------------------------------------------------------------

mpibalancing::ControlLoopLoadBalancer::strategies::JoinDeepestLevelFirstStrategy::JoinDeepestLevelFirstStrategy(
            HistorySet<int, WorkerData>& workerHistorySet
) : _workerHistorySet(workerHistorySet) 
{}

mpibalancing::ControlLoopLoadBalancer::strategies::JoinDeepestLevelFirstStrategy::~JoinDeepestLevelFirstStrategy() {}

int mpibalancing::ControlLoopLoadBalancer::strategies::JoinDeepestLevelFirstStrategy::run( int worker ) {
    int result = peano::parallel::loadbalancing::Continue;
    const WorkerData& workerData = _workerHistorySet.getHistory(worker).getPastItem(1);

    mpibalancing::ControlLoopLoadBalancer::reductions::FindDeepestLevelReduction findDeepestLevelReduction;
    _workerHistorySet.reduce(findDeepestLevelReduction);

    if (workerData.getCurrentLevel() == findDeepestLevelReduction.getDeepestLevel() 
            && workerData.getLocalWorkload() == workerData.getTotalWorkload() // check if worker is actually joinable
    ) {
        result = peano::parallel::loadbalancing::Join;
    }
    return result;
}

