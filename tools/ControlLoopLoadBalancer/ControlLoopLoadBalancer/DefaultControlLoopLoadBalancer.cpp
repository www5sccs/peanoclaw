#include "ControlLoopLoadBalancer/DefaultControlLoopLoadBalancer.h"
#include "ControlLoopLoadBalancer/strategies/JoinDueToEraseStrategy.h"
#include "ControlLoopLoadBalancer/strategies/ThresholdStrategy.h"
#include "ControlLoopLoadBalancer/Reductions.h"

using namespace  mpibalancing::ControlLoopLoadBalancer;

tarch::logging::Log mpibalancing::ControlLoopLoadBalancer::DefaultStrategy::_log("mpibalancing::ControlLoopLoadBalancer::DefaultStrategy");

mpibalancing::ControlLoopLoadBalancer::DefaultStrategy::DefaultStrategy(
    History<WorkerData>& masterHistory, 
    HistorySet<int, WorkerData>& workerHistorySet,
    History<GridStateData>& gridStateHistory
) :
    _masterHistory(masterHistory), 
    _workerHistorySet(workerHistorySet), 
    _gridStateHistory(gridStateHistory)
{
}

mpibalancing::ControlLoopLoadBalancer::DefaultStrategy::~DefaultStrategy() {}

int mpibalancing::ControlLoopLoadBalancer::DefaultStrategy::run( int worker ) {
    int result = peano::parallel::loadbalancing::Continue;
    const WorkerData& workerData = _workerHistorySet.getHistory(worker).getPastItem(1);
    const GridStateData& gridStateData = _gridStateHistory.getPastItem(1);
    const WorkerData& masterData = _masterHistory.getCurrentItem();

    mpibalancing::ControlLoopLoadBalancer::strategies::RetryStrategy _retryStrategy(_workerHistorySet);
    mpibalancing::ControlLoopLoadBalancer::strategies::JoinDueToEraseStrategy _joinDueToEraseStrategy(_masterHistory, _workerHistorySet, _gridStateHistory);
    mpibalancing::ControlLoopLoadBalancer::strategies::JoinDeepestLevelFirstStrategy _joinDeepestLevelFirstStrategy(_workerHistorySet);
    mpibalancing::ControlLoopLoadBalancer::strategies::ForkGreedyStrategy _greedyStrategy;
    mpibalancing::ControlLoopLoadBalancer::strategies::MaximumForkGreedyStrategy _maxGreedyStrategy(_workerHistorySet);
    mpibalancing::ControlLoopLoadBalancer::strategies::ThresholdStrategy _thresholdStrategy(_workerHistorySet);

    if (workerData.getActualLoadBalancingCommand() == peano::parallel::loadbalancing::UndefinedLoadBalancingFlag) {
        logDebug("this worker has no history - it is probably a new one: ", worker);
        result = peano::parallel::loadbalancing::Continue;
    } else {
        /*if (!gridStateData.isGridStationary()) {
            result = peano::parallel::loadbalancing::Continue;
        } else*/
        {
            /*result = _retryStrategy.run(worker);
              if (result != peano::parallel::loadbalancing::Join) {
                result = _joinDueToEraseStrategy.run(worker);
            }*/
            mpibalancing::ControlLoopLoadBalancer::reductions::EraseIssueReduction eraseIssueReduction;
            _workerHistorySet.reduce(eraseIssueReduction);
 
            int threshold_result = _thresholdStrategy.run(worker);
            int join_result = _joinDueToEraseStrategy.run(worker);

            if (join_result == peano::parallel::loadbalancing::ForkGreedy) {
                result = threshold_result;
            } else {
                result = join_result;
            }
        }
    }
    return result;
}

// --------------------------------------------------------------------------------------------------------

tarch::logging::Log mpibalancing::ControlLoopLoadBalancer::DefaultControlLoopLoadBalancer::_log("mpibalancing::ControlLoopLoadBalancer::DefaultControlLoopLoadBalancer");

mpibalancing::ControlLoopLoadBalancer::DefaultControlLoopLoadBalancer::DefaultControlLoopLoadBalancer() :
    _strategy(_masterHistory,_workerHistorySet,_gridStateHistory),
    _filterStrategy(_masterHistory,_workerHistorySet,_gridStateHistory)
{
    _loadBalancingSuspended = false;
}

mpibalancing::ControlLoopLoadBalancer::DefaultControlLoopLoadBalancer::~DefaultControlLoopLoadBalancer() {}
 
HistorySet<int, WorkerData >& mpibalancing::ControlLoopLoadBalancer::DefaultControlLoopLoadBalancer::getWorkerHistorySet() {
    return _workerHistorySet;
}

History<WorkerData>& mpibalancing::ControlLoopLoadBalancer::DefaultControlLoopLoadBalancer::getMasterHistory() {
    return _masterHistory;
}

History<GridStateData>& mpibalancing::ControlLoopLoadBalancer::DefaultControlLoopLoadBalancer::getGridStateHistory() {
    return _gridStateHistory;
}

Strategy& mpibalancing::ControlLoopLoadBalancer::DefaultControlLoopLoadBalancer::getStrategy(void) {
    if (_loadBalancingSuspended) {
        return _continueStrategy;
    } else {
        return _strategy;
    }
}

FilterStrategy& mpibalancing::ControlLoopLoadBalancer::DefaultControlLoopLoadBalancer::getFilterStrategy(void) {
    return _filterStrategy;
}

void mpibalancing::ControlLoopLoadBalancer::DefaultControlLoopLoadBalancer::suspendLoadBalancing(bool flag) {
    _loadBalancingSuspended = flag;
}
