#include "peanoclaw/parallel/PeanoClawControlLoopLoadBalancer.h"
#include "ControlLoopLoadBalancer/strategies/JoinDueToEraseStrategy.h"
#include "ControlLoopLoadBalancer/strategies/ThresholdStrategy.h"
#include "ControlLoopLoadBalancer/Reductions.h"

using namespace  mpibalancing::ControlLoopLoadBalancer;

tarch::logging::Log peanoclaw::parallel::PeanoClawStrategy::_log( "peanoclaw::parallel::PeanoClawStrategy" );

peanoclaw::parallel::PeanoClawStrategy::PeanoClawStrategy(
    History<mpibalancing::ControlLoopLoadBalancer::WorkerData>& masterHistory,
    HistorySet<int, mpibalancing::ControlLoopLoadBalancer::WorkerData>& workerHistorySet,
    History<mpibalancing::ControlLoopLoadBalancer::GridStateData>& gridStateHistory
) :
    _masterHistory(masterHistory),
    _workerHistorySet(workerHistorySet),
    _gridStateHistory(gridStateHistory)
{
}

peanoclaw::parallel::PeanoClawStrategy::~PeanoClawStrategy() {}

int peanoclaw::parallel::PeanoClawStrategy::run( int worker ) {
    int result = peano::parallel::loadbalancing::Continue;
    const WorkerData& workerData = _workerHistorySet.getHistory(worker).getPastItem(1);
    const GridStateData& gridStateData = _gridStateHistory.getPastItem(1);
    //const WorkerData& masterData = _masterHistory.getCurrentItem();

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
        if (!gridStateData.isGridStationary()) {
            result = peano::parallel::loadbalancing::Continue;
        } else
        {
            /*result = _retryStrategy.run(worker);
              if (result != peano::parallel::loadbalancing::Join) {
                result = _joinDueToEraseStrategy.run(worker);
            }*/


#if 1
            mpibalancing::ControlLoopLoadBalancer::reductions::EraseIssueReduction eraseIssueReduction;
            _workerHistorySet.reduce(eraseIssueReduction);

            int join_result = _joinDueToEraseStrategy.run(worker);

            if (join_result == peano::parallel::loadbalancing::ForkAllChildrenAndBecomeAdministrativeRank) {
                result = _thresholdStrategy.run(worker);
            } else {
                result = join_result;
            }
#else
            result = _greedyStrategy.run(worker);
#endif
        }
    }
    return result;
}

// --------------------------------------------------------------------------------------------------------

tarch::logging::Log peanoclaw::parallel::PeanoClawControlLoopLoadBalancer::_log( "peanoclaw::parallel::PeanoClawControlLoopLoadBalancer" );

peanoclaw::parallel::PeanoClawControlLoopLoadBalancer::PeanoClawControlLoopLoadBalancer() :
    _strategy(_masterHistory,_workerHistorySet,_gridStateHistory),
    _filterStrategy(_masterHistory,_workerHistorySet,_gridStateHistory)
{
    _loadBalancingSuspended = false;
}

peanoclaw::parallel::PeanoClawControlLoopLoadBalancer::~PeanoClawControlLoopLoadBalancer() {}

HistorySet<int, WorkerData >& peanoclaw::parallel::PeanoClawControlLoopLoadBalancer::getWorkerHistorySet() {
    return _workerHistorySet;
}

History<WorkerData>& peanoclaw::parallel::PeanoClawControlLoopLoadBalancer::getMasterHistory() {
    return _masterHistory;
}

History<GridStateData>& peanoclaw::parallel::PeanoClawControlLoopLoadBalancer::getGridStateHistory() {
    return _gridStateHistory;
}

Strategy& peanoclaw::parallel::PeanoClawControlLoopLoadBalancer::getStrategy(void) {
    if (_loadBalancingSuspended) {
        return _continueStrategy;
    } else {
        return _strategy;
    }
}

FilterStrategy& peanoclaw::parallel::PeanoClawControlLoopLoadBalancer::getFilterStrategy(void) {
    return _filterStrategy;
}

void peanoclaw::parallel::PeanoClawControlLoopLoadBalancer::suspendLoadBalancing(bool flag) {
    _loadBalancingSuspended = flag;
}
