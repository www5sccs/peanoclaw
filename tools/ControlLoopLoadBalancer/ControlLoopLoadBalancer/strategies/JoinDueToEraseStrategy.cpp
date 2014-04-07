#include "ControlLoopLoadBalancer/strategies/JoinDueToEraseStrategy.h"
#include "ControlLoopLoadBalancer/Reductions.h"

tarch::logging::Log mpibalancing::ControlLoopLoadBalancer::strategies::JoinDueToEraseStrategy::JoinDueToEraseStrategy::_log(
    "mpibalancing::ControlLoopLoadBalancer::strategies::JoinDueToEraseStrategy::JoinDueToEraseStrategy"
);

mpibalancing::ControlLoopLoadBalancer::strategies::JoinDueToEraseStrategy::JoinDueToEraseStrategy(
    History<WorkerData>& masterHistory, 
    HistorySet<int, WorkerData>& workerHistorySet,
    History<GridStateData>& gridStateHistory
) : _masterHistory(masterHistory), 
    _workerHistorySet(workerHistorySet), 
    _gridStateHistory(gridStateHistory) 
{
}

mpibalancing::ControlLoopLoadBalancer::strategies::JoinDueToEraseStrategy::~JoinDueToEraseStrategy() {}

int mpibalancing::ControlLoopLoadBalancer::strategies::JoinDueToEraseStrategy::run(int worker) {
    int result = peano::parallel::loadbalancing::ContinueButTryToJoinWorkers;
    const WorkerData& workerData = _workerHistorySet.getHistory(worker).getPastItem(1);
    const WorkerData& masterData = _masterHistory.getCurrentItem();
    const GridStateData& gridStateData = _gridStateHistory.getPastItem(1);

    // TODO: documentation
    //if (tarch::parallel::Node::getInstance().isGlobalMaster()) {
    //    return peano::parallel::loadbalancing::Continue;
    //}

    // 1. check if there is at least one worker that complained about a failed erase attempt.
    mpibalancing::ControlLoopLoadBalancer::reductions::EraseIssueReduction eraseIssueReduction;
    _workerHistorySet.reduce(eraseIssueReduction);
 
    // determine further our maximum level overall workers and the deepest level
    mpibalancing::ControlLoopLoadBalancer::reductions::FindDeepestLevelReduction findDeepestLevelReduction;
    _workerHistorySet.reduce(findDeepestLevelReduction);
  
    // deepest level first
    mpibalancing::ControlLoopLoadBalancer::reductions::FindDeepestLevelReduction deepestLevelReduction;
    _workerHistorySet.reduce(deepestLevelReduction);

    // 2. check if the current worker had an issue during an erase attempt in the past iteration.
    if (eraseIssueReduction.workersCouldNotEraseDueToDecomposition() ) {
        logInfo("run()", "could not erase due to decomposition");
        std::cout << "could not erase due to decomposition" << std::endl;

        bool atLeastOneSubtreeCouldNotEraseDueToDecomposition = eraseIssueReduction.getSmallestSubtreeRank() != -1;
        bool atLeastOneWorkerCouldNotEraseDueToDecomposition = eraseIssueReduction.getSmallestWorkerRank() != -1;
 
            if (workerData.getCouldNotEraseDueToDecompositionFlag()) {
                result = peano::parallel::loadbalancing::Join;
            } else {
                result = peano::parallel::loadbalancing::ContinueButTryToJoinWorkers;
            }

#if 1

        if (gridStateData.isTraversalInverted()) {
            if (workerData.getRank() == eraseIssueReduction.getSmallestTroubleWorkerRank() ) {
                logInfo("run(worker)", "smallest worker " << workerData.getRank() 
                      << " [ " << workerData.getBoundingBoxOffset() << " ] - [ " 
                      << ( workerData.getBoundingBoxOffset() + workerData.getBoundingBoxSize() ) << ", levels=" << workerData.getCurrentLevel() << "-" << workerData.getMaxLevel()
                      << " ]  -> JOIN");
                // Beam the enterprise crew member back on board.
                result = peano::parallel::loadbalancing::Join;
            } else {
                logDebug("run(worker)", "rank " << workerData.getRank() << " waiting for smallest worker  " << eraseIssueReduction.getSmallestWorkerRank() << " -> ContinueButTryToJoinWorkers");
                result = peano::parallel::loadbalancing::ContinueButTryToJoinWorkers;
            }
      
            /*if (workerData.getRank() == deepestLevelReduction.getDeepestLevelRank() && workerData.getCouldNotEraseDueToDecompositionFlag() ) {
                logInfo("run()", "deepest level worker" << workerData.getRank() 
                          << " [ " << workerData.getBoundingBoxOffset() << " ] - [ " 
                          << ( workerData.getBoundingBoxOffset() + workerData.getBoundingBoxSize() ) << ", levels=" << workerData.getCurrentLevel() << "-" << workerData.getMaxLevel()
                          << " ]  -> JOIN");
                result = peano::parallel::loadbalancing::Join;
            } else {
                logDebug("run()", "rank " << workerData.getRank() << " waiting for deepest level worker  " << deepestLevelReduction.getDeepestLevelRank() << " -> ContinueButTryToJoinWorkers");
                result = peano::parallel::loadbalancing::ContinueButTryToJoinWorkers;
            }*/
          
        } else {
            if (workerData.getCouldNotEraseDueToDecompositionFlag()) {
                result = peano::parallel::loadbalancing::Join;
            } else {
                result = peano::parallel::loadbalancing::Continue;
            }

            /*if (workerData.getRank() == eraseIssueReduction.getLargestWorkerRank()) {
                logInfo("run(worker)", "largest worker " << workerData.getRank() 
                      << " [ " << workerData.getBoundingBoxOffset() << " ] - [ " 
                      << ( workerData.getBoundingBoxOffset() + workerData.getBoundingBoxSize() ) << ", levels=" << workerData.getCurrentLevel() << "-" << workerData.getMaxLevel()
                      << " ]  -> JOIN");
                // Beam the enterprise crew member back on board.
                result = peano::parallel::loadbalancing::Join;
            } else {
                logDebug("run(worker)", "rank " << workerData.getRank() << " waiting for largest worker  " << eraseIssueReduction.getLargestWorkerRank() << " -> ContinueButTryToJoinWorkers");
                result = peano::parallel::loadbalancing::ContinueButTryToJoinWorkers;
            }*/

            /*if (workerData.getRank() == deepestLevelReduction.getDeepestLevelRank()) {
                logInfo("run()", "deepest level worker" << workerData.getRank() 
                          << " [ " << workerData.getBoundingBoxOffset() << " ] - [ " 
                          << ( workerData.getBoundingBoxOffset() + workerData.getBoundingBoxSize() ) << ", levels=" << workerData.getCurrentLevel() << "-" << workerData.getMaxLevel()
                          << " ]  -> JOIN");
                result = peano::parallel::loadbalancing::ContinueButTryToJoinWorkers;
            } else {
                logDebug("run()", "rank " << workerData.getRank() << " waiting for deepest level worker  " << deepestLevelReduction.getDeepestLevelRank() << " -> ContinueButTryToJoinWorkers");
                result = peano::parallel::loadbalancing::ContinueButTryToJoinWorkers;
            }*/
        }
#endif

        result = peano::parallel::loadbalancing::Join;

    // 4. check if we got some issues ourself during past erase attempts.
#if 1
    } else if (gridStateData.couldNotEraseDueToDecomposition()) {
        if (workerData.getRank() == eraseIssueReduction.getSmallestWorkerRank() ) {
                logInfo("run(worker)", "smallest worker " << workerData.getRank() 
                      << " [ " << workerData.getBoundingBoxOffset() << " ] - [ " 
                      << ( workerData.getBoundingBoxOffset() + workerData.getBoundingBoxSize() ) << ", levels=" << workerData.getCurrentLevel() << "-" << workerData.getMaxLevel()
                      << " ]  -> JOIN");
                // Beam the enterprise crew member back on board.
            result = peano::parallel::loadbalancing::Join;
            //result = peano::parallel::loadbalancing::ContinueButTryToJoinWorkers;
        } else if (workerData.getRank() == deepestLevelReduction.getDeepestLevelRank()) {
            logInfo("run()", "deepest level " << workerData.getRank() 
                      << " [ " << workerData.getBoundingBoxOffset() << " ] - [ " 
                      << ( workerData.getBoundingBoxOffset() + workerData.getBoundingBoxSize() ) << ", levels=" << workerData.getCurrentLevel() << "-" << workerData.getMaxLevel()
                      << " ]  -> JOIN");
            result = peano::parallel::loadbalancing::Join;
            //result = peano::parallel::loadbalancing::ContinueButTryToJoinWorkers;
        } else {
            logDebug("run()", "rank " << workerData.getRank() << " waiting for deepest level worker  " << deepestLevelReduction.getDeepestLevelRank() << " -> ContinueButTryToJoinWorkers");
            result = peano::parallel::loadbalancing::ContinueButTryToJoinWorkers;
        }

        // 5. As we send these issues to the master anyway, let it try to handle it in some way.
        // We are done here!
#endif
    } 
    else {
        logInfo("run()", "everything fine here - no erase trouble here");

        if (masterData.getActualLoadBalancingCommand() == peano::parallel::loadbalancing::ContinueButTryToJoinWorkers) {

            if (workerData.getRank() == eraseIssueReduction.getSmallestWorkerRank()) {
                logInfo("run()", "joining smallest worker " << workerData.getRank() << " due to join request from master"
                      << " [ " << workerData.getBoundingBoxOffset() << " ] - [ " 
                      << ( workerData.getBoundingBoxOffset() + workerData.getBoundingBoxSize() ) << ", levels=" << workerData.getCurrentLevel() << "-" << workerData.getMaxLevel()
                      << " ]  -> JOIN");

                // Beam the enterprise crew member back on board.
                result = peano::parallel::loadbalancing::Join;
            } else if (workerData.getRank() == deepestLevelReduction.getDeepestLevelRank() ) {
                logInfo("run()", "deepest level worker" << workerData.getRank() 
                          << " [ " << workerData.getBoundingBoxOffset() << " ] - [ " 
                          << ( workerData.getBoundingBoxOffset() + workerData.getBoundingBoxSize() ) << ", levels=" << workerData.getCurrentLevel() << "-" << workerData.getMaxLevel()
                          << " ]  -> JOIN");
                result = peano::parallel::loadbalancing::Join;
                result = peano::parallel::loadbalancing::ContinueButTryToJoinWorkers;
            } else {
                logInfo("run()", "rank " << workerData.getRank() << " waiting for smallest or deepest level worker " 
                        <<  eraseIssueReduction.getSmallestWorkerRank() 
                        << "due to join request from master -> ContinueButTryToJoinWorkers");
                result = peano::parallel::loadbalancing::ContinueButTryToJoinWorkers;
            }

            // smallest trouble worker strategy did not work - deadloop
            result = peano::parallel::loadbalancing::Join;
            //result = peano::parallel::loadbalancing::ContinueButTryToJoinWorkers;
        } else {
            result = peano::parallel::loadbalancing::ForkAllChildrenAndBecomeAdministrativeRank;
        }
    }

    return result;
}


