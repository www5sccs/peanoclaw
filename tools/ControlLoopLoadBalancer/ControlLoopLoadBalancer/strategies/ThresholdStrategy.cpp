#include "tarch/parallel/Node.h"

#include "ControlLoopLoadBalancer/strategies/ThresholdStrategy.h" 
#include "ControlLoopLoadBalancer/Reductions.h"

mpibalancing::ControlLoopLoadBalancer::strategies::ThresholdStrategy::ThresholdStrategy(HistorySet<int, WorkerData>& workerHistorySet)
    : _workerHistorySet(workerHistorySet)
{
}

mpibalancing::ControlLoopLoadBalancer::strategies::ThresholdStrategy::~ThresholdStrategy() {}

int mpibalancing::ControlLoopLoadBalancer::strategies::ThresholdStrategy::run( int worker ) {
    int result = peano::parallel::loadbalancing::Continue;
    const WorkerData& workerData = _workerHistorySet.getHistory(worker).getPastItem(1);
    
    mpibalancing::ControlLoopLoadBalancer::reductions::ThresholdSumReduction total_sum(0.0,0.0);
    _workerHistorySet.reduce(total_sum);

    double average_localWorkload = -1;
    double average_totalWorkload = -1;
    
    if (total_sum.getProcessesWithUpperLocalWorkload() > 0) {
        average_localWorkload = total_sum.getSumOfUpperLocalWorkload() / total_sum.getProcessesWithUpperLocalWorkload();
    }
  
    if (total_sum.getProcessesWithUpperTotalWorkload() > 0) {
        average_totalWorkload = total_sum.getSumOfUpperTotalWorkload() / total_sum.getProcessesWithUpperTotalWorkload();
    }

    mpibalancing::ControlLoopLoadBalancer::reductions::ThresholdSumReduction threshold_sum(average_localWorkload,average_totalWorkload);
    _workerHistorySet.reduce(threshold_sum);

    double upperThreshold_localWorkload = -1;
    double lowerThreshold_localWorkload = -1;
    double upperThreshold_totalWorkload = -1;
    double lowerThreshold_totalWorkload = -1;

    if (threshold_sum.getProcessesWithUpperLocalWorkload() > 0) {
        upperThreshold_localWorkload = threshold_sum.getSumOfUpperLocalWorkload() / threshold_sum.getProcessesWithUpperLocalWorkload();
    }

    if (threshold_sum.getProcessesWithLowerLocalWorkload() > 0) {
        lowerThreshold_localWorkload = threshold_sum.getSumOfLowerLocalWorkload() / threshold_sum.getProcessesWithLowerLocalWorkload();
    }
 
    if (threshold_sum.getProcessesWithUpperTotalWorkload() > 0) {
        upperThreshold_totalWorkload = threshold_sum.getSumOfUpperTotalWorkload() / threshold_sum.getProcessesWithUpperTotalWorkload();
    }

    if (threshold_sum.getProcessesWithLowerTotalWorkload() > 0) {
        lowerThreshold_totalWorkload = threshold_sum.getSumOfLowerTotalWorkload() / threshold_sum.getProcessesWithLowerTotalWorkload();
    }

    // total workload thresholds
    bool aboveUpperTotalThreshold = workerData.getTotalWorkload() >= upperThreshold_totalWorkload;
    bool betweenUpperTotalThresholdAndAverage = workerData.getTotalWorkload() < upperThreshold_totalWorkload && 
                                                workerData.getTotalWorkload() >= average_totalWorkload;
    bool betweenTotalAverageAndLowerThreshold = workerData.getTotalWorkload() >= lowerThreshold_totalWorkload && 
                                                workerData.getTotalWorkload() < average_totalWorkload;
    bool belowLowerTotalThreshold = workerData.getTotalWorkload() < lowerThreshold_totalWorkload;


    // local workload thresholds
    bool aboveUpperLocalThreshold = workerData.getLocalWorkload() >= upperThreshold_localWorkload;
    bool betweenUpperLocalThresholdAndAverage = workerData.getLocalWorkload() < upperThreshold_localWorkload && 
                                                workerData.getLocalWorkload() >= average_localWorkload;
    bool betweenLocalAverageAndLowerThreshold = workerData.getLocalWorkload() >= lowerThreshold_localWorkload && 
                                                workerData.getLocalWorkload() < average_localWorkload;
    bool belowLowerLocalThreshold = workerData.getLocalWorkload() < lowerThreshold_localWorkload;
 
    //std::cout << "total_thresholds: " << lowerThreshold_totalWorkload << " " << average_totalWorkload << " " << upperThreshold_totalWorkload << std::endl;
    //std::cout << "local_thresholds: " << lowerThreshold_localWorkload << " " << average_localWorkload << " " << upperThreshold_localWorkload << std::endl;
    //std::cout << "worker load: " << workerData.getLocalWorkload() << " " << workerData.getTotalWorkload() << std::endl;

    if (aboveUpperTotalThreshold) {
        //std::cout << "upper total threshold" << std::endl;
        if (aboveUpperLocalThreshold && (workerData.getLocalWorkload() > average_localWorkload * 8 || workerData.getLocalWorkload() == workerData.getTotalWorkload() ) ) {
            result = peano::parallel::loadbalancing::ForkGreedy;
        } else if (betweenUpperLocalThresholdAndAverage && (workerData.getLocalWorkload() * 8 > average_localWorkload * 9 || workerData.getLocalWorkload() == workerData.getTotalWorkload() ) ) {
            result = peano::parallel::loadbalancing::ForkOnce;
        } else if (betweenLocalAverageAndLowerThreshold) {
            result = peano::parallel::loadbalancing::Continue;
        } else if (belowLowerLocalThreshold) {
            result = peano::parallel::loadbalancing::ContinueButTryToJoinWorkers;
        }
    } else if (betweenUpperTotalThresholdAndAverage) {
        //std::cout << "between average and upper total threshold" << std::endl;

        if (aboveUpperLocalThreshold && ( workerData.getLocalWorkload() * 8 >= average_localWorkload * 9  || workerData.getLocalWorkload() == workerData.getTotalWorkload() ) ) {
            result = peano::parallel::loadbalancing::ForkOnce;
        } else if (betweenUpperLocalThresholdAndAverage) {
            result = peano::parallel::loadbalancing::Continue;
        } else if (betweenLocalAverageAndLowerThreshold) {
            result = peano::parallel::loadbalancing::ContinueButTryToJoinWorkers;
        } else if (belowLowerLocalThreshold) {
            result = peano::parallel::loadbalancing::Join;
        }
    } else if (betweenTotalAverageAndLowerThreshold) {
        //std::cout << "between total lower total threshold and total average" << std::endl;

        if (aboveUpperLocalThreshold) {
            result = peano::parallel::loadbalancing::Continue;
        } else if (betweenUpperLocalThresholdAndAverage) {
            result = peano::parallel::loadbalancing::ContinueButTryToJoinWorkers;
        } else if (betweenLocalAverageAndLowerThreshold) {
            result = peano::parallel::loadbalancing::Join;
        } else if (belowLowerLocalThreshold) {
            result = peano::parallel::loadbalancing::Join;
        }
    } else if (belowLowerTotalThreshold) {
        //std::cout << "below lower total threshold" << std::endl;

        if (aboveUpperLocalThreshold) {
            result = peano::parallel::loadbalancing::ContinueButTryToJoinWorkers;
        } else if (betweenUpperLocalThresholdAndAverage) {
            result = peano::parallel::loadbalancing::Join;
        } else if (betweenLocalAverageAndLowerThreshold) {
            result = peano::parallel::loadbalancing::Join;
        } else if (belowLowerLocalThreshold) {
            result = peano::parallel::loadbalancing::Join;
        }
    } else {
        // is there a case i am not aware of?
        result = peano::parallel::loadbalancing::Continue;
    }

    if (result == peano::parallel::loadbalancing::Continue) {
        if (workerData.getLocalWorkload() > 1000) {
            result = peano::parallel::loadbalancing::ForkGreedy;
        } 
    }

    // special case: very first worker
    if (tarch::parallel::Node::getInstance().isGlobalMaster()) {
        if (workerData.getLocalWorkload() == workerData.getTotalWorkload()) {
            result = peano::parallel::loadbalancing::ForkAllChildrenAndBecomeAdministrativeRank;
        } else if (workerData.getLocalWorkload() * 10 >= 9 * workerData.getTotalWorkload() ) {
            result = peano::parallel::loadbalancing::ForkGreedy;
        } else {
            result = peano::parallel::loadbalancing::Continue;
        }
    }
    return result;
}
