#include "ControlLoopLoadBalancer/Reductions.h"

mpibalancing::ControlLoopLoadBalancer::reductions::FindDeepestLevelReduction::FindDeepestLevelReduction() {
    _deepestLevelRank = -1;
    _deepestLevel = 0;
    _maxLevelRank = -1;
    _maxLevel = 0;
}

mpibalancing::ControlLoopLoadBalancer::reductions::FindDeepestLevelReduction::~FindDeepestLevelReduction() {}

void mpibalancing::ControlLoopLoadBalancer::reductions::FindDeepestLevelReduction::evaluate(
    const mpibalancing::ControlLoopLoadBalancer::History<mpibalancing::ControlLoopLoadBalancer::WorkerData>& data) 
{
   if (_deepestLevel < data.getPastItem(1).getCurrentLevel()) {
        _deepestLevelRank = data.getPastItem(1).getRank();
        _deepestLevel = data.getPastItem(1).getCurrentLevel();
   }

   if (_maxLevel < data.getPastItem(1).getMaxLevel()) {
        _maxLevelRank = data.getPastItem(1).getRank();
        _maxLevel = data.getPastItem(1).getMaxLevel();
   }
}

int mpibalancing::ControlLoopLoadBalancer::reductions::FindDeepestLevelReduction::getDeepestLevelRank() const {
    return _deepestLevelRank;
}

int mpibalancing::ControlLoopLoadBalancer::reductions::FindDeepestLevelReduction::getDeepestLevel() const {
    return _deepestLevel;
}

int mpibalancing::ControlLoopLoadBalancer::reductions::FindDeepestLevelReduction::getMaxLevelRank() const {
    return _maxLevelRank;
}

int mpibalancing::ControlLoopLoadBalancer::reductions::FindDeepestLevelReduction::getMaxLevel() const {
    return _maxLevel;
}

// -----------------------------------------------------------------------------------------------------------------------


mpibalancing::ControlLoopLoadBalancer::reductions::EraseIssueReduction::EraseIssueReduction() {
    _workersCouldNotEraseDueToDecomposition = false;

    _firstWorkerRank = -1;
    _lastWorkerRank = -1;

    _smallestWorkerRank = -1;
    _smallestNumberOfWorkerCells = 0.0;
  
    _smallestTroubleWorkerRank = -1;
    _smallestNumberOfTroubleWorkerCells = 0.0;
 
    _largestWorkerRank = -1;
    _largestNumberOfWorkerCells = 0.0;

    _smallestSubtreeRank = -1;
    _smallestNumberOfSubtreeCells = 0.0;

    _largestSubtreeRank = -1;
    _largestNumberOfSubtreeCells = 0.0;
}

mpibalancing::ControlLoopLoadBalancer::reductions::EraseIssueReduction::~EraseIssueReduction() {}

void mpibalancing::ControlLoopLoadBalancer::reductions::EraseIssueReduction::evaluate(const mpibalancing::ControlLoopLoadBalancer::History<mpibalancing::ControlLoopLoadBalancer::WorkerData>& data) {
    const WorkerData& workerData = data.getPastItem(1);
    const WorkerData& currentWorkerData = data.getPastItem(0);

    _workersCouldNotEraseDueToDecomposition |= workerData.getCouldNotEraseDueToDecompositionFlag();

    if (currentWorkerData.isJoinAllowed()) {
        if (_firstWorkerRank == -1) {
            _firstWorkerRank = workerData.getRank();
        }

        _lastWorkerRank = workerData.getRank();
    }
 
    if ( currentWorkerData.isJoinAllowed() 
            && ( _smallestWorkerRank == -1 || _smallestNumberOfTroubleWorkerCells > workerData.getLocalWorkload() ) ) {
         _smallestWorkerRank = workerData.getRank();
         _smallestNumberOfWorkerCells = workerData.getLocalWorkload();
    }

    if (workerData.getCouldNotEraseDueToDecompositionFlag()) {
        // determine very first worker in set which had trouble during an erase
        if ( currentWorkerData.isJoinAllowed() 
                && ( _smallestTroubleWorkerRank == -1 || _smallestNumberOfTroubleWorkerCells > workerData.getLocalWorkload() ) ) {
             _smallestTroubleWorkerRank = workerData.getRank();
             _smallestNumberOfTroubleWorkerCells = workerData.getLocalWorkload();
        }

        // determine very first worker in set which had trouble during an erase
        if ( currentWorkerData.isJoinAllowed()
                && ( _largestWorkerRank == -1 || _largestNumberOfWorkerCells <= workerData.getLocalWorkload() ) ) {
             _largestWorkerRank = workerData.getRank();
             _largestNumberOfWorkerCells = workerData.getLocalWorkload();
        }
 
     
        // determine that worker which had trouble during an erase and the smallest total workload
        if ( workerData.getTotalWorkload() != workerData.getLocalWorkload() 
                && ( _smallestSubtreeRank == -1 || _smallestNumberOfSubtreeCells > workerData.getTotalWorkload() ) ) {
             _smallestSubtreeRank = workerData.getRank();
             _smallestNumberOfSubtreeCells = workerData.getTotalWorkload();
        }

        // determine that worker which had trouble during an erase and the largest total workload
        if ( workerData.getTotalWorkload() != workerData.getLocalWorkload() 
                && ( _largestSubtreeRank == -1 || _largestNumberOfSubtreeCells <= workerData.getTotalWorkload() ) ) {
             _largestSubtreeRank = workerData.getRank();
             _largestNumberOfSubtreeCells = workerData.getTotalWorkload();
        }
     }
}

bool mpibalancing::ControlLoopLoadBalancer::reductions::EraseIssueReduction::workersCouldNotEraseDueToDecomposition() const {
    return _workersCouldNotEraseDueToDecomposition;
}
 
int mpibalancing::ControlLoopLoadBalancer::reductions::EraseIssueReduction::getFirstWorkerRank() const {
    return _firstWorkerRank;
}

int mpibalancing::ControlLoopLoadBalancer::reductions::EraseIssueReduction::getLastWorkerRank() const {
    return _lastWorkerRank;
}

int mpibalancing::ControlLoopLoadBalancer::reductions::EraseIssueReduction::getSmallestWorkerRank() const {
    return _smallestWorkerRank;
}

int mpibalancing::ControlLoopLoadBalancer::reductions::EraseIssueReduction::getSmallestTroubleWorkerRank() const {
    return _smallestTroubleWorkerRank;
}

int mpibalancing::ControlLoopLoadBalancer::reductions::EraseIssueReduction::getLargestWorkerRank() const {
    return _largestWorkerRank;
}

int mpibalancing::ControlLoopLoadBalancer::reductions::EraseIssueReduction::getSmallestSubtreeRank() const {
    return _smallestSubtreeRank;
}

int mpibalancing::ControlLoopLoadBalancer::reductions::EraseIssueReduction::getLargestSubtreeRank() const {
    return _largestSubtreeRank;
}

// -----------------------------------------------------------------------------------------------------------------------


mpibalancing::ControlLoopLoadBalancer::reductions::MaximumTotalWorkloadReduction::MaximumTotalWorkloadReduction() {
    _maximumTotalWorkload = 0.0;
}

mpibalancing::ControlLoopLoadBalancer::reductions::MaximumTotalWorkloadReduction::~MaximumTotalWorkloadReduction() {}

void mpibalancing::ControlLoopLoadBalancer::reductions::MaximumTotalWorkloadReduction::evaluate(const History<WorkerData>& data) {
    _maximumTotalWorkload = fmax(_maximumTotalWorkload, data.getPastItem(1).getTotalWorkload());
}

double mpibalancing::ControlLoopLoadBalancer::reductions::MaximumTotalWorkloadReduction::getMaximumTotalWorkload() const {
    return _maximumTotalWorkload;
}

// -----------------------------------------------------------------------------------------------------------------------

mpibalancing::ControlLoopLoadBalancer::reductions::EmergencyJoin::EmergencyJoin() {
    _workerRank = -1;
}

mpibalancing::ControlLoopLoadBalancer::reductions::EmergencyJoin::~EmergencyJoin() {}

void mpibalancing::ControlLoopLoadBalancer::reductions::EmergencyJoin::evaluate(
    const mpibalancing::ControlLoopLoadBalancer::History<mpibalancing::ControlLoopLoadBalancer::WorkerData>& data) 
{
    const WorkerData& workerData = data.getPastItem(1);

    if (!workerData.getCouldNotEraseDueToDecompositionFlag() 
            && _workerRank == -1
    ) {
        _workerRank = data.getPastItem(1).getRank();
    }
}

int mpibalancing::ControlLoopLoadBalancer::reductions::EmergencyJoin::getWorkerRank() const {
    return _workerRank;
}

// -----------------------------------------------------------------------------------------------------------------------

mpibalancing::ControlLoopLoadBalancer::reductions::ThresholdSumReduction::ThresholdSumReduction(double localThreshold, double totalThreshold) {
    _localThreshold = localThreshold;
    _totalThreshold = totalThreshold;
    _sumOfUpperLocalWorkload = 0.0;
    _processesWithUpperLocalWorkload = 0;
    _sumOfLowerLocalWorkload = 0.0;
    _processesWithLowerLocalWorkload = 0;
    _sumOfUpperTotalWorkload = 0.0;
    _processesWithUpperTotalWorkload = 0;
    _sumOfLowerTotalWorkload = 0.0;
    _processesWithLowerTotalWorkload = 0;
}

mpibalancing::ControlLoopLoadBalancer::reductions::ThresholdSumReduction::~ThresholdSumReduction() {}

void mpibalancing::ControlLoopLoadBalancer::reductions::ThresholdSumReduction::evaluate(
    const mpibalancing::ControlLoopLoadBalancer::History<mpibalancing::ControlLoopLoadBalancer::WorkerData>& data) 
{
    const WorkerData& workerData = data.getPastItem(1);

    if (workerData.getLocalWorkload() >= _localThreshold) {
        _sumOfUpperLocalWorkload += workerData.getLocalWorkload();
        _processesWithUpperLocalWorkload++;
    } else {
        _sumOfLowerLocalWorkload += workerData.getLocalWorkload();
        _processesWithLowerLocalWorkload++;
    }

    if (workerData.getTotalWorkload() >= _totalThreshold) {
        _sumOfUpperTotalWorkload += workerData.getTotalWorkload();
        _processesWithUpperTotalWorkload++;
    } else {
        _sumOfLowerTotalWorkload += workerData.getTotalWorkload();
        _processesWithLowerTotalWorkload++;
    }
}

double mpibalancing::ControlLoopLoadBalancer::reductions::ThresholdSumReduction::getSumOfUpperLocalWorkload() {
    return _sumOfUpperLocalWorkload;
}

int mpibalancing::ControlLoopLoadBalancer::reductions::ThresholdSumReduction::getProcessesWithUpperLocalWorkload() {
    return _processesWithUpperLocalWorkload;
}

double mpibalancing::ControlLoopLoadBalancer::reductions::ThresholdSumReduction::getSumOfLowerLocalWorkload() {
    return _sumOfLowerLocalWorkload;
}

int mpibalancing::ControlLoopLoadBalancer::reductions::ThresholdSumReduction::getProcessesWithLowerLocalWorkload() {
    return _processesWithLowerLocalWorkload;
}

double mpibalancing::ControlLoopLoadBalancer::reductions::ThresholdSumReduction::getSumOfUpperTotalWorkload() {
    return _sumOfUpperTotalWorkload;
}

int mpibalancing::ControlLoopLoadBalancer::reductions::ThresholdSumReduction::getProcessesWithUpperTotalWorkload() {
    return _processesWithUpperTotalWorkload;
}

double mpibalancing::ControlLoopLoadBalancer::reductions::ThresholdSumReduction::getSumOfLowerTotalWorkload() {
    return _sumOfLowerTotalWorkload;
}

int mpibalancing::ControlLoopLoadBalancer::reductions::ThresholdSumReduction::getProcessesWithLowerTotalWorkload() {
    return _processesWithLowerTotalWorkload;
}

