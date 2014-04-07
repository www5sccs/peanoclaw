#ifndef _CONTROL_LOOP_LOAD_BALANCER_REDUCTIONS_H_
#define _CONTROL_LOOP_LOAD_BALANCER_REDUCTIONS_H_

#include "ControlLoopLoadBalancer/WorkerData.h"
#include "ControlLoopLoadBalancer/HistorySet.h"

namespace mpibalancing {
    namespace ControlLoopLoadBalancer {
        namespace reductions {
            class FindDeepestLevelReduction;
            class EraseIssueReduction;
            class MaximumTotalWorkloadReduction;
            class EmergencyJoin;
            class ThresholdSumReduction;
        }
    }
}

// --------------------------------------------------------------------------------------------------------------------------------------------

class mpibalancing::ControlLoopLoadBalancer::reductions::FindDeepestLevelReduction
    : public mpibalancing::ControlLoopLoadBalancer::HistorySetReduction<mpibalancing::ControlLoopLoadBalancer::WorkerData> 
{
    public:
        FindDeepestLevelReduction();
        virtual ~FindDeepestLevelReduction();
        virtual void evaluate(const mpibalancing::ControlLoopLoadBalancer::History<mpibalancing::ControlLoopLoadBalancer::WorkerData>& data);
 
        int getDeepestLevelRank() const;
        int getDeepestLevel() const;
 
        int getMaxLevelRank() const;
        int getMaxLevel() const;
    private:
        int _deepestLevelRank;
        int _deepestLevel;
        int _maxLevelRank;
        int _maxLevel;
};

// --------------------------------------------------------------------------------------------------------------------------------------------

class mpibalancing::ControlLoopLoadBalancer::reductions::EraseIssueReduction
    : public mpibalancing::ControlLoopLoadBalancer::HistorySetReduction<mpibalancing::ControlLoopLoadBalancer::WorkerData> 
{
    public:
        EraseIssueReduction();
        virtual ~EraseIssueReduction();
        virtual void evaluate(const mpibalancing::ControlLoopLoadBalancer::History<mpibalancing::ControlLoopLoadBalancer::WorkerData>& data);

        bool workersCouldNotEraseDueToDecomposition() const;
        int getFirstWorkerRank() const;
        int getLastWorkerRank() const;

        int getSmallestWorkerRank() const;
        int getSmallestTroubleWorkerRank() const;
        int getLargestWorkerRank() const;
        int getSmallestSubtreeRank() const;
        int getLargestSubtreeRank() const;

    private:
        bool _workersCouldNotEraseDueToDecomposition;

        int _firstWorkerRank;
        int _lastWorkerRank;

        int _smallestWorkerRank;
        double _smallestNumberOfWorkerCells;
  
        int _smallestTroubleWorkerRank;
        double _smallestNumberOfTroubleWorkerCells;
 
        int _largestWorkerRank;
        double _largestNumberOfWorkerCells;

        int _smallestSubtreeRank;
        double _smallestNumberOfSubtreeCells;

        int _largestSubtreeRank;
        double _largestNumberOfSubtreeCells;
};


// --------------------------------------------------------------------------------------------------------------------------------------------

class mpibalancing::ControlLoopLoadBalancer::reductions::MaximumTotalWorkloadReduction 
    : public mpibalancing::ControlLoopLoadBalancer::HistorySetReduction<mpibalancing::ControlLoopLoadBalancer::WorkerData> 
{
    public:
        MaximumTotalWorkloadReduction();
        virtual ~MaximumTotalWorkloadReduction();
        virtual void evaluate(const mpibalancing::ControlLoopLoadBalancer::History<mpibalancing::ControlLoopLoadBalancer::WorkerData>& data);

        double getMaximumTotalWorkload() const;
    private:
        double _maximumTotalWorkload;
};

// --------------------------------------------------------------------------------------------------------------------------------------------

class mpibalancing::ControlLoopLoadBalancer::reductions::EmergencyJoin
    : public mpibalancing::ControlLoopLoadBalancer::HistorySetReduction<mpibalancing::ControlLoopLoadBalancer::WorkerData> 
{
    public:
        EmergencyJoin();
        virtual ~EmergencyJoin();
        virtual void evaluate(const mpibalancing::ControlLoopLoadBalancer::History<mpibalancing::ControlLoopLoadBalancer::WorkerData>& data);

        int getWorkerRank() const;
    private:
        int _workerRank;
};

// --------------------------------------------------------------------------------------------------------------------------------------------

class mpibalancing::ControlLoopLoadBalancer::reductions::ThresholdSumReduction
    : public mpibalancing::ControlLoopLoadBalancer::HistorySetReduction<mpibalancing::ControlLoopLoadBalancer::WorkerData> 
{
    public:
        ThresholdSumReduction(double localThreshold, double totalThreshold);
        virtual ~ThresholdSumReduction();
        virtual void evaluate(const mpibalancing::ControlLoopLoadBalancer::History<mpibalancing::ControlLoopLoadBalancer::WorkerData>& data);

        double getSumOfUpperLocalWorkload();
        int getProcessesWithUpperLocalWorkload();

        double getSumOfLowerLocalWorkload();
        int getProcessesWithLowerLocalWorkload();

        double getSumOfUpperTotalWorkload();
        int getProcessesWithUpperTotalWorkload();

        double getSumOfLowerTotalWorkload();
        int getProcessesWithLowerTotalWorkload();

    private:
        // input
        double _localThreshold;
        double _totalThreshold;

        // output
        double _sumOfUpperLocalWorkload;
        int _processesWithUpperLocalWorkload;

        double _sumOfLowerLocalWorkload;
        int _processesWithLowerLocalWorkload;
  
        double _sumOfUpperTotalWorkload;
        int _processesWithUpperTotalWorkload;

        double _sumOfLowerTotalWorkload;
        int _processesWithLowerTotalWorkload;
};

#endif //_CONTROL_LOOP_LOAD_BALANCER_REDUCTIONS_H_
