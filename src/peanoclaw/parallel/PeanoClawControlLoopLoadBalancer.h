#ifndef PEANOCLAW_PARALLEL_DEFAULT_CONTROL_LOOP_LOAD_BALANCER_H_
#define PEANOCLAW_PARALLEL_DEFAULT_CONTROL_LOOP_LOAD_BALANCER_H_

#include "ControlLoopLoadBalancer/ControlLoopLoadBalancer.h"

namespace peanoclaw {
  namespace parallel {
    class PeanoClawStrategy;
    class PeanoClawControlLoopLoadBalancer;
  }
}

class peanoclaw::parallel::PeanoClawStrategy : public mpibalancing::ControlLoopLoadBalancer::Strategy {
    public:
        PeanoClawStrategy(
            mpibalancing::ControlLoopLoadBalancer::History<mpibalancing::ControlLoopLoadBalancer::WorkerData>& masterHistory,
            mpibalancing::ControlLoopLoadBalancer::HistorySet<int, mpibalancing::ControlLoopLoadBalancer::WorkerData>& workerHistorySet,
            mpibalancing::ControlLoopLoadBalancer::History<mpibalancing::ControlLoopLoadBalancer::GridStateData>& gridStateHistory
        );

        virtual ~PeanoClawStrategy();

        virtual int run( int worker );
    private:
        static tarch::logging::Log _log;

        mpibalancing::ControlLoopLoadBalancer::History<mpibalancing::ControlLoopLoadBalancer::WorkerData>& _masterHistory;
        mpibalancing::ControlLoopLoadBalancer::HistorySet<int, mpibalancing::ControlLoopLoadBalancer::WorkerData>& _workerHistorySet;
        mpibalancing::ControlLoopLoadBalancer::History<mpibalancing::ControlLoopLoadBalancer::GridStateData>& _gridStateHistory;
};

class peanoclaw::parallel::PeanoClawControlLoopLoadBalancer : public mpibalancing::ControlLoopLoadBalancer::ControlLoopLoadBalancer {
    public:
        PeanoClawControlLoopLoadBalancer();
        virtual ~PeanoClawControlLoopLoadBalancer();

        virtual mpibalancing::ControlLoopLoadBalancer::HistorySet< int, mpibalancing::ControlLoopLoadBalancer::WorkerData >& getWorkerHistorySet();
        virtual mpibalancing::ControlLoopLoadBalancer::History<mpibalancing::ControlLoopLoadBalancer::WorkerData>& getMasterHistory();
        virtual mpibalancing::ControlLoopLoadBalancer::History<mpibalancing::ControlLoopLoadBalancer::GridStateData>& getGridStateHistory();

        virtual mpibalancing::ControlLoopLoadBalancer::Strategy& getStrategy(void);
        virtual mpibalancing::ControlLoopLoadBalancer::FilterStrategy& getFilterStrategy(void);

        void suspendLoadBalancing(bool flag=true);
    private:
        static tarch::logging::Log _log;

        bool _loadBalancingSuspended;

        mpibalancing::ControlLoopLoadBalancer::StdHistoryMap< int, mpibalancing::ControlLoopLoadBalancer::WorkerData, mpibalancing::ControlLoopLoadBalancer::FIFOHistory<mpibalancing::ControlLoopLoadBalancer::WorkerData, 2> > _workerHistorySet;
        mpibalancing::ControlLoopLoadBalancer::FIFOHistory< mpibalancing::ControlLoopLoadBalancer::WorkerData, 2> _masterHistory;
        mpibalancing::ControlLoopLoadBalancer::FIFOHistory< mpibalancing::ControlLoopLoadBalancer::GridStateData, 2> _gridStateHistory;

        PeanoClawStrategy _strategy;
        mpibalancing::ControlLoopLoadBalancer::strategies::ContinueStrategy _continueStrategy;
        mpibalancing::ControlLoopLoadBalancer::BasicFilterStrategy _filterStrategy;
};

#endif // PEANOCLAW_PARALLEL_DEFAULT_CONTROL_LOOP_LOAD_BALANCER_H_
