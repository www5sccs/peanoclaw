#ifndef _DEFAULT_CONTROL_LOOP_LOAD_BALANCER_H_
#define _DEFAULT_CONTROL_LOOP_LOAD_BALANCER_H_

#include "ControlLoopLoadBalancer/ControlLoopLoadBalancer.h"

namespace mpibalancing {
    namespace ControlLoopLoadBalancer {
        class DefaultStrategy;
        class DefaultControlLoopLoadBalancer;
    }
}

class mpibalancing::ControlLoopLoadBalancer::DefaultStrategy : public Strategy {
    public:
        DefaultStrategy(
            History<WorkerData>& masterHistory, 
            HistorySet<int, WorkerData>& workerHistorySet,
            History<GridStateData>& gridStateHistory
        );

        virtual ~DefaultStrategy();
 
        virtual int run( int worker );
    private:
        static tarch::logging::Log _log;

        History<WorkerData>& _masterHistory;
        HistorySet<int, WorkerData>& _workerHistorySet;
        History<GridStateData>& _gridStateHistory;
};

class mpibalancing::ControlLoopLoadBalancer::DefaultControlLoopLoadBalancer : public mpibalancing::ControlLoopLoadBalancer::ControlLoopLoadBalancer {
    public:
        DefaultControlLoopLoadBalancer();
        virtual ~DefaultControlLoopLoadBalancer();
  
        virtual HistorySet< int, WorkerData >& getWorkerHistorySet();
        virtual History<WorkerData>& getMasterHistory();
        virtual History<GridStateData>& getGridStateHistory();

        virtual Strategy& getStrategy(void);
        virtual FilterStrategy& getFilterStrategy(void);
 
        void suspendLoadBalancing(bool flag=true);
    private:
        static tarch::logging::Log _log;

        bool _loadBalancingSuspended;

        StdHistoryMap< int, WorkerData, FIFOHistory<WorkerData, 2> > _workerHistorySet;
        FIFOHistory< WorkerData, 2> _masterHistory;
        FIFOHistory< GridStateData, 2> _gridStateHistory;

        DefaultStrategy _strategy;
        mpibalancing::ControlLoopLoadBalancer::strategies::ContinueStrategy _continueStrategy;
        BasicFilterStrategy _filterStrategy;
};

#endif // _DEFAULT_CONTROL_LOOP_LOAD_BALANCER_H_
