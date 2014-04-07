#ifndef _CONTROL_LOOP_LOAD_BALANCER_STRATEGY_H_
#define _CONTROL_LOOP_LOAD_BALANCER_STRATEGY_H_

#include "peano/parallel/loadbalancing/OracleForOnePhase.h"

#include "ControlLoopLoadBalancer/HistorySet.h"
#include "ControlLoopLoadBalancer/WorkerData.h"
#include "ControlLoopLoadBalancer/GridStateData.h"

namespace mpibalancing {
    namespace ControlLoopLoadBalancer {
        class Strategy;
        namespace strategies {
            class ForkGreedyStrategy;
            class MaximumForkGreedyStrategy;
            class ContinueStrategy;
            class RetryStrategy;
            class JoinDeepestLevelFirstStrategy;
        }
    }
}

class mpibalancing::ControlLoopLoadBalancer::Strategy {
    public:
        virtual ~Strategy();
        virtual int run( int worker ) = 0;

    protected:
        Strategy();

};

// This strategy tries to fork as long as possible without considering any information
class mpibalancing::ControlLoopLoadBalancer::strategies::ForkGreedyStrategy : public Strategy {
    public:
        ForkGreedyStrategy();
        virtual ~ForkGreedyStrategy();
        virtual int run( int worker );
};


// This strategy tries to fork only the worker which had the highest total workload in the previous iteration.
class mpibalancing::ControlLoopLoadBalancer::strategies::MaximumForkGreedyStrategy : public Strategy {
    public:
        MaximumForkGreedyStrategy(HistorySet<int, WorkerData>& workerHistorySet);
        virtual ~MaximumForkGreedyStrategy();
        virtual int run( int worker );
    private:
        HistorySet<int, WorkerData>& _workerHistorySet;
};

class mpibalancing::ControlLoopLoadBalancer::strategies::ContinueStrategy : public Strategy {
    public:
        ContinueStrategy(bool forksAllowed=true);
        virtual ~ContinueStrategy();
        virtual int run( int worker );

        void allowForks(bool flag=true);
    private:
        bool _forksAllowed;
};

// This strategy tries to retry the last desired load balancing action in case it is not equal 
// to the one which was actually issued. This might be useful if the domain of a worker should
// be erased but the grid prevents this as this might cause instability. 
// Hence, we retry this Join as it might become possible again in the near future.
// Take the JoinDueToErase strategy below as an example how to detect and resolve this issue in 
// conjuction with this Retry strategy.
class mpibalancing::ControlLoopLoadBalancer::strategies::RetryStrategy : public Strategy {
    public:
        RetryStrategy(HistorySet<int, WorkerData>& workerHistorySet);
        virtual ~RetryStrategy();
        virtual int run( int worker );
    private:
        HistorySet<int, WorkerData>& _workerHistorySet;
};


class mpibalancing::ControlLoopLoadBalancer::strategies::JoinDeepestLevelFirstStrategy : public Strategy {
    public:
        JoinDeepestLevelFirstStrategy(
            HistorySet<int, WorkerData>& workerHistorySet
        );
        virtual ~JoinDeepestLevelFirstStrategy();
        virtual int run( int worker );
    private:
        HistorySet<int, WorkerData>& _workerHistorySet;
};

#endif // _CONTROL_LOOP_LOAD_BALANCER_STRATEGY_H_
