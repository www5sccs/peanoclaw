#ifndef _CONTROL_LOOP_LOAD_BALANCER_FILTERSTRATEGY_H_
#define _CONTROL_LOOP_LOAD_BALANCER_FILTERSTRATEGY_H_

#include "History.h"
#include "HistorySet.h"
#include "WorkerData.h"
#include "GridStateData.h"

#include "tarch/logging/Log.h"

namespace mpibalancing {
    namespace ControlLoopLoadBalancer {
        class FilterStrategy;
        class PassThroughFilterStrategy;
        class BasicFilterStrategy;
    }
}

class mpibalancing::ControlLoopLoadBalancer::FilterStrategy {
    public:
        virtual ~FilterStrategy();
        virtual int run( int worker, int desiredCommand ) = 0;

    protected:
        FilterStrategy();
};

// This filter strategy just forwards the desired command without any filtering.
class mpibalancing::ControlLoopLoadBalancer::PassThroughFilterStrategy : public FilterStrategy {
    public:
        PassThroughFilterStrategy();
        virtual ~PassThroughFilterStrategy();
        virtual int run( int worker, int desiredCommand );
};

// This filter strategy checks forkIsAllowed and joinIsAllowed as well as _joinsAllowed to determine if we
// are actually allowed to perform these operations in general.
class mpibalancing::ControlLoopLoadBalancer::BasicFilterStrategy : public FilterStrategy {
    public:
        BasicFilterStrategy(
            History<WorkerData>& masterHistory, 
            HistorySet<int, WorkerData>& workerHistorySet, 
            History<GridStateData>& gridStateHistory
        );
        virtual ~BasicFilterStrategy();
        virtual int run( int worker, int desiredCommand );

    private:
        static tarch::logging::Log _log;

        History<WorkerData>& _masterHistory;
        HistorySet<int, WorkerData>& _workerHistorySet;
        History<GridStateData>& _gridStateHistory;
};


#endif // _CONTROL_LOOP_LOAD_BALANCER_FILTERSTRATEGY_H_
