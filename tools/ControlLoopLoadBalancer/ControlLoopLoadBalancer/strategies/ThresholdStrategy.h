#include "ControlLoopLoadBalancer/Strategy.h" 

namespace mpibalancing {
    namespace ControlLoopLoadBalancer {
        namespace strategies {
            class ThresholdStrategy;
        }
    }
}

class mpibalancing::ControlLoopLoadBalancer::strategies::ThresholdStrategy : public Strategy {
    public:
        ThresholdStrategy(HistorySet<int, WorkerData>& workerHistorySet);
        virtual ~ThresholdStrategy();
        virtual int run( int worker );
    private:
        HistorySet<int, WorkerData>& _workerHistorySet;
};

