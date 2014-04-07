#include "tarch/logging/Log.h"

#include "ControlLoopLoadBalancer/Strategy.h" 

namespace mpibalancing {
    namespace ControlLoopLoadBalancer {
        namespace strategies {
            class JoinDueToEraseStrategy;
            class EraseIssueReduction;
        }
    }
}

// This strategy starts by detecting a couldNotEraseDueToDecomposition issue.
// Peano provides this kind of information as part of the bottom-up communication and 
// as part of its grid state. As this information propragates up along the master-worker 
// topology, we have to determine if our worker has dedicated worker for its own.
// If so, we are not able to join and we have to inform these workers that they should not 
// fork further but try to join as soon as possible.
// This can be achieved by the ContinueButTryToJoin load balancing command in a top-down fashion.
// Hence, if we got this specific load balancing command we first check if one of our workers 
// was not able to erase due to decomposition. If it is possible to join that particular worker 
// then we will try to send the Join command, which might not go through as the grid might not 
// be stationary though. However, we may now include the Retry Strategy above to repeat the 
// join attempt over and over again. If this particular worker has workers on its own we forward
// the ContinueButTryToJoin load balancing command. 
// If we get to a point where we have workers but none of them has erase issues then we may check
// our local grid state if any erase attempts have failed during the last iteration.
// In such a special case we will force all of our workers to join.
class mpibalancing::ControlLoopLoadBalancer::strategies::JoinDueToEraseStrategy : public Strategy {
    public:
        JoinDueToEraseStrategy(
            History<WorkerData>& masterHistory, 
            HistorySet<int, WorkerData>& workerHistorySet,
            History<GridStateData>& gridStateHistory
        );
        virtual ~JoinDueToEraseStrategy();
        virtual int run( int worker );
    private:
        static tarch::logging::Log _log;

        History<WorkerData>& _masterHistory;
        HistorySet<int, WorkerData>& _workerHistorySet;
        History<GridStateData>& _gridStateHistory;
};

