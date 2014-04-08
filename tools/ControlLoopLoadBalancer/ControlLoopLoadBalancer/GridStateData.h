#ifndef _CONTROL_LOOP_LOAD_BALANCER_STATEDATA_H_
#define _CONTROL_LOOP_LOAD_BALANCER_STATEDATA_H_

#include <ostream>

namespace mpibalancing {
    namespace ControlLoopLoadBalancer {
        class GridStateData;
    }
}

// This class contains valueable information about ourself and the current grid status, 
// both globally and locally. Nevertheless, this class is work in progress as peano is 
// evolving pretty fast with respect to parellelization.
class mpibalancing::ControlLoopLoadBalancer::GridStateData {
    public:
        GridStateData();
        virtual ~GridStateData();

        const bool isTraversalInverted() const;
        void setTraversalInverted(bool flag);

        const bool areJoinsAllowed() const;
        void setJoinsAllowed(bool flag);

        const bool hasForkFailed() const;
        void setForkFailed(bool flag);

        const bool isGridStationary() const;
        void setGridStationary(bool flag);

        const bool isGridBalanced() const;
        void setGridBalanced(bool flag);

        const bool couldNotEraseDueToDecomposition() const;
        void setCouldNotEraseDueToDecomposition(bool flag);
 
        const bool subWorkerIsInvolvedInJoinOrFork() const;
        void setSubWorkerIsInvolvedInJoinOrFork(bool flag);

        // Use this function to reset all available data.
        void reset();
    private:
        bool _traversalInverted;
        bool _joinsAllowed;
        bool _forkFailed;
        bool _gridStationary;
        bool _gridBalanced;
        bool _couldNotEraseDueToDecomposition;
        bool _subWorkerIsInvolvedInJoinOrFork;
};

std::ostream& operator<<(std::ostream& stream, const mpibalancing::ControlLoopLoadBalancer::GridStateData& workerData);

#endif
