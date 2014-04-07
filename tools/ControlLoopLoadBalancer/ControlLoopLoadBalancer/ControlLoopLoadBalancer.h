#ifndef _CONTROL_LOOP_LOAD_BALANCER_H_
#define _CONTROL_LOOP_LOAD_BALANCER_H_

#include "tarch/logging/Log.h"

#include "ControlLoopLoadBalancer/WorkerData.h"
#include "ControlLoopLoadBalancer/GridStateData.h"

#include "ControlLoopLoadBalancer/History.h"
#include "ControlLoopLoadBalancer/HistorySet.h"
#include "ControlLoopLoadBalancer/Strategy.h"
#include "ControlLoopLoadBalancer/FilterStrategy.h"

namespace mpibalancing {
    namespace ControlLoopLoadBalancer {
        class ControlLoopLoadBalancer;
    }
}

class mpibalancing::ControlLoopLoadBalancer::ControlLoopLoadBalancer {
    public:
        virtual ~ControlLoopLoadBalancer();
 
        // set join and fork capabilities
        void setJoinsAllowed(bool flag);
        void setForkFailed(bool flag);

        // handle events from OracleForOnePhase
        void receivedStartCommand( int commandFromMaster );
        int getCommandForWorker( int workerRank, bool forkIsAllowed, bool joinIsAllowed );
        void receivedTerminateCommand(
          int     workerRank,
          double  waitedTime,
          double  workerNumberOfInnerVertices,
          double  workerNumberOfBoundaryVertices,
          double  workerNumberOfOuterVertices,
          double  workerNumberOfInnerCells,
          double  workerNumberOfOuterCells,
          int     workerMaxLevel,
          double  workerLocalWorkload,
          double  workerTotalWorkload,
          double  workerMaxWorkload,
          double  workerMinWorkload,
          int     currentLevel,
          double  parentCellLocalWorkload,
          const tarch::la::Vector<DIMENSIONS,double>& boundingBoxOffset,
          const tarch::la::Vector<DIMENSIONS,double>& boundingBoxSize,
          bool workerCouldNotEraseDueToDecomposition
        );
  
        virtual void reset();

    protected:
        ControlLoopLoadBalancer();
 
        virtual mpibalancing::ControlLoopLoadBalancer::HistorySet<int,WorkerData>& getWorkerHistorySet() = 0;
        virtual mpibalancing::ControlLoopLoadBalancer::History<WorkerData>& getMasterHistory() = 0;
        virtual mpibalancing::ControlLoopLoadBalancer::History<GridStateData>& getGridStateHistory() = 0;
        
        virtual mpibalancing::ControlLoopLoadBalancer::Strategy& getStrategy(void) = 0;
        virtual mpibalancing::ControlLoopLoadBalancer::FilterStrategy& getFilterStrategy(void) = 0;

        virtual void updateStrategies(void);

        // This method advances all available histories in time in a convenient way.
        // If you have further histories or other time dependent properties then
        // overload this method and handle them appropriately.
        virtual void advanceInTime();

    private:
        static tarch::logging::Log _log;
};

#endif // _CONTROL_LOOP_LOAD_BALANCER_H_
