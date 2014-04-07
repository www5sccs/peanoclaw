// This file is part of the Peano project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#ifndef _ORACLE_FOR_ONE_PHASE_CONTROL_LOOP_WRAPPER_H_
#define _ORACLE_FOR_ONE_PHASE_CONTROL_LOOP_WRAPPER_H_

#include "peano/parallel/loadbalancing/OracleForOnePhase.h"
#include "tarch/logging/Log.h"

#include "ControlLoopLoadBalancer.h"

namespace mpibalancing {
    class OracleForOnePhaseControlLoopWrapper;
}

/**
 * @author Roland Wittmann
 * This is just a plain wrapper interface to feed data into the actual Control Loop Load Balancing infrastructure.
 * For more specific details about these Oracle specific commands please look up the documentation in OracleForOnePhase.
 */
class mpibalancing::OracleForOnePhaseControlLoopWrapper : public peano::parallel::loadbalancing::OracleForOnePhase {
    public:
        OracleForOnePhaseControlLoopWrapper(bool joinsAllowed, mpibalancing::ControlLoopLoadBalancer::ControlLoopLoadBalancer& controlLoopLoadBalancer);
        virtual ~OracleForOnePhaseControlLoopWrapper();

        virtual void receivedStartCommand(int commandFromMaster);
        virtual int getCommandForWorker( int workerRank, bool forkIsAllowed, bool joinIsAllowed );
        virtual void receivedTerminateCommand(
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
          bool    workerCouldNotEraseDueToDecomposition
        );

        virtual void plotStatistics();
        virtual OracleForOnePhase* createNewOracle(int adapterNumber) const;
        virtual void forkFailed();
        virtual int getCoarsestRegularInnerAndOuterGridLevel() const;

    private:
        static tarch::logging::Log  _log;

        // This is the heart of this specific Oracle and as such it contains all the valuable information
        // about the past and the present workloads of worker, master and ourself.
        // Moreover, it tries to determine appropriate load balancing actions for each of our workers,
        // based on all available information.
        mpibalancing::ControlLoopLoadBalancer::ControlLoopLoadBalancer& _controlLoopLoadBalancer;
        bool _joinsAllowed;
};

#endif // _ORACLE_FOR_ONE_PHASE_CONTROL_LOOP_WRAPPER_H_
