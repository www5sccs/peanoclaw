#include "tarch/Assertions.h"
#include "tarch/la/ScalarOperations.h"
#include "tarch/parallel/Node.h"
#include "tarch/parallel/NodePool.h"
#include "peano/parallel/loadbalancing/Oracle.h"

#include "OracleForOnePhaseControlLoopWrapper.h"


using namespace peano::parallel::loadbalancing;

tarch::logging::Log mpibalancing::OracleForOnePhaseControlLoopWrapper::_log( "mpibalancing::OracleForOnePhaseControlLoopWrapper" );

mpibalancing::OracleForOnePhaseControlLoopWrapper::OracleForOnePhaseControlLoopWrapper(
        bool joinsAllowed,
        mpibalancing::ControlLoopLoadBalancer::ControlLoopLoadBalancer& controlLoopLoadBalancer):
  _joinsAllowed(joinsAllowed),
  _controlLoopLoadBalancer(controlLoopLoadBalancer)
{
}

mpibalancing::OracleForOnePhaseControlLoopWrapper::~OracleForOnePhaseControlLoopWrapper() {
    // Nothing has to be done here as we only rely on external information which is provided by the ControlLoopLoadBalancer.
}

void mpibalancing::OracleForOnePhaseControlLoopWrapper::receivedStartCommand(int commandFromMaster) {
    // TODO: store incoming data from our master.
    // currently this only includes the loadbalancing flag.
    // However, in future versions of Peano this might include global information of the grid as well
    // This call also marks a new data gathering round: all histories get a new slot which are filled as we go through.
    _controlLoopLoadBalancer.setJoinsAllowed(_joinsAllowed);
    _controlLoopLoadBalancer.receivedStartCommand(commandFromMaster);
}

int mpibalancing::OracleForOnePhaseControlLoopWrapper::getCommandForWorker( int workerRank, bool forkIsAllowed, bool joinIsAllowed ) {
  int result = Continue;
  logTraceInWith4Arguments( "getCommandForWorker(int,bool)", workerRank, forkIsAllowed, joinIsAllowed, _joinsAllowed );

  if (tarch::parallel::Node::getInstance().isGlobalMaster()) {
    // as the global master has no master itself, there is no reason that receivedStartCommand is called.
    // Hence we call it just before we send our command to our single worker.
    // This way we keep all our histories consistent and rolling.
    receivedStartCommand(Continue);
  }

  // TODO: if the worker asks the first time for a command, there is no history which could be used to give a reasonale answer.
  // Nevertheless, we may send Continue or use information from other workers in its neighbourhood to extrapolate/interpolate an
  // estimated values which we might use for speculative loadbalancing
  result = _controlLoopLoadBalancer.getCommandForWorker(workerRank, forkIsAllowed, joinIsAllowed);

  logTraceOutWith1Argument( "getCommandForWorker(int,bool)", result );
  return result;
}

void mpibalancing::OracleForOnePhaseControlLoopWrapper::receivedTerminateCommand(
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
      bool workerCouldNotEraseDueToDecomposition)
{
    // our worker finished its current iteration: let us feed our new data (which might be the first data given by this particular worker)
    // into our history information to give more accurate loadbalancing in our consequent turns
    _controlLoopLoadBalancer.receivedTerminateCommand(
        workerRank,
        waitedTime,
        workerNumberOfInnerVertices,
        workerNumberOfBoundaryVertices,
        workerNumberOfOuterVertices,
        workerNumberOfInnerCells,
        workerNumberOfOuterCells,
        workerMaxLevel,
        workerLocalWorkload,
        workerTotalWorkload,
        workerMaxWorkload,
        workerMinWorkload,
        currentLevel,
        parentCellLocalWorkload,
        boundingBoxOffset,
        boundingBoxSize,
        workerCouldNotEraseDueToDecomposition
    );
}

void mpibalancing::OracleForOnePhaseControlLoopWrapper::plotStatistics() {
}

mpibalancing::OracleForOnePhaseControlLoopWrapper::OracleForOnePhase* mpibalancing::OracleForOnePhaseControlLoopWrapper::createNewOracle(int adapterNumber) const {
  // Spawn a new oracle wrapper instance which uses the same state as this one, just to ensure consistent loadbalancing commands across different mappings
  return new OracleForOnePhaseControlLoopWrapper(_joinsAllowed,_controlLoopLoadBalancer);
}

void mpibalancing::OracleForOnePhaseControlLoopWrapper::forkFailed() {
    // a fork attempt has failed before, let's use this information to prevent further Forks from being issued.
    // However, as there is no notification when forks are again possible we implement some sort of decay for this value.
    // Otherwise, all load balancing attempts would rule out future fork attempts soley based on this information.
    _controlLoopLoadBalancer.setForkFailed(true);
}

int mpibalancing::OracleForOnePhaseControlLoopWrapper::getCoarsestRegularInnerAndOuterGridLevel() const {
    return 3;
}
