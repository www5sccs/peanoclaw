#include "GridStateData.h"

mpibalancing::ControlLoopLoadBalancer::GridStateData::GridStateData() {
    reset();
}

mpibalancing::ControlLoopLoadBalancer::GridStateData::~GridStateData() {}

const bool mpibalancing::ControlLoopLoadBalancer::GridStateData::isTraversalInverted() const {
    return _traversalInverted;
}

void mpibalancing::ControlLoopLoadBalancer::GridStateData::setTraversalInverted(bool flag) {
    _traversalInverted = flag;
}

const bool mpibalancing::ControlLoopLoadBalancer::GridStateData::areJoinsAllowed() const {
    return _joinsAllowed;
}

void mpibalancing::ControlLoopLoadBalancer::GridStateData::setJoinsAllowed(bool flag) {
    _joinsAllowed = flag;
}

const bool mpibalancing::ControlLoopLoadBalancer::GridStateData::hasForkFailed() const {
    return _forkFailed;
}

void mpibalancing::ControlLoopLoadBalancer::GridStateData::setForkFailed(bool flag) {
    _forkFailed = flag;
}

const bool mpibalancing::ControlLoopLoadBalancer::GridStateData::isGridStationary() const {
    return _gridStationary;
}

void mpibalancing::ControlLoopLoadBalancer::GridStateData::setGridStationary(bool flag) {
    _gridStationary = flag;
}

const bool mpibalancing::ControlLoopLoadBalancer::GridStateData::isGridBalanced() const {
    return _gridBalanced;
}

void mpibalancing::ControlLoopLoadBalancer::GridStateData::setGridBalanced(bool flag) {
    _gridBalanced = flag;
}

const bool mpibalancing::ControlLoopLoadBalancer::GridStateData::couldNotEraseDueToDecomposition() const {
    return _couldNotEraseDueToDecomposition;
}

void mpibalancing::ControlLoopLoadBalancer::GridStateData::setCouldNotEraseDueToDecomposition(bool flag) {
    _couldNotEraseDueToDecomposition = flag;
}

const bool mpibalancing::ControlLoopLoadBalancer::GridStateData::subWorkerIsInvolvedInJoinOrFork() const {
    return _subWorkerIsInvolvedInJoinOrFork;
}

void mpibalancing::ControlLoopLoadBalancer::GridStateData::setSubWorkerIsInvolvedInJoinOrFork(bool flag) {
   _subWorkerIsInvolvedInJoinOrFork = flag;
}

void mpibalancing::ControlLoopLoadBalancer::GridStateData::reset() {
    _joinsAllowed = false;
    _forkFailed = false;
    _gridStationary = false;
    _gridBalanced = false;
    _couldNotEraseDueToDecomposition = false;
    _subWorkerIsInvolvedInJoinOrFork = false;
}

std::ostream& operator<<(std::ostream& stream, const mpibalancing::ControlLoopLoadBalancer::GridStateData& workerData) {
  stream << "GridStateData";
  return stream;
}
