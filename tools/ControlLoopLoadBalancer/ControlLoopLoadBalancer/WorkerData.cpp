#include "peano/parallel/loadbalancing/OracleForOnePhase.h"

#include "WorkerData.h"

mpibalancing::ControlLoopLoadBalancer::WorkerData::WorkerData() {
    reset();
}

mpibalancing::ControlLoopLoadBalancer::WorkerData::~WorkerData() { }

const int mpibalancing::ControlLoopLoadBalancer::WorkerData::getRank() const {
    return _rank;
}

void mpibalancing::ControlLoopLoadBalancer::WorkerData::setRank(int rank) {
    _rank = rank;
}

const tarch::la::Vector<DIMENSIONS,double>& mpibalancing::ControlLoopLoadBalancer::WorkerData::getBoundingBoxOffset() const {
    return _boundingBoxOffset;
}

void mpibalancing::ControlLoopLoadBalancer::WorkerData::setBoundingBoxOffset(const tarch::la::Vector<DIMENSIONS,double>& offset) {
    _boundingBoxOffset = offset;
}

const tarch::la::Vector<DIMENSIONS,double>& mpibalancing::ControlLoopLoadBalancer::WorkerData::getBoundingBoxSize() const {
    return _boundingBoxSize;
}

void mpibalancing::ControlLoopLoadBalancer::WorkerData::setBoundingBoxSize(const tarch::la::Vector<DIMENSIONS,double>& size) {
    _boundingBoxSize = size;
}

const bool mpibalancing::ControlLoopLoadBalancer::WorkerData::isForkAllowed() const {
    return _forkIsAllowed;
}

void mpibalancing::ControlLoopLoadBalancer::WorkerData::setForkAllowed(bool flag) {
    _forkIsAllowed = flag;
}

const bool mpibalancing::ControlLoopLoadBalancer::WorkerData::isJoinAllowed() const {
    return _joinIsAllowed;
}

void mpibalancing::ControlLoopLoadBalancer::WorkerData::setJoinAllowed(bool flag) {
    _joinIsAllowed = flag;
}

const int mpibalancing::ControlLoopLoadBalancer::WorkerData::getDesiredLoadBalancingCommand() const {
    return _desiredLoadBalancingCommand;
}
 
void mpibalancing::ControlLoopLoadBalancer::WorkerData::setDesiredLoadBalancingCommand(int command) {
    _desiredLoadBalancingCommand = command;
}

const int mpibalancing::ControlLoopLoadBalancer::WorkerData::getActualLoadBalancingCommand() const {
    return _actualLoadBalancingCommand;
}
 
void mpibalancing::ControlLoopLoadBalancer::WorkerData::setActualLoadBalancingCommand(int command) {
    _actualLoadBalancingCommand = command;
}

const double mpibalancing::ControlLoopLoadBalancer::WorkerData::getWaitedTime() const {
    return _waitedTime;
}

void mpibalancing::ControlLoopLoadBalancer::WorkerData::setWaitedTime(double time) {
    _waitedTime = time;
}

const double mpibalancing::ControlLoopLoadBalancer::WorkerData::getNumberOfInnerVertices() const {
    return _numberOfInnerVertices;
}

void mpibalancing::ControlLoopLoadBalancer::WorkerData::setNumberOfInnerVertices(double vertices) {
    _numberOfInnerVertices = vertices;
}

const double mpibalancing::ControlLoopLoadBalancer::WorkerData::getNumberOfBoundaryVertices() const {
    return _numberOfBoundaryVertices;
}

void mpibalancing::ControlLoopLoadBalancer::WorkerData::setNumberOfBoundaryVertices(double vertices) {
    _numberOfBoundaryVertices = vertices;
}

const double mpibalancing::ControlLoopLoadBalancer::WorkerData::getNumberOfOuterVertices() const {
    return _numberOfOuterVertices;
}

void mpibalancing::ControlLoopLoadBalancer::WorkerData::setNumberOfOuterVertices(double vertices) {
    _numberOfOuterVertices = vertices;
}

const double mpibalancing::ControlLoopLoadBalancer::WorkerData::getNumberOfInnerCells() const {
    return _numberOfInnerCells;
}

void mpibalancing::ControlLoopLoadBalancer::WorkerData::setNumberOfInnerCells(double cells) {
    _numberOfInnerCells = cells;
}

const double mpibalancing::ControlLoopLoadBalancer::WorkerData::getNumberOfOuterCells() const {
    return _numberOfOuterCells;
}

void mpibalancing::ControlLoopLoadBalancer::WorkerData::setNumberOfOuterCells(double cells) {
    _numberOfOuterCells = cells;
}

const int mpibalancing::ControlLoopLoadBalancer::WorkerData::getMaxLevel() const {
    return _maxLevel;
}

void mpibalancing::ControlLoopLoadBalancer::WorkerData::setMaxLevel(int level) {
    _maxLevel = level;
}

const int mpibalancing::ControlLoopLoadBalancer::WorkerData::getCurrentLevel() const {
    return _currentLevel;
}

void mpibalancing::ControlLoopLoadBalancer::WorkerData::setCurrentLevel(int level) {
    _currentLevel = level;
}

const double mpibalancing::ControlLoopLoadBalancer::WorkerData::getLocalWorkload() const {
    return _localWorkload;
}

void mpibalancing::ControlLoopLoadBalancer::WorkerData::setLocalWorkload(double workload) {
    _localWorkload = workload;
}

const double mpibalancing::ControlLoopLoadBalancer::WorkerData::getTotalWorkload() const {
    return _totalWorkload;
}

void mpibalancing::ControlLoopLoadBalancer::WorkerData::setTotalWorkload(double workload) {
    _totalWorkload = workload;
}

const double mpibalancing::ControlLoopLoadBalancer::WorkerData::getMaxWorkload() const {
    return _maxWorkload;
}

void mpibalancing::ControlLoopLoadBalancer::WorkerData::setMaxWorkload(double workload) {
    _maxWorkload = workload;
}

const double mpibalancing::ControlLoopLoadBalancer::WorkerData::getMinWorkload() const {
    return _minWorkload;
}

void mpibalancing::ControlLoopLoadBalancer::WorkerData::setMinWorkload(double workload) {
    _minWorkload = workload;
}


const double mpibalancing::ControlLoopLoadBalancer::WorkerData::getParentCellLocalWorkload() const {
    return _parentCellLocalWorkload;
}

void mpibalancing::ControlLoopLoadBalancer::WorkerData::setParentCellLocalWorkload(double workload) { 
    _parentCellLocalWorkload = workload;
}

const bool mpibalancing::ControlLoopLoadBalancer::WorkerData::getCouldNotEraseDueToDecompositionFlag() const {
    return _couldNotEraseDueToDecomposition;
}

void mpibalancing::ControlLoopLoadBalancer::WorkerData::setCouldNotEraseDueToDecompositionFlag(bool flag) {
    _couldNotEraseDueToDecomposition = flag;
}

void mpibalancing::ControlLoopLoadBalancer::WorkerData::reset() {
    _rank = -1;
    _boundingBoxOffset = tarch::la::Vector<DIMENSIONS,double>(0.0);
    _boundingBoxSize = tarch::la::Vector<DIMENSIONS,double>(0.0);
    _desiredLoadBalancingCommand = peano::parallel::loadbalancing::UndefinedLoadBalancingFlag;
    _actualLoadBalancingCommand = peano::parallel::loadbalancing::UndefinedLoadBalancingFlag;
    _waitedTime = 0.0;
    _numberOfInnerVertices = 0.0;
    _numberOfBoundaryVertices = 0.0;
    _numberOfOuterVertices = 0.0;
    _numberOfInnerCells = 0.0;
    _numberOfOuterCells = 0.0;
    _maxLevel = 0;
    _currentLevel = 0;
    _localWorkload = 0.0;
    _totalWorkload = 0.0;
    _maxWorkload = 0.0;
    _minWorkload = 0.0;
    _parentCellLocalWorkload = 0.0;
    _couldNotEraseDueToDecomposition = false;
}

bool mpibalancing::ControlLoopLoadBalancer::WorkerData::operator<(const WorkerData& right) {
    return getRank() < right.getRank();
}
