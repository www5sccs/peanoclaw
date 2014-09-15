/*
 * LevelAwareRoundRobinNodePoolStrategy.cpp
 *
 *  Created on: Sep 15, 2014
 *      Author: kristof
 */
#include "peanoclaw/parallel/LevelAwareRoundRobinNodePoolStrategy.h"

#include "peano/utils/Dimensions.h"

tarch::logging::Log peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::_log("peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy");

peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::Node::Node()
  : _isIdle(true),
    _rank(-1)
{
}

peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::Node::Node(int rank)
  : _isIdle(true),
    _rank(rank)
{
}

bool peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::Node::isIdle() const {
  return _isIdle;
}

void peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::Node::setIdle() {
  _isIdle = true;
}

void peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::Node::setWorking() {
  _isIdle = false;
}

int peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::Node::getRank() const {
  return _rank;
}

void peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::Level::addNode(int rank) {
  _nodes[rank] = Node(rank);
}

void peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::Level::removeNode(int rank) {
  _nodes.erase(rank);
}

int peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::Level::removeNextIdleNode() {
  std::map<int,Node>::iterator i = _nodes.begin();
  if(i == _nodes.end()) {
    return -1;
  }
  int rank = i->first;
  _nodes.erase(i);
  return rank;
}

bool peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::Level::hasNode(int rank) const {
  return _nodes.find(rank) != _nodes.end();
}

bool peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::Level::isIdle(int rank) const {
  return _nodes.find(rank)->second.isIdle();
}

void peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::Level::setIdle(int rank) {
  _nodes[rank].setIdle();
}

void peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::Level::setWorking(int rank) {
  _nodes[rank].setWorking();
}

bool peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::Level::hasIdleRank() const {
  return getIdleRank() != -1;
}

int peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::Level::getIdleRank() const {
  for(std::map<int,Node>::const_iterator i = _nodes.begin(); i != _nodes.end(); i++) {
    if(i->second.isIdle()) {
      return i->first;
    }
  }
  return -1;
}

int peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::getLevelIndexForRank(int rank) const {
  if(rank == 0) {
    return 0;
  }

  return (int)(std::ceil(pow(rank, 1.0 / (pow(3.0, DIMENSIONS))))) + 1;
}

peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::Level&
peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::getLevel(int levelIndex) {
  if((int)_level.size() < levelIndex+1) {
    _level.resize(levelIndex+1);
  }

  return _level[levelIndex];
}

bool peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::hasLevel(int levelIndex) const {
  return (int)_level.size() > levelIndex;
}

peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::LevelAwareRoundRobinNodePoolStrategy()
  : _tag(-100),
    _numberOfNodes(0),
    _numberOfIdleNodes(0),
    _level()
{
}

peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::~LevelAwareRoundRobinNodePoolStrategy() {
}

void peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::setNodePoolTag(int tag) {
  _tag = tag;
}

tarch::parallel::messages::WorkerRequestMessage extractElementFromRequestQueue(tarch::parallel::NodePoolStrategy::RequestQueue& queue) {
  assertion( !queue.empty() );
  tarch::parallel::messages::WorkerRequestMessage result = queue.front();
  queue.pop_front();
  return result;
}


void peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::fillWorkerRequestQueue(RequestQueue& queue) {
  #ifdef Parallel
  assertion( _tag >= 0 );
  while ( tarch::parallel::messages::WorkerRequestMessage::isMessageInQueue(_tag, true) ) {
    tarch::parallel::messages::WorkerRequestMessage message;
    message.receive(MPI_ANY_SOURCE,_tag, true, SendAndReceiveLoadBalancingMessagesBlocking);
    queue.push_back( message );
  }
  #endif
}

void peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::addNode(
  const tarch::parallel::messages::RegisterAtNodePoolMessage& node
) {
  Level& level = getLevel(getLevelIndexForRank(node.getSenderRank()));
  level.addNode(node.getSenderRank());

  _numberOfNodes++;
  _numberOfIdleNodes++;
}


void peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::removeNode( int rank ) {
  Level& level = getLevel(getLevelIndexForRank(rank));
  level.removeNode(rank);

  _numberOfNodes--;
  _numberOfIdleNodes--;
}


int peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::getNumberOfIdleNodes() const {
  return _numberOfIdleNodes;
}

void peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::setNodeIdle( int rank ) {
  Level& level = getLevel(getLevelIndexForRank(rank));
  if(!level.isIdle(rank)) {
    _numberOfIdleNodes++;
  }
  level.setIdle(rank);
}


int peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::reserveNode(int forMaster) {
  Level& level = getLevel(getLevelIndexForRank(forMaster) + 1);

  int rank = level.getIdleRank();
  level.setWorking(rank);
  _numberOfIdleNodes--;
  return rank;
}

void peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::reserveParticularNode(int rank) {
  Level& level = getLevel(getLevelIndexForRank(rank));
  assertion(level.isIdle(rank));
  level.setWorking(rank);
  _numberOfIdleNodes--;
}

bool peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::isRegisteredNode(int rank) const {
  int levelIndex = getLevelIndexForRank(rank);
  if(!hasLevel(levelIndex)) {
    return false;
  }

  const Level& level = _level[levelIndex];
  return level.hasNode(rank);
}


bool peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::isIdleNode(int rank) const {
  if(!isRegisteredNode(rank)) {
    return false;
  }
  const Level& level = _level[getLevelIndexForRank(rank)];
  return level.isIdle(rank);
}

int peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::getNumberOfRegisteredNodes() const {
  return _numberOfNodes;
}

std::string peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::toString() const {
  return "LevelAwareRoundRobinNodePoolStrategy";
}

bool peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::hasIdleNode(int forMaster) const {
  int levelIndex = getLevelIndexForRank(forMaster) + 1;
  if(!hasLevel(levelIndex)) {
    return false;
  }
  const Level& level = _level[levelIndex];
  return level.hasIdleRank();
}

int peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy::removeNextIdleNode() {
  for(size_t levelIndex = 0; levelIndex < _level.size(); levelIndex++) {
    int rank = _level[levelIndex].removeNextIdleNode();
    if(rank > -1) {
      return rank;
    }
  }
  return -1;
}

