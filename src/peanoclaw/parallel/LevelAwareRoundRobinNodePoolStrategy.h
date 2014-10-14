/*
 * RoundRobinNodePoolStrategy.h
 *
 *  Created on: Sep 15, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_PARALLEL_LEVELAWAREROUNDROBINNODEPOOLSTRATEGY_H_
#define PEANOCLAW_PARALLEL_LEVELAWAREROUNDROBINNODEPOOLSTRATEGY_H_

namespace peanoclaw {
  namespace parallel {
    class LevelAwareRoundRobinNodePoolStrategy;
  }
}

#include "tarch/parallel/NodePoolStrategy.h"
#include "tarch/parallel/FCFSNodePoolStrategy.h"

#include <vector>
#include <map>

class peanoclaw::parallel::LevelAwareRoundRobinNodePoolStrategy : public tarch::parallel::NodePoolStrategy {

  class Node {
    private:
      enum State {
        Registered,
        Idle,
        Working
      };
      State _state;

      int   _rank;

    public:
      Node();
      Node(int rank);

      bool isIdle() const;
      void setWorking();
      void setIdle();
      int getRank() const;

      bool operator<(const Node& right) const;
  };

  class Level {
    private:
      std::map<int, Node> _nodes;

    public:
      void addNode(int rank);
      void removeNode(int rank);
      int  removeNextIdleNode();
      bool hasNode(int rank) const;

      bool isIdle(int rank) const;
      void setIdle(int rank);
      void setWorking(int rank);

      bool hasIdleRank() const;
      int getIdleRank() const;

      std::string toString() const;
  };

  /**
   * Logging Device
   */
  static tarch::logging::Log _log;

  /**
   * Tag on which the node pool works
   */
  int _tag;

  int _numberOfNodes;

  int _numberOfIdleNodes;

  std::vector<Level> _level;

  int getLevelIndexForRank(int rank) const;

  Level& getLevel(int levelIndex);

  bool hasLevel(int levelIndex) const;

public:
  /**
   * Constructor
   *
   * Construct all the attributes.
   */
  LevelAwareRoundRobinNodePoolStrategy();
  virtual ~LevelAwareRoundRobinNodePoolStrategy();

  virtual void setNodePoolTag(int tag);

  virtual tarch::parallel::messages::WorkerRequestMessage extractElementFromRequestQueue(tarch::parallel::NodePoolStrategy::RequestQueue& queue);

  virtual void fillWorkerRequestQueue(RequestQueue& queue);

  virtual void addNode(const tarch::parallel::messages::RegisterAtNodePoolMessage& node );

  virtual void removeNode( int rank );

  virtual int getNumberOfIdleNodes() const;

  virtual void setNodeIdle( int rank );

  virtual int reserveNode(int forMaster);

  virtual void reserveParticularNode(int rank);

  virtual bool isRegisteredNode(int rank) const;

  virtual bool isIdleNode(int rank) const;

  virtual int getNumberOfRegisteredNodes() const;

  virtual std::string toString() const;

  virtual bool hasIdleNode(int forMaster) const;

  virtual int removeNextIdleNode();
};


#endif /* PEANOCLAW_PARALLEL_LEVELAWAREROUNDROBINNODEPOOLSTRATEGY_H_ */
