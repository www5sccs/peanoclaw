/*
 * DefaultTransfer.h
 *
 *  Created on: May 6, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_INTERSUBGRIDCOMMUNICATION_DEFAULTTRANSFER_H_
#define PEANOCLAW_INTERSUBGRIDCOMMUNICATION_DEFAULTTRANSFER_H_

namespace peanoclaw {
  namespace interSubgridCommunication {
    template<int NumberOfUnknowns>
    class DefaultTransfer;
  }
}

template<int NumberOfUnknowns>
class peanoclaw::interSubgridCommunication::DefaultTransfer{

  public:
    void transfer(
      const tarch::la::Vector<DIMENSIONS, int>&    size,
      const tarch::la::Vector<DIMENSIONS, int>&    sourceOffset,
      const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
      peanoclaw::Patch& source,
      peanoclaw::Patch&       destination
    );
};

#include "peanoclaw/interSubgridCommunication/DefaultTransfer.cpph"

#endif /* PEANOCLAW_INTERSUBGRIDCOMMUNICATION_DEFAULTTRANSFER_H_ */
