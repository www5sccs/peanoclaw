/*
 * DefaultTransfer.h
 *
 *  Created on: May 6, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_INTERSUBGRIDCOMMUNICATION_DEFAULTTRANSFER_H_
#define PEANOCLAW_INTERSUBGRIDCOMMUNICATION_DEFAULTTRANSFER_H_

#include "peanoclaw/Patch.h"
#include "peanoclaw/grid/SubgridAccessor.h"

#include "tarch/la/Vector.h"
#include "peano/utils/Dimensions.h"

namespace peanoclaw {
  namespace interSubgridCommunication {

    class DefaultTransfer;

    template<int NumberOfUnknowns>
    class DefaultTransferTemplate;
  }
}

/**
 * Implements the transfer between ghostlayers with the same resolution
 * or from uNew to uOld.
 *
 * Use the class DefaultTransfer to hide this template.
 *
 * The template is used to optimize the loop over the unknowns per cell.
 */
template<int NumberOfUnknowns>
class peanoclaw::interSubgridCommunication::DefaultTransferTemplate{

  public:
    void transferGhostlayer(
      const tarch::la::Vector<DIMENSIONS, int>&    size,
      const tarch::la::Vector<DIMENSIONS, int>&    sourceOffset,
      const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
      peanoclaw::Patch& source,
      peanoclaw::Patch& destination
    );

    /*
    * Copies the data from uNew to uOld, i.e. this actually performs the
    * transition to the next timestep by setting uOld(t) to uOld(t+1).
    * The ghost layer stays unchanged.
    */
    void copyUNewToUOld(peanoclaw::Patch& subgrid);

    void swapUOldAndUNew(peanoclaw::Patch& subgrid);
};

/**
 * Implements the transfer between ghostlayers with the same resolution
 * or from uNew to uOld.
 *
 * Internally this class dispatches the call to the appropriate
 * templatized version of DefaultTransferTemplate.
 */
class peanoclaw::interSubgridCommunication::DefaultTransfer {

public:
  void transferGhostlayer(
    const tarch::la::Vector<DIMENSIONS, int>&    size,
    const tarch::la::Vector<DIMENSIONS, int>&    sourceOffset,
    const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
    peanoclaw::Patch& source,
    peanoclaw::Patch&       destination
  ) {
    assertionEquals(source.getUnknownsPerSubcell(), destination.getUnknownsPerSubcell());

    switch(source.getUnknownsPerSubcell()) {
      case 1:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<1> transfer1;
        transfer1.transferGhostlayer(size, sourceOffset, destinationOffset, source, destination);
        break;
      case 2:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<2> transfer2;
        transfer2.transferGhostlayer(size, sourceOffset, destinationOffset, source, destination);
        break;
      case 3:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<3> transfer3;
        transfer3.transferGhostlayer(size, sourceOffset, destinationOffset, source, destination);
        break;
      case 4:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<4> transfer4;
        transfer4.transferGhostlayer(size, sourceOffset, destinationOffset, source, destination);
        break;
      case 5:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<5> transfer5;
        transfer5.transferGhostlayer(size, sourceOffset, destinationOffset, source, destination);
        break;
      case 6:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<6> transfer6;
        transfer6.transferGhostlayer(size, sourceOffset, destinationOffset, source, destination);
        break;
      case 7:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<7> transfer7;
        transfer7.transferGhostlayer(size, sourceOffset, destinationOffset, source, destination);
        break;
      case 8:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<8> transfer8;
        transfer8.transferGhostlayer(size, sourceOffset, destinationOffset, source, destination);
        break;
      case 9:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<9> transfer9;
        transfer9.transferGhostlayer(size, sourceOffset, destinationOffset, source, destination);
        break;
      case 10:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<10> transfer10;
        transfer10.transferGhostlayer(size, sourceOffset, destinationOffset, source, destination);
        break;
      default:
        assertionFail("Number of unknowns " << source.getUnknownsPerSubcell() << " not supported!");
    }

    #ifdef Asserts
    peanoclaw::grid::SubgridAccessor destinationAccessor = destination.getAccessor();
    dfor(subcellIndex, size) {
      tarch::la::Vector<DIMENSIONS, int> actualSubcellIndex = subcellIndex + destinationOffset;
      for(int unknown = 0; unknown < destination.getUnknownsPerSubcell(); unknown++) {
        double value = destinationAccessor.getValueUOld(actualSubcellIndex, unknown);
        assertionEquals6(value, value,
            destination,
            destination.toStringUOldWithGhostLayer(),
            source.toStringUNew(),
            sourceOffset,
            destinationOffset,
            size);
      }
    }
    #endif
  }

  void copyUNewToUOld(peanoclaw::Patch& subgrid) {
    switch(subgrid.getUnknownsPerSubcell()) {
      case 1:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<1> transfer1;
        transfer1.copyUNewToUOld(subgrid);
        break;
      case 2:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<2> transfer2;
        transfer2.copyUNewToUOld(subgrid);
        break;
      case 3:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<3> transfer3;
        transfer3.copyUNewToUOld(subgrid);
        break;
      case 4:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<4> transfer4;
        transfer4.copyUNewToUOld(subgrid);
        break;
      case 5:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<5> transfer5;
        transfer5.copyUNewToUOld(subgrid);
        break;
      case 6:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<6> transfer6;
        transfer6.copyUNewToUOld(subgrid);
        break;
      case 7:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<7> transfer7;
        transfer7.copyUNewToUOld(subgrid);
        break;
      case 8:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<8> transfer8;
        transfer8.copyUNewToUOld(subgrid);
        break;
      case 9:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<9> transfer9;
        transfer9.copyUNewToUOld(subgrid);
        break;
      case 10:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<10> transfer10;
        transfer10.copyUNewToUOld(subgrid);
        break;
      default:
        assertionFail("Number of unknowns " << subgrid.getUnknownsPerSubcell() << " not supported!");
    }
  }

  void swapUNewAndUOld(peanoclaw::Patch& subgrid) {
    switch(subgrid.getUnknownsPerSubcell()) {
      case 1:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<1> transfer1;
        transfer1.swapUOldAndUNew(subgrid);
        break;
      case 2:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<2> transfer2;
        transfer2.swapUOldAndUNew(subgrid);
        break;
      case 3:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<3> transfer3;
        transfer3.swapUOldAndUNew(subgrid);
        break;
      case 4:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<4> transfer4;
        transfer4.swapUOldAndUNew(subgrid);
        break;
      case 5:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<5> transfer5;
        transfer5.swapUOldAndUNew(subgrid);
        break;
      case 6:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<6> transfer6;
        transfer6.swapUOldAndUNew(subgrid);
        break;
      case 7:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<7> transfer7;
        transfer7.swapUOldAndUNew(subgrid);
        break;
      case 8:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<8> transfer8;
        transfer8.swapUOldAndUNew(subgrid);
        break;
      case 9:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<9> transfer9;
        transfer9.swapUOldAndUNew(subgrid);
        break;
      case 10:
        peanoclaw::interSubgridCommunication::DefaultTransferTemplate<10> transfer10;
        transfer10.swapUOldAndUNew(subgrid);
        break;
      default:
        assertionFail("Number of unknowns " << subgrid.getUnknownsPerSubcell() << " not supported!");
    }
  }

};

#include "peanoclaw/interSubgridCommunication/DefaultTransfer.cpph"

#endif /* PEANOCLAW_INTERSUBGRIDCOMMUNICATION_DEFAULTTRANSFER_H_ */
