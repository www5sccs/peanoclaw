/*
 * Extrapolation.cpp
 *
 *  Created on: Jul 8, 2013
 *      Author: kristof
 */
#include "peanoclaw/interSubgridCommunication/Extrapolation.h"
#include "peanoclaw/interSubgridCommunication/aspects/CornerTraversal.h"

#include "peano/utils/Loop.h"

void peanoclaw::interSubgridCommunication::CornerExtrapolation::operator()(
  peanoclaw::Patch& patch,
  const peanoclaw::Area& area,
  const tarch::la::Vector<DIMENSIONS,int> cornerIndex
) const {
  dfor(subcellIndexInArea, area._size) {
    tarch::la::Vector<DIMENSIONS, int> subcellIndex = subcellIndexInArea + area._offset;
    int linearIndexSubcell = patch.getLinearIndexUOld(subcellIndex);

    //Reset value in corner
    for(int unknown = 0; unknown < patch.getUnknownsPerSubcell(); unknown++) {
      patch.setValueUOld(linearIndexSubcell, unknown, 0.0);
    }

    for(int d = 0; d < DIMENSIONS; d++) {
      tarch::la::Vector<DIMENSIONS, int> support1 = subcellIndex;
      tarch::la::Vector<DIMENSIONS, int> support2 = support1;
      support1(d) = (cornerIndex(d) == 0) ? 0 : (patch.getSubdivisionFactor()(d) - 1);
      support2(d) = (cornerIndex(d) == 0) ? 1 : (patch.getSubdivisionFactor()(d) - 2);
      int distanceSupport1 = abs(support1(d) - subcellIndex(d));
      int distanceSupport2 = abs(support2(d) - subcellIndex(d));

      int linearIndexSupport1 = patch.getLinearIndexUOld(support1);
      int linearIndexSupport2 = patch.getLinearIndexUOld(support2);

      //TODO unterweg debug
//      std::cout << "Computing value for cell " << subcellIndex
//          << " with support1=" << support1 << " (d=" << distanceSupport1 << ",v=" << patch.getValueUOld(linearIndexSupport1, 0) << ")"
//          << ", support2=" << support2 << " (d=" << distanceSupport2 << ",v=" << patch.getValueUOld(linearIndexSupport2, 0) << ")"
//          << ", delta=" << ((patch.getValueUOld(linearIndexSupport1, 0) * (distanceSupport1+1)
//             + patch.getValueUOld(linearIndexSupport2, 0) * (distanceSupport2-1))
//             / (double)DIMENSIONS)
//          << std::endl;

      for(int unknown = 0; unknown < patch.getUnknownsPerSubcell(); unknown++) {
        patch.setValueUOld(linearIndexSubcell, unknown, patch.getValueUOld(linearIndexSubcell, unknown)
            + (patch.getValueUOld(linearIndexSupport1, unknown) * (distanceSupport1+1)
             - patch.getValueUOld(linearIndexSupport2, unknown) * (distanceSupport2-1))
             / (double)DIMENSIONS);
      }
    }
  }
}

void peanoclaw::interSubgridCommunication::Extrapolation::extrapolateEdges() {
  assertionFail("Not implemented yet!");
}

void peanoclaw::interSubgridCommunication::Extrapolation::extrapolateCorners() {
  peanoclaw::interSubgridCommunication::aspects::CornerTraversal<CornerExtrapolation> cornerTraversal(
    _patch,
    CornerExtrapolation()
  );
}

peanoclaw::interSubgridCommunication::Extrapolation::Extrapolation(
  peanoclaw::Patch& patch
) : _patch(patch) {
}

void peanoclaw::interSubgridCommunication::Extrapolation::extrapolateGhostlayer() {
  #ifdef Dim3
  extrapolateEdges();
  #endif
  extrapolateCorners();
}



