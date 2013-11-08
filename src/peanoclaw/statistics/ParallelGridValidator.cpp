/*
 * ParallelGridValidator.cpp
 *
 *  Created on: Jun 29, 2013
 *      Author: kristof
 */
#include "peanoclaw/statistics/ParallelGridValidator.h"

#include "peanoclaw/Patch.h"
#include "peanoclaw/interSubgridCommunication/aspects/FaceAdjacentPatchTraversal.h"
#include "peanoclaw/interSubgridCommunication/aspects/EdgeAdjacentPatchTraversal.h"
#include "peanoclaw/interSubgridCommunication/aspects/CornerAdjacentPatchTraversal.h"

#include "peanoclaw/Heap.h"
#include "peano/utils/Dimensions.h"
#include "peano/utils/Loop.h"

#include "tarch/Assertions.h"
#include "tarch/la/Vector.h"


tarch::logging::Log peanoclaw::statistics::ParallelGridValidator::_log("peanoclaw::statistics::ParallelGridValidator");

tarch::la::Vector<DIMENSIONS,double> peanoclaw::statistics::ParallelGridValidator::getNeighborPositionOnLevel(
  const tarch::la::Vector<DIMENSIONS,double>& finePosition,
  const tarch::la::Vector<DIMENSIONS,double>& fineSize,
  int                                         fineLevel,
  int                                         coarseLevel,
  const tarch::la::Vector<DIMENSIONS, int>&   discreteNeighborPosition
) {
  tarch::la::Vector<DIMENSIONS,double> normalizedFinePosition = finePosition - _domainOffset;

  tarch::la::Vector<DIMENSIONS,double> epsilon(1e-12);
  tarch::la::Vector<DIMENSIONS,double> neighborPosition
    = normalizedFinePosition + tarch::la::multiplyComponents(discreteNeighborPosition.convertScalar<double>(), fineSize);
  tarch::la::Vector<DIMENSIONS,double> coarseSize = _domainSize / std::pow(3.0, coarseLevel-1);

  tarch::la::Vector<DIMENSIONS,double> projectedNeighborPosition
    = tarch::la::multiplyComponents(neighborPosition, tarch::la::invertEntries(coarseSize)) + epsilon;
  for(int d = 0; d < DIMENSIONS; d++) {
    projectedNeighborPosition(d) = floor(projectedNeighborPosition(d));
  }
  tarch::la::Vector<DIMENSIONS,int> coarseDiscreteNeighborPosition
    = projectedNeighborPosition.convertScalar<int>();
  tarch::la::Vector<DIMENSIONS,double> coarseNeighborPosition
    = tarch::la::multiplyComponents(coarseDiscreteNeighborPosition.convertScalar<double>(), coarseSize);

  return coarseNeighborPosition + _domainOffset;
}

bool peanoclaw::statistics::ParallelGridValidator::validateNeighborPatchOnSameRank(
    const peanoclaw::statistics::PatchDescriptionDatabase& database,
    const PatchDescription& patch,
    const tarch::la::Vector<DIMENSIONS,double>& neighborPosition,
    int level,
    const tarch::la::Vector<DIMENSIONS, int> discreteNeighborPosition
) {
  if(database.containsPatch(neighborPosition, level, patch.getRank())) {
#ifdef Parallel
    assertion4(
        !database.getPatch(neighborPosition, level, patch.getRank()).getIsRemote(),
        discreteNeighborPosition,
        patch.toString(),
        database.getPatch(neighborPosition, level, patch.getRank()).toString(),
        discreteNeighborPosition
    );
#endif
    return true;
  }
  return false;
}

bool peanoclaw::statistics::ParallelGridValidator::validateNeighborPatchOnRemoteRank(
    const peanoclaw::statistics::PatchDescriptionDatabase& database,
    const PatchDescription& patch,
    const tarch::la::Vector<DIMENSIONS,double>& neighborPosition,
    int level,
    const tarch::la::Vector<DIMENSIONS, int> discreteNeighborPosition
) {
#ifdef Parallel
  int remoteRank = patch.getAdjacentRanks(peano::utils::dLinearised(discreteNeighborPosition+1, 3));
  bool localPatchFound = database.containsPatch(neighborPosition, level, patch.getRank());
  bool remotePatchFound = database.containsPatch(neighborPosition, level, remoteRank);
  if(localPatchFound && remotePatchFound) {
    PatchDescription localPatch = database.getPatch(neighborPosition, level, patch.getRank());
    PatchDescription remotePatch = database.getPatch(neighborPosition, level, remoteRank);
    assertionEquals4(
        localPatch.getIsRemote(),
        !remotePatch.getIsRemote(),
        patch.toString(),
        localPatch.toString(),
        remotePatch.toString(),
        discreteNeighborPosition
    );
    if(!patch.getIsRemote()) {
      assertion4(
          localPatch.getIsRemote(),
          patch.toString(),
          localPatch.toString(),
          remotePatch.toString(),
          discreteNeighborPosition
      );
      assertion4(
          !remotePatch.getIsRemote(),
          patch.toString(),
          localPatch.toString(),
          remotePatch.toString(),
          discreteNeighborPosition
      );
    }
    return true;
  } else if (localPatchFound && !remotePatchFound) {
    PatchDescription localPatch = database.getPatch(neighborPosition, level, patch.getRank());
    if(localPatch.getIsRemote() == patch.getIsRemote()) {
      //Forking
      return true;
    } else {
      logError(
          "validateNeighborPatch(...)",
          "Only local patch found! " << std::endl
          << "patch: " << patch.toString() << std::endl
          << "localPatch: " << localPatch.toString() << std::endl
          << ", discreteNeighborPosition=" << discreteNeighborPosition
      );
      assertionFail("Only local patch found!");
    }
  } else if (!localPatchFound && remotePatchFound) {
    //For a remote patch not all neighboring patches need to exist
    //on the same node.
    if(!patch.getIsRemote()) {
      PatchDescription remotePatch = database.getPatch(neighborPosition, level, remoteRank);
      logError(
          "validateNeighborPatch(...)",
          "Only remote patch found! " << std::endl
          << "patch: " << patch.toString() << std::endl
          << "remote patch: " << remotePatch.toString() << std::endl
          << "discreteNeighborPosition=" << discreteNeighborPosition
      );
      assertionFail("Only remote patch found!");
    }
  }
  return false;
#else
  assertionFail("Should not be called in non-parallel run.");
#endif
}

peanoclaw::statistics::ParallelGridValidator::ParallelGridValidator(
  tarch::la::Vector<DIMENSIONS,double> domainOffset,
  tarch::la::Vector<DIMENSIONS,double> domainSize,
  bool                                 useDimensionalSplittingOptimization
) : _domainOffset(domainOffset),
    _domainSize(domainSize),
    _useDimensionalSplittingOptimization(useDimensionalSplittingOptimization
) {
}

void peanoclaw::statistics::ParallelGridValidator::validateNeighborPatch(
//    const peanoclaw::statistics::PatchDescriptionDatabase& database,
    const PatchDescription& patch,
    const tarch::la::Vector<DIMENSIONS, int> discreteNeighborPosition
) {
  #ifdef Parallel
  if(
    patch.getSkipGridIterations() != 0 ||
    (patch.getAdjacentRanks(peano::utils::dLinearised(discreteNeighborPosition+1, 3)) == -2 &&
    (patch.getIsRemote() || patch.getAdjacentRanks((THREE_POWER_D-1)/2) != patch.getRank()))
  ) {
    //Don't validate patches that are remote or if they've been forked to a different
    //node, if their adjacent ranks are not set correctly
    return;
  }
  assertion3(
    patch.getAdjacentRanks(peano::utils::dLinearised(discreteNeighborPosition+1, 3)) != -2,
    patch.toString(),
    tarch::parallel::Node::getInstance().getRank(),
    discreteNeighborPosition
  );
  #endif


  bool foundNeighborPatch = false;
  for(int level = patch.getLevel(); level >= 0; level--) {
    tarch::la::Vector<DIMENSIONS, double> neighborPosition = patch.getPosition();
    neighborPosition = getNeighborPositionOnLevel(
      patch.getPosition(),
      patch.getSize(),
      patch.getLevel(),
      level,
      discreteNeighborPosition
    );

    if (tarch::la::oneGreater(_domainOffset, neighborPosition)
    || tarch::la::oneGreaterEquals(neighborPosition, _domainOffset + _domainSize)
    ) {
      //Neighbor outside domain
      foundNeighborPatch = true;
    } else {
      if(patch.getAdjacentRanks(peano::utils::dLinearised(discreteNeighborPosition+1, 3)) == patch.getRank()) {
        //Validate patch on same rank
        foundNeighborPatch |= validateNeighborPatchOnSameRank(
            _descriptions,
            patch,
            neighborPosition,
            level,
            discreteNeighborPosition
        );
      } else {
        //Validate patch on remote rank
        foundNeighborPatch |= validateNeighborPatchOnRemoteRank(
            _descriptions,
            patch,
            neighborPosition,
            level,
            discreteNeighborPosition
        );
      }
    }

    if(foundNeighborPatch) {
      break;
    }
  }

  if(!foundNeighborPatch) {
    logError("validateNeighborPatch(...)", "Neighbor patch not found for "
        << patch.toString()
        << ", discreteNeighborPosition=" << discreteNeighborPosition
        << ", neighborPosition=" << getNeighborPositionOnLevel(
            patch.getPosition(),
            patch.getSize(),
            patch.getLevel(),
            patch.getLevel(),
            discreteNeighborPosition
          )
    );
    assertionFail("");
    return;
  }
}

void peanoclaw::statistics::ParallelGridValidator::validatePatches() {
  //Check adjacency information
  for(PatchDescriptionDatabase::MapType::iterator i = _descriptions.begin();
      i != _descriptions.end();
      i++) {
    if(_useDimensionalSplittingOptimization) {
      for(int dimension = 0; dimension < DIMENSIONS; dimension++) {
        for(int direction = -1; direction <= 1; direction+=2) {
          tarch::la::Vector<DIMENSIONS, int> neighborIndex(0);
          neighborIndex(dimension) = direction;
          validateNeighborPatch(
              i->second,
              neighborIndex
          );
        }
      }
    } else {
      for(int neighborIndex = 0; neighborIndex < THREE_POWER_D; neighborIndex++) {
        if(neighborIndex != (THREE_POWER_D-1)/2) {
          validateNeighborPatch(
              i->second,
              peano::utils::dDelinearised(neighborIndex, 3) - 1
          );
        }
      }
    }
  }
}

void peanoclaw::statistics::ParallelGridValidator::findAdjacentPatches(
    const peanoclaw::Vertex&                         fineGridVertex,
    const tarch::la::Vector<DIMENSIONS,double>&      fineGridX,
    int                                              level,
    int                                              localRank
) {
  logInfo("findAdjacentPatches", "Finding patches for vertex " << fineGridX << ", " << level
          << ", hanging=" << fineGridVertex.isHangingNode()
          << ": " << fineGridVertex.toString());

  for(int i = 0; i < TWO_POWER_D; i++) {
    if(fineGridVertex.getAdjacentCellDescriptionIndexInPeanoOrder(i) != -1) {
      int cellDescriptionIndex = fineGridVertex.getAdjacentCellDescriptionIndexInPeanoOrder(i);

      CellDescription& cellDescription = CellDescriptionHeap::getInstance().getData(cellDescriptionIndex).at(0);
      Patch adjacentPatch(cellDescription);

      if(adjacentPatch.getLevel() == level) {
        PatchDescription patchDescription;
        patchDescription.setAdjacentRanks(tarch::la::Vector<THREE_POWER_D,int>(-2));
        if(_descriptions.containsPatch(adjacentPatch.getPosition(), adjacentPatch.getLevel(), localRank)) {
          patchDescription = _descriptions.getPatch(adjacentPatch.getPosition(), adjacentPatch.getLevel(), localRank);
          assertionNumericalEquals2(patchDescription.getPosition(), adjacentPatch.getPosition(), patchDescription.toString(), adjacentPatch);
          assertionNumericalEquals2(patchDescription.getSize(), adjacentPatch.getSize(), patchDescription.toString(), adjacentPatch);
          assertionEquals2(patchDescription.getLevel(), adjacentPatch.getLevel(), patchDescription.toString(), adjacentPatch);
        }
        #ifdef Parallel
        if(_descriptions.containsPatch(adjacentPatch.getPosition(), adjacentPatch.getLevel(), tarch::parallel::Node::getInstance().getRank())) {
          patchDescription = _descriptions.getPatch(adjacentPatch.getPosition(), adjacentPatch.getLevel(), tarch::parallel::Node::getInstance().getRank());
          assertionNumericalEquals2(patchDescription.getPosition(), adjacentPatch.getPosition(), patchDescription.toString(), adjacentPatch);
          assertionNumericalEquals2(patchDescription.getSize(), adjacentPatch.getSize(), patchDescription.toString(), adjacentPatch);
          assertionEquals2(patchDescription.getLevel(), adjacentPatch.getLevel(), patchDescription.toString(), adjacentPatch);
        }
        #endif
        patchDescription.setPosition(adjacentPatch.getPosition());
        patchDescription.setSize(adjacentPatch.getSize());
        patchDescription.setLevel(adjacentPatch.getLevel());
        patchDescription.setIsReferenced(true);
        patchDescription.setIsVirtual(adjacentPatch.isVirtual());
        patchDescription.setCellDescriptionIndex(adjacentPatch.getCellDescriptionIndex());
        patchDescription.setSkipGridIterations(adjacentPatch.shouldSkipNextGridIteration() ? -1 : 0);

        #ifdef Parallel
        if(localRank == tarch::parallel::Node::getInstance().getRank()) {
          patchDescription.setIsRemote(adjacentPatch.isRemote());
        } else {
          patchDescription.setIsRemote(fineGridVertex.getAdjacentRanks()(i) != localRank);
        }
        patchDescription.setAdjacentRanks((THREE_POWER_D-1)/2, tarch::parallel::Node::getInstance().getRank());
        patchDescription.setRank(localRank);

        //Get adjacency information
        tarch::la::Vector<DIMENSIONS, int> localPosition = peano::utils::dDelinearised( i, 2 );
        for(int vertexBasedNeighborIndex = 0; vertexBasedNeighborIndex < TWO_POWER_D; vertexBasedNeighborIndex++) {
          if(i != vertexBasedNeighborIndex) {
            tarch::la::Vector<DIMENSIONS, int> neighborPosition
            = peano::utils::dDelinearised( vertexBasedNeighborIndex, 2 ) - localPosition;
            int patchBasedNeighborIndex = peano::utils::dLinearised(neighborPosition + 1, 3);

            //TODO unterweg debug
            logDebug("findAdjacentPatches", "Setting adjacent ranks"
                << ", x=" << fineGridX
                << ", index=" << i
                << ", localPosition=" << localPosition
                << ", vertexBasedIndex=" << vertexBasedNeighborIndex
                << ", vertexBasedPosition=" << peano::utils::dDelinearised( vertexBasedNeighborIndex, 2 )
            << ", neighborPosition=" << neighborPosition
            << ", patchBasedIndex=" << patchBasedNeighborIndex
            << ", ranks=" << fineGridVertex.getAdjacentRanks()
            << ", rank=" << fineGridVertex.getAdjacentRanks()(vertexBasedNeighborIndex));

            patchDescription.setAdjacentRanks(
                patchBasedNeighborIndex,
                fineGridVertex.getAdjacentRanks()(vertexBasedNeighborIndex)
            );
          }
        }
        #else
        patchDescription.setRank(0);
        patchDescription.setAdjacentRanks(0);
        #endif

        _descriptions.insertPatch(patchDescription);
        assertion2(_descriptions.containsPatch(adjacentPatch.getPosition(), adjacentPatch.getLevel(), localRank), patchDescription.toString(), adjacentPatch);

        //Keep local copy when moving patch to different node (i.e. insert another copy of
        //the patch description to the database)
        #ifdef Parallel
        if(localRank != tarch::parallel::Node::getInstance().getRank()) {
          PatchDescription localDescription = patchDescription;
          localDescription.setAdjacentRanks((THREE_POWER_D-1)/2, tarch::parallel::Node::getInstance().getRank());
          localDescription.setRank(tarch::parallel::Node::getInstance().getRank());
          localDescription.setIsRemote(fineGridVertex.getAdjacentRanks()(i) != tarch::parallel::Node::getInstance().getRank());
          _descriptions.insertPatch(localDescription);
        }
        #endif
      }
    }
  }
}

void peanoclaw::statistics::ParallelGridValidator::deletePatchIfNotRemote(
  const tarch::la::Vector<DIMENSIONS,double>&      position,
  int                                              level
) {
  #ifdef Parallel
  int localRank = tarch::parallel::Node::getInstance().getRank();
  #else
  int localRank = 0;
  #endif

  if(_descriptions.containsPatch(
    position,
    level,
    localRank
  )) {
    PatchDescription patchDescription = _descriptions.getPatch(
      position,
      level,
      localRank
    );

    #ifdef Parallel
    if(!patchDescription.getIsRemote()) {
      _descriptions.erasePatch(
        position,
        level,
        tarch::parallel::Node::getInstance().getRank()
      );
    }
    #else
    _descriptions.erasePatch(
      position,
      level,
      0
    );
    #endif
  }
}

std::vector<peanoclaw::records::PatchDescription> peanoclaw::statistics::ParallelGridValidator::getAllPatches() {
  return _descriptions.getAllPatches();
}

