/*
 * SubgridCommunicator.cpp
 *
 *  Created on: Dec 4, 2013
 *      Author: kristof
 */
#include "peanoclaw/parallel/SubgridCommunicator.h"

#include "peanoclaw/Area.h"
#include "peanoclaw/ParallelSubgrid.h"

#include "tarch/Assertions.h"

tarch::logging::Log peanoclaw::parallel::SubgridCommunicator::_log("peanoclaw::parallel::SubgridCommunicator");

std::vector<peanoclaw::records::CellDescription> peanoclaw::parallel::SubgridCommunicator::receiveCellDescription() {
  logTraceIn("receiveCellDescription()");

  std::vector<peanoclaw::records::CellDescription> remoteCellDescriptionVector;
  if(_packCommunication) {

//    Serialization::ReceiveBuffer& recvbuffer = peano::parallel::SerializationMap::getInstance().getReceiveBuffer(_remoteRank);
//    assertion1(recvbuffer.isBlockAvailable(), "cannot read heap data from Serialization Buffer - not enough blocks");
//
//    Serialization::Block block = recvbuffer.nextBlock();
//
//    size_t numberOfCellDescriptions = block.size() / sizeof(CellDescription::Packed);
//
//    //assertion1(numberOfCellDescriptions > 0, "no cell descriptions, huh? we always send one");
//
//    remoteCellDescriptionVector.resize(numberOfCellDescriptions);
//
//    int block_position = 0;
//    //std::cout << " ||||||| unpacking " << numberOfCellDescriptions << " cell descriptions " << std::endl;
//    for (size_t i=0; i < numberOfCellDescriptions; i++) {
//        CellDescription::Packed packed;
//        MPI_Unpack(block.data(), block.size(), &block_position, &packed, 1, CellDescription::Packed::Datatype, MPI_COMM_WORLD );
//        remoteCellDescriptionVector[i] = packed.convert();
//    }
  } else {
    remoteCellDescriptionVector = CellDescriptionHeap::getInstance().receiveData(_remoteRank, _position, _level, peano::heap::NeighbourCommunication);
  }

  logTraceOut("receiveCellDescription()");
  return remoteCellDescriptionVector;
}

peanoclaw::parallel::SubgridCommunicator::SubgridCommunicator(
  int                                         remoteRank,
  const tarch::la::Vector<DIMENSIONS,double>& position,
  int                                         level,
  peano::heap::MessageType                    messageType,
  bool                                        onlySendOverlappedCells,
  bool                                        packCommunication
) : _remoteRank(remoteRank),
    _position(position),
    _level(level),
    _messageType(messageType),
    _onlySendOverlappedCells(onlySendOverlappedCells),
    _packCommunication(packCommunication) {

}

void peanoclaw::parallel::SubgridCommunicator::sendSubgrid(Patch& subgrid) {
  logTraceInWith1Argument("sendSubgrid(Subgrid)", subgrid);

  sendCellDescription(subgrid.getCellDescriptionIndex());

  if(subgrid.getUIndex() != -1) {
    if(_onlySendOverlappedCells) {
      sendOverlappedCells(subgrid);
    } else {
      sendDataArray(subgrid.getUIndex());
    }
  } else if(_messageType == peano::heap::NeighbourCommunication) {
    sendPaddingDataArray();
  } else {
    //Don't send anything
  }

  logTraceOut("sendSubgrid(Subgrid)");
}

void peanoclaw::parallel::SubgridCommunicator::sendPaddingSubgrid() {
  logTraceIn("sendPaddingSubgrid()");

  sendPaddingCellDescription();
  sendPaddingDataArray();

  logTraceOut("sendPaddingSubgrid()");
}

void peanoclaw::parallel::SubgridCommunicator::sendCellDescription(int cellDescriptionIndex)
{
  logTraceInWith1Argument("sendCellDescription", cellDescriptionIndex);
  #if defined(Asserts) && defined(Parallel)
  CellDescription& cellDescription = CellDescriptionHeap::getInstance().getData(cellDescriptionIndex).at(0);
  assertion1(!cellDescription.getIsPaddingSubgrid(), cellDescription.toString());
  #endif

  if(_packCommunication) {
    if (_messageType == peano::heap::NeighbourCommunication) {
//      Serialization::SendBuffer& sendbuffer = peano::parallel::SerializationMap::getInstance().getSendBuffer(_remoteRank);
//
//      std::vector<CellDescription>& localCellDescriptionVector = CellDescriptionHeap::getInstance().getData(cellDescriptionIndex);
//
//      size_t numberOfCellDescriptions = localCellDescriptionVector.size();
//      int cellDescriptionSize = sizeof(CellDescription::Packed);
//      Serialization::Block block = sendbuffer.reserveBlock(cellDescriptionSize*numberOfCellDescriptions);
//
//      int block_position = 0;
//      //std::cout << " ||||||| packing " << numberOfCellDescriptions << " cell descriptions " << std::endl;
//      for (size_t i=0; i < numberOfCellDescriptions; i++) {
//          CellDescription::Packed packed = localCellDescriptionVector[i].convert();
//          MPI_Pack(&packed, 1, CellDescription::Packed::Datatype, block.data(), block.size(), &block_position, MPI_COMM_WORLD );
//      }
    } else {
        CellDescriptionHeap::getInstance().sendData(cellDescriptionIndex, _remoteRank, _position, _level, _messageType);
    }
  } else {
    CellDescriptionHeap::getInstance().sendData(cellDescriptionIndex, _remoteRank, _position, _level, _messageType);
  }
  logTraceOut("sendCellDescription");
}

void peanoclaw::parallel::SubgridCommunicator::sendPaddingCellDescription() {
  logTraceIn("sendPaddingCellDescription");
  int cellDescriptionIndex = CellDescriptionHeap::getInstance().createData();

  if(_packCommunication) {

    if (_messageType == peano::heap::NeighbourCommunication) {
//      std::vector<CellDescription>& localCellDescriptionVector = CellDescriptionHeap::getInstance().getData(cellDescriptionIndex);
//
//      Serialization::SendBuffer& sendbuffer = peano::parallel::SerializationMap::getInstance().getSendBuffer(_remoteRank);
//
//      //    size_t numberOfCellDescriptions = 1;
//      //    int cellDescriptionSize = sizeof(CellDescription::Packed);
//      Serialization::Block block = sendbuffer.reserveBlock(0);
//
//      //    int block_position = 0;
//      //std::cout << " ||||||| packing " << numberOfCellDescriptions << " padded cell descriptions " << std::endl;
//      //    for (size_t i=0; i < numberOfCellDescriptions; i++) {
//      //        CellDescription::Packed packed; // padding patch
//      //        MPI_Pack(&packed, 1, CellDescription::Packed::Datatype, block.data(), block.size(), &block_position, MPI_COMM_WORLD );
//      //    }
//
//      //CellDescriptionHeap::getInstance().sendData(cellDescriptionIndex, _remoteRank, _position, _level, _messageType);
    } else {
      CellDescriptionHeap::getInstance().sendData(cellDescriptionIndex, _remoteRank, _position, _level, _messageType);
    }

  } else {

    CellDescriptionHeap::getInstance().sendData(cellDescriptionIndex, _remoteRank, _position, _level, _messageType);

  }

  CellDescriptionHeap::getInstance().deleteData(cellDescriptionIndex);
  logTraceOut("sendPaddingCellDescription");
}

void peanoclaw::parallel::SubgridCommunicator::sendDataArray(int index) {
  logTraceInWith3Arguments("sendDataArray", index, _position, _level);

  DataHeap::getInstance().sendData(index, _remoteRank, _position, _level, _messageType);

  logTraceOut("sendDataArray");
}

void peanoclaw::parallel::SubgridCommunicator::sendOverlappedCells(
  Patch& subgrid
) {
  logTraceInWith1Argument("sendOverlappedCellsOfDataArray(...)", subgrid);

  ParallelSubgrid parallelSubgrid(subgrid);

  Area areas[THREE_POWER_D_MINUS_ONE];
  int numberOfAreas = Area::getAreasOverlappedByRemoteGhostlayers(
    parallelSubgrid.getAdjacentRanks(),
    parallelSubgrid.getOverlapOfRemoteGhostlayers(),
    subgrid.getSubdivisionFactor(),
    _remoteRank,
    areas
  );

  //TODO unterweg debug
//  std::cout << "Sending " << numberOfAreas << " areas from " << tarch::parallel::Node::getInstance().getRank() << " to " << _remoteRank
//      << " for subgrid " << subgrid << std::endl;
//  for(int i = 0; i < numberOfAreas; i++) {
//    std::cout << "\t" << areas[i]._offset << ", " << areas[i]._size << std::endl;
//  }

  int numberOfCells = 0;
  for(int i = 0; i < numberOfAreas; i++) {
    numberOfCells += tarch::la::volume(areas[i]._size);
  }

  //Allocate data array
  int numberOfEntries = numberOfCells * subgrid.getUnknownsPerSubcell() * 2;
  int temporaryIndex = DataHeap::getInstance().createData(numberOfEntries, numberOfEntries);
  std::vector<Data>& temporaryArray = DataHeap::getInstance().getData(temporaryIndex);

  int entry = 0;
  for(int i = 0; i < numberOfAreas; i++) {
    Area& area = areas[i];

    //U new
    dfor(subcellIndex, area._size) {
      int linearIndex = subgrid.getLinearIndexUNew(area._offset + subcellIndex);
      for(int unknown = 0; unknown < subgrid.getUnknownsPerSubcell(); unknown++) {
        temporaryArray[entry++] = subgrid.getValueUNew(linearIndex, unknown);
      }
    }

    //U old
    dfor(subcellIndex, area._size) {
      int linearIndex = subgrid.getLinearIndexUOld(area._offset + subcellIndex);
      for(int unknown = 0; unknown < subgrid.getUnknownsPerSubcell(); unknown++) {
        temporaryArray[entry++] = subgrid.getValueUOld(linearIndex, unknown);
      }
    }
  }

  //Send
  sendDataArray(temporaryIndex);

  assertion1(DataHeap::getInstance().isValidIndex(temporaryIndex), temporaryIndex);

  DataHeap::getInstance().deleteData(temporaryIndex);

  logTraceOut("sendOverlappedCellsOfDataArray(...)");
}

void peanoclaw::parallel::SubgridCommunicator::sendPaddingDataArray() {
  logTraceInWith2Arguments("sendPaddingDataArray", _position, _level);
  int index = DataHeap::getInstance().createData();
  sendDataArray(index);
  DataHeap::getInstance().deleteData(index);

  logTraceOut("sendPaddingDataArray");
}

void peanoclaw::parallel::SubgridCommunicator::receivePaddingSubgrid() {
  logTraceIn("receivePaddingSubgrid()");
  #ifdef Parallel
  logDebug("", "Receiving padding patch from " << _remoteRank << " at " << _position << " on level " << _level);

  std::vector<CellDescription> remoteCellDescriptionVector = receiveCellDescription();

  assertionEquals4(remoteCellDescriptionVector.size(), 0, _position, _level, _remoteRank, remoteCellDescriptionVector[0].toString());

  //UNew
  receiveDataArray();
  #endif
  logTraceOut("receivePaddingSubgrid()");
}

int peanoclaw::parallel::SubgridCommunicator::receiveDataArray() {
  logTraceIn("receiveDataArray");

  int localIndex = DataHeap::getInstance().createData();

  DataHeap::getInstance().receiveData(
    localIndex,
    _remoteRank,
    _position,
    _level,
    _messageType
  );

  logTraceOut("receiveDataArray");
  return localIndex;
}

void peanoclaw::parallel::SubgridCommunicator::receiveOverlappedCells(
  const CellDescription& remoteCellDescription,
  Patch&                 subgrid
) {
  logTraceInWith1Argument("receiveOverlappedCells", subgrid);

  Area areas[THREE_POWER_D_MINUS_ONE];
  int numberOfAreas = Area::getAreasOverlappedByRemoteGhostlayers(
    remoteCellDescription.getAdjacentRanks(),
    remoteCellDescription.getOverlapByRemoteGhostlayer(),
    remoteCellDescription.getSubdivisionFactor(),
    tarch::parallel::Node::getInstance().getRank(),
    areas
  );

  //TODO unterweg debug
//  std::cout << "Receiving " << numberOfAreas << " areas from " << _remoteRank << " on " << tarch::parallel::Node::getInstance().getRank()
//      << " for subgrid " << subgrid.getPosition() << ", " << subgrid.getSize()
//      << ", adj:" << remoteCellDescription.getAdjacentRanks() << ", overlap:" << remoteCellDescription.getOverlapByRemoteGhostlayer() << ": " << std::endl;
//  for(int i = 0; i < numberOfAreas; i++) {
//    std::cout << "\t" << areas[i]._offset << ", " << areas[i]._size << std::endl;
//  }


  #ifdef Asserts
  int numberOfCells = 0;
  for(int i = 0; i < numberOfAreas; i++) {
    numberOfCells += tarch::la::volume(areas[i]._size);
  }
  #endif

  //Allocate data array
  std::vector<Data> remoteData = DataHeap::getInstance().receiveData(
                                    _remoteRank,
                                    _position,
                                    _level,
                                    _messageType
                                  );

  assertion1(subgrid.getUIndex() != -1, subgrid);
  int entry = 0;
  for(int i = 0; i < numberOfAreas; i++) {
    Area& area = areas[i];

    //U new
    dfor(subcellIndex, area._size) {
      int linearIndex = subgrid.getLinearIndexUNew(area._offset + subcellIndex);
      for(int unknown = 0; unknown < remoteCellDescription.getUnknownsPerSubcell(); unknown++) {
        subgrid.setValueUNewAndResize(linearIndex, unknown, remoteData[entry++].getU());
      }

      //TODO unterweg debug
//      std::cout << "Setting cell " << (area._offset + subcellIndex) << std::endl;

      assertion3(tarch::la::greater(subgrid.getValueUNew(linearIndex, 0), 0.0), subgrid, subcellIndex, subgrid.getValueUNew(linearIndex, 0));
    }

    //U old
    dfor(subcellIndex, area._size) {
      int linearIndex = subgrid.getLinearIndexUOld(area._offset + subcellIndex);
      for(int unknown = 0; unknown < remoteCellDescription.getUnknownsPerSubcell(); unknown++) {
        subgrid.setValueUOldAndResize(linearIndex, unknown, remoteData[entry++].getU());
      }

      assertion3(tarch::la::greater(subgrid.getValueUOld(linearIndex, 0), 0.0), subgrid, subcellIndex, subgrid.getValueUOld(linearIndex, 0));
    }
  }

//  assertion4(subgrid.getUSize() >= entry, entry, subgrid.getUSize(), subgrid, remoteCellDescription.getSubdivisionFactor());
//  assertion2(DataHeap::getInstance().getData(subgrid.getUIndex()).size() >= entry, DataHeap::getInstance().getData(subgrid.getUIndex()).size(), entry);

  logTraceOut("receiveOverlappedCells");
}

void peanoclaw::parallel::SubgridCommunicator::deleteArraysFromSubgrid(
  Patch& subgrid
) {
  logTraceInWith1Argument("deleteArraysFromSubgrid", cellDescriptionIndex);
  if(subgrid.getUIndex() != -1) {
    DataHeap::getInstance().deleteData(subgrid.getUIndex());
  }
  logTraceOut("deleteArraysFromSubgrid");
}
