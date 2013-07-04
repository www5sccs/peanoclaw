/*
 * PatchDescriptionDatabase.cpp
 *
 *  Created on: Jun 25, 2013
 *      Author: kristof
 */
#include "peanoclaw/statistics/PatchDescriptionDatabase.h"

tarch::la::Vector<DIMENSIONS_PLUS_TWO, double> peanoclaw::statistics::PatchDescriptionDatabase::createKey(
  tarch::la::Vector<DIMENSIONS, double> position,
  int level,
  int rank
) const {
  tarch::la::Vector<DIMENSIONS_PLUS_TWO, double> key;
  for(int d = 0; d < DIMENSIONS; d++) {
    key(d) = position(d);
  }
  key(DIMENSIONS) = level;
  key(DIMENSIONS+1) = rank;

  return key;
}

bool peanoclaw::statistics::PatchDescriptionDatabase::containsPatch(
  tarch::la::Vector<DIMENSIONS, double> position,
  int level,
  int rank
) const {
  return _database.find(createKey(position, level, rank)) != _database.end();
}

peanoclaw::records::PatchDescription peanoclaw::statistics::PatchDescriptionDatabase::getPatch(
  tarch::la::Vector<DIMENSIONS, double> position,
  int level,
  int rank
) const {
  return _database.find(createKey(position, level, rank))->second;
}

void peanoclaw::statistics::PatchDescriptionDatabase::insertPatch(
  const PatchDescription& patch
) {
  _database[createKey(patch.getPosition(), patch.getLevel(), patch.getRank())] = patch;
}

void peanoclaw::statistics::PatchDescriptionDatabase::erasePatch(
  tarch::la::Vector<DIMENSIONS, double> position,
  int level,
  int rank
) {
  _database.erase(createKey(position, level, rank));
}

std::vector<peanoclaw::records::PatchDescription> peanoclaw::statistics::PatchDescriptionDatabase::getAllPatches() {
  std::vector<PatchDescription> descriptionVector;

  for(
      std::map<tarch::la::Vector<DIMENSIONS_PLUS_TWO,double>, PatchDescription, tarch::la::VectorCompare<DIMENSIONS_PLUS_TWO> >::iterator i = _database.begin();
      i != _database.end();
      i++
  ) {
    descriptionVector.push_back(i->second);
  }

  return descriptionVector;
}

size_t peanoclaw::statistics::PatchDescriptionDatabase::size() const {
  return _database.size();
}

void peanoclaw::statistics::PatchDescriptionDatabase::clear() {
  _database.clear();
}
