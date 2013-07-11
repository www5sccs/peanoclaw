/*
 * PatchDescriptionDatabase.h
 *
 *  Created on: Jun 25, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_STATISTICS_PATCHDESCRIPTIONDATABASE_H_
#define PEANOCLAW_STATISTICS_PATCHDESCRIPTIONDATABASE_H_

#include "peanoclaw/records/PatchDescription.h"

#include "tarch/la/VectorCompare.h"

#include <map>
#include <vector>

namespace peanoclaw {
  namespace statistics {
    class PatchDescriptionDatabase;
  }
}

#define DIMENSIONS_PLUS_TWO (DIMENSIONS+2)

class peanoclaw::statistics::PatchDescriptionDatabase {
  private:
    typedef peanoclaw::records::PatchDescription PatchDescription;
  public:
    typedef std::map<tarch::la::Vector<DIMENSIONS_PLUS_TWO,double>, PatchDescription, tarch::la::VectorCompare<DIMENSIONS_PLUS_TWO> > MapType;

  private:

    MapType _database;

    tarch::la::Vector<DIMENSIONS_PLUS_TWO, double> createKey(tarch::la::Vector<DIMENSIONS, double> position, int level, int rank) const;

  public:
    bool containsPatch(
      tarch::la::Vector<DIMENSIONS, double> position,
      int level,
      int rank
    ) const;

    PatchDescription getPatch(
      tarch::la::Vector<DIMENSIONS, double> position,
      int level,
      int rank
    ) const;

    void insertPatch(
      const PatchDescription& patch
    );

    void erasePatch(
      tarch::la::Vector<DIMENSIONS, double> position,
      int level,
      int rank
    );

    std::vector<PatchDescription> getAllPatches();

    /**
     * Returns the number of entries in the database.
     */
    size_t size() const;

    MapType::iterator begin();

    MapType::iterator end();

    void clear();
};



#endif /* PEANOCLAW_STATISTICS_PATCHDESCRIPTIONDATABASE_H_ */
