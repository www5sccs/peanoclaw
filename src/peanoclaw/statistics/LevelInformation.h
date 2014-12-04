#ifndef PEANO_APPLICATIONS_PEANOCLAW_STATISTICS_LEVELINFORMATION_H
#define PEANO_APPLICATIONS_PEANOCLAW_STATISTICS_LEVELINFORMATION_H

namespace peanoclaw {
  namespace statistics {
    struct LevelInformation;
  }
}

struct peanoclaw::statistics::LevelInformation {
  double _region;
  int _level;
  double _numberOfPatches;
  double _numberOfCells;
  double _numberOfCellUpdates;
  double _createdPatches;
  double _destroyedPatches;
  double _patchesBlockedDueToNeighbors;
  double _patchesBlockedDueToGlobalTimestep;
  double _patchesSkippingIteration;
  double _patchesCoarsening;

  LevelInformation();
};

#endif //PEANO_APPLICATIONS_PEANOCLAW_STATISTICS_LEVELINFORMATION_H
