/*
 * SchemeExecutor.h
 *
 *  Created on: Sep 22, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_SOLVER_EULER3D_SCHEMEEXECUTOR_H_
#define PEANOCLAW_SOLVER_EULER3D_SCHEMEEXECUTOR_H_

#include "peanoclaw/Patch.h"
#include "peanoclaw/grid/SubgridAccessor.h"

#include "Uni/EulerEquations/RoeSolver"
#include "Uni/StructuredGrid/GridIterator"

#include "peano/utils/Dimensions.h"
#include "tarch/la/Vector.h"
#include "tarch/logging/Log.h"

#include <map>

#ifdef PEANOCLAW_EULER3D
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/compat/thread>
#endif

namespace peanoclaw {
  namespace solver {
    namespace euler3d {
      class SchemeExecutor;
    }
  }
}

class peanoclaw::solver::euler3d::SchemeExecutor {

  private:
    /**
     * Logging device.
     */
    static tarch::logging::Log _log;

    peanoclaw::Patch& _subgrid;
    peanoclaw::grid::SubgridAccessor& _accessor;
    Uni::EulerEquations::RoeSolver _scheme;

    #ifdef PEANOCLAW_EULER3D
    std::map<tbb::tbb_thread::id, int> _cellUpdatesPerThread;
    #endif

    double _dt;

    int _leftCellOffset;
    int _rightCellOffset;
    int _bottomCellOffset;
    int _topCellOffset;
    int _backCellOffset;
    int _frontCellOffset;
    int _uOldOffset; //Offset in terms of double entries between uOld and uNew (due to ghostlayer)

    int _numberOfUnknowns;

    tarch::la::Vector<DIMENSIONS,int> _stride;

    double* _uOldArray;

    void solveTimestep(
      int linearSubcellIndex
    );

  public:
    double _maxLambda;
    double _maxDensity;
    double _minDensity;

    #ifdef PEANOCLAW_EULER3D
    SchemeExecutor(
      SchemeExecutor& other, tbb::split
    );
    #endif

    SchemeExecutor(
      peanoclaw::Patch& subgrid,
      double const& timeStepSize
    );

    #ifdef PEANOCLAW_EULER3D
    void operator()(
      tbb::blocked_range<int> const& range
    );
    #endif

    void operator()(
      const tarch::la::Vector<DIMENSIONS,int>& subcellIndex
    );

    void join(SchemeExecutor const& other);

    /**
     * Returns the maximum wave speed for the last timestep.
     */
    double getMaximumLambda() const;

    /**
     * Logs the statistics that have been assembled up to now.
     */
    void logStatistics() const;

};

#endif /* PEANOCLAW_SOLVER_EULER3D_SCHEMEEXECUTOR_H_ */
