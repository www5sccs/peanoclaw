// This file is part of the Peano project. For conditions of distribution and 
// use, please see the copyright notice at www.peano-framework.org
#ifndef _PEANOCLAW_STATE_H_ 
#define _PEANOCLAW_STATE_H_

#include "records/State.h"
#include "peano/grid/State.h"

#include "peano/grid/Checkpoint.h"

#include "Numerics.h"

#include "statistics/LevelInformation.h"
#include "statistics/Probe.h"

namespace peanoclaw { 
      class State;
      /**
       * Forward declaration
       */
      class Vertex;
      /**
       * Forward declaration
       */
      class Cell;
      
      namespace repositories {
        /** 
         * Forward declaration
         */
        class RepositoryArrayStack;
        class RepositorySTDStack;
      }
}


/**
 * Blueprint for solver state.
 * 
 * This file has originally been created by the PDT and may be manually extended to 
 * the needs of your application. We do not recommend to remove anything!
 */
class peanoclaw::State: public peano::grid::State< peanoclaw::records::State > { 
  private: 
    typedef class peano::grid::State< peanoclaw::records::State >  Base;

    /**
     * Needed for checkpointing.
     */
    friend class peanoclaw::repositories::RepositoryArrayStack;
    friend class peanoclaw::repositories::RepositorySTDStack;
  
    void writeToCheckpoint( peano::grid::Checkpoint<Vertex,Cell>&  checkpoint ) const;    
    void readFromCheckpoint( const peano::grid::Checkpoint<Vertex,Cell>&  checkpoint );    
  
  /**
   * Logging device.
   */
  static tarch::logging::Log _log;

  Numerics* _numerics;

  std::vector<peanoclaw::statistics::Probe> _probeList;

  std::vector<peanoclaw::statistics::LevelInformation> _levelStatisticsForLastGridIteration;
  std::vector<peanoclaw::statistics::LevelInformation> _totalLevelStatistics;


  public:
    /**
     * Default Constructor
     *
     * This constructor is required by the framework's data container. Do not 
     * remove it.
     */
    State();

    /**
     * Constructor
     *
     * This constructor is required by the framework's data container. Do not 
     * remove it. It is kind of a copy constructor that converts an object which 
     * comprises solely persistent attributes into a full attribute. This very 
     * functionality is implemented within the super type, i.e. this constructor 
     * has to invoke the correponsing super type's constructor and not the super 
     * type standard constructor.
     */
    State(const Base::PersistentState& argument);

      int getPlotNumber() const;

      void setPlotNumber(int plotNumber);

      int getSubPlotNumber() const;

      void setSubPlotNumber(int plotNumber);

      void setUnknownsPerSubcell(int unknownsPerSubcell);

      int getUnknownsPerSubcell() const;

      void setAuxiliarFieldsPerSubcell(int auxiliarFieldsPerSubcell);

      int getAuxiliarFieldsPerSubcell() const;

      void setDefaultSubdivisionFactor(const tarch::la::Vector<DIMENSIONS, int>& defaultSubdivisionFactor);

      tarch::la::Vector<DIMENSIONS, int> getDefaultSubdivisionFactor() const;

      void setDefaultGhostLayerWidth(int defaultGhostLayerWidth);

      int getDefaultGhostLayerWidth() const;

      tarch::la::Vector<DIMENSIONS, double> getInitialMinimalMeshWidth() const;

      void setInitialMinimalMeshWidth(const tarch::la::Vector<DIMENSIONS, double>& h);

      void setInitialTimestepSize(double initialTimestepSize);

      double getInitialTimestepSize() const;

      void setNumerics(peanoclaw::Numerics& numerics);

      peanoclaw::Numerics* getNumerics() const;

      void setProbeList(std::vector<peanoclaw::statistics::Probe> probeList);

      std::vector<peanoclaw::statistics::Probe>& getProbeList();

      void setGlobalTimestepEndTime(double globalTimestepEndTime);

      double getGlobalTimestepEndTime() const;

      void setAllPatchesEvolvedToGlobalTimestep(bool value);

      bool getAllPatchesEvolvedToGlobalTimestep() const;

      /**
       * Sets the offset and size of the rectangular domain.
       */
      void setDomain(const tarch::la::Vector<DIMENSIONS, double>& offset, const tarch::la::Vector<DIMENSIONS, double>& size);

      tarch::la::Vector<DIMENSIONS, double> getDomainOffset();

      tarch::la::Vector<DIMENSIONS, double> getDomainSize();

      /**
       * The maximum global time interval is the interval between the latest uOld and the newest uNew
       * in the whole grid. This method updates the global time interval, i.e. if the given
       * startMaximumLocalTimeInterval is smaller than the global one that is already stored, the
       * latter is overwritten otherwise this parameter is ignored. The same holds for
       * endMaximumLocalTimeInterval and the global end even though here we're looking for the
       * maximum of course.
       *
       * The minimum global time interval is the same but for the newest uOld and the latest uNew in
       * the whole grid.
       */
      void updateGlobalTimeIntervals(
        double startMaximumLocalTimeInterval,
        double endMaximumLocalTimeInterval,
        double startMinimumLocalTimeInterval,
        double endMinimumLocalTimeInterval
      );

      /**
       * For the search of the global time interval the values have to be set in the beginning
       * of a grid iteration. I.e. this method sets the start of the global maximum time interval
       * to double::max, the end to -double::max, the start of the global minimum time interval to
       * -double::max and the end to double::max.
       */
      void resetGlobalTimeIntervals();

      double getStartMaximumGlobalTimeInterval() const;

      double getEndMaximumGlobalTimeInterval() const;

      double getStartMinimumGlobalTimeInterval() const;

      double getEndMinimumGlobalTimeInterval() const;

      /**
       * Resets the total number of cell updates to zero so that the total
       * number can be accumulated.
       */
      void resetTotalNumberOfCellUpdates();

      double getTotalNumberOfCellUpdates() const;

      /**
       * Resets the minimal timestep for a minimum search in a grid iteration.
       */
      void resetMinimalTimestep();

      /**
       * Updates the minimal timestep. If the given argument is smaller than the
       * current minimal timestep the latter is set to the given value. Otherwise
       * nothing changes.
       */
      void updateMinimalTimestep(double timestep);

      /**
       * Returns the currently found minimal timestep.
       */
      double getMinimalTimestep() const;

      /**
       * Sets the level statistics for the last grid iteration.
       */
      void setLevelStatisticsForLastGridIteration(const std::vector<peanoclaw::statistics::LevelInformation>& levelStatistics);

      /**
       * Plots grid statistics for the last grid iteration
       */
      void plotStatisticsForLastGridIteration() const;

      /**
       * Plot grid statistics for the complete simulation
       */
      void plotTotalStatistics() const;

      void setAdditionalLevelsForPredefinedRefinement(int levels);

      int getAdditionalLevelsForPredefinedRefinement() const;

      void setIsInitializing(bool isInitializing);

      bool getIsInitializing() const;

      void setInitialRefinementTriggered(bool initialRefinementTriggered);

      bool getInitialRefinementTriggered() const;

      void setUseDimensionalSplitting(bool useDimensionalSplitting);

      bool useDimensionalSplitting() const;
};


#endif
