// This file is part of the Peano project. For conditions of distribution and 
// use, please see the copyright notice at www.peano-framework.org
#ifndef _PEANOCLAW_REPOSITORIES_REPOSITORY_ARRAY_STD_H_ 
#define _PEANOCLAW_REPOSITORIES_REPOSITORY_ARRAY_STD_H_ 


#include "peanoclaw/repositories/Repository.h"
#include "peanoclaw/records/RepositoryState.h"

#include "peanoclaw/State.h"
#include "peanoclaw/Vertex.h"
#include "peanoclaw/Cell.h"

#include "peano/grid/Grid.h"
#include "peano/stacks/CellSTDStack.h"
#include "peano/stacks/VertexSTDStack.h"


 #include "peanoclaw/adapters/InitialiseGrid.h" 
 #include "peanoclaw/adapters/InitialiseAndValidateGrid.h" 
 #include "peanoclaw/adapters/Plot.h" 
 #include "peanoclaw/adapters/Query.h" 
 #include "peanoclaw/adapters/Remesh.h" 
 #include "peanoclaw/adapters/SolveTimestep.h" 
 #include "peanoclaw/adapters/SolveTimestepAndValidateGrid.h" 
 #include "peanoclaw/adapters/SolveTimestepAndPlot.h" 
 #include "peanoclaw/adapters/SolveTimestepAndPlotAndValidateGrid.h" 
 #include "peanoclaw/adapters/GatherCurrentSolution.h" 
 #include "peanoclaw/adapters/Cleanup.h" 



namespace peanoclaw {
      namespace repositories {
        class RepositorySTDStack;  
      }
}


class peanoclaw::repositories::RepositorySTDStack: public peanoclaw::repositories::Repository {
  private:
    static tarch::logging::Log _log;
  
    peano::geometry::Geometry& _geometry;
    
    typedef peano::stacks::CellSTDStack<peanoclaw::Cell>       CellStack;
    typedef peano::stacks::VertexSTDStack<peanoclaw::Vertex>   VertexStack;

    CellStack    _cellStack;
    VertexStack  _vertexStack;
    peanoclaw::State          _solverState;
    peano::grid::RegularGridContainer<peanoclaw::Vertex,peanoclaw::Cell>  _regularGridContainer;
    peano::grid::TraversalOrderOnTopLevel                                         _traversalOrderOnTopLevel;

    peano::grid::Grid<peanoclaw::Vertex,peanoclaw::Cell,peanoclaw::State,VertexStack,CellStack,peanoclaw::adapters::InitialiseGrid> _gridWithInitialiseGrid;
    peano::grid::Grid<peanoclaw::Vertex,peanoclaw::Cell,peanoclaw::State,VertexStack,CellStack,peanoclaw::adapters::InitialiseAndValidateGrid> _gridWithInitialiseAndValidateGrid;
    peano::grid::Grid<peanoclaw::Vertex,peanoclaw::Cell,peanoclaw::State,VertexStack,CellStack,peanoclaw::adapters::Plot> _gridWithPlot;
    peano::grid::Grid<peanoclaw::Vertex,peanoclaw::Cell,peanoclaw::State,VertexStack,CellStack,peanoclaw::adapters::Query> _gridWithQuery;
    peano::grid::Grid<peanoclaw::Vertex,peanoclaw::Cell,peanoclaw::State,VertexStack,CellStack,peanoclaw::adapters::Remesh> _gridWithRemesh;
    peano::grid::Grid<peanoclaw::Vertex,peanoclaw::Cell,peanoclaw::State,VertexStack,CellStack,peanoclaw::adapters::SolveTimestep> _gridWithSolveTimestep;
    peano::grid::Grid<peanoclaw::Vertex,peanoclaw::Cell,peanoclaw::State,VertexStack,CellStack,peanoclaw::adapters::SolveTimestepAndValidateGrid> _gridWithSolveTimestepAndValidateGrid;
    peano::grid::Grid<peanoclaw::Vertex,peanoclaw::Cell,peanoclaw::State,VertexStack,CellStack,peanoclaw::adapters::SolveTimestepAndPlot> _gridWithSolveTimestepAndPlot;
    peano::grid::Grid<peanoclaw::Vertex,peanoclaw::Cell,peanoclaw::State,VertexStack,CellStack,peanoclaw::adapters::SolveTimestepAndPlotAndValidateGrid> _gridWithSolveTimestepAndPlotAndValidateGrid;
    peano::grid::Grid<peanoclaw::Vertex,peanoclaw::Cell,peanoclaw::State,VertexStack,CellStack,peanoclaw::adapters::GatherCurrentSolution> _gridWithGatherCurrentSolution;
    peano::grid::Grid<peanoclaw::Vertex,peanoclaw::Cell,peanoclaw::State,VertexStack,CellStack,peanoclaw::adapters::Cleanup> _gridWithCleanup;

     
   peanoclaw::records::RepositoryState               _repositoryState;
   
    tarch::timing::Measurement _measureInitialiseGridCPUTime;
    tarch::timing::Measurement _measureInitialiseAndValidateGridCPUTime;
    tarch::timing::Measurement _measurePlotCPUTime;
    tarch::timing::Measurement _measureQueryCPUTime;
    tarch::timing::Measurement _measureRemeshCPUTime;
    tarch::timing::Measurement _measureSolveTimestepCPUTime;
    tarch::timing::Measurement _measureSolveTimestepAndValidateGridCPUTime;
    tarch::timing::Measurement _measureSolveTimestepAndPlotCPUTime;
    tarch::timing::Measurement _measureSolveTimestepAndPlotAndValidateGridCPUTime;
    tarch::timing::Measurement _measureGatherCurrentSolutionCPUTime;
    tarch::timing::Measurement _measureCleanupCPUTime;

    tarch::timing::Measurement _measureInitialiseGridCalendarTime;
    tarch::timing::Measurement _measureInitialiseAndValidateGridCalendarTime;
    tarch::timing::Measurement _measurePlotCalendarTime;
    tarch::timing::Measurement _measureQueryCalendarTime;
    tarch::timing::Measurement _measureRemeshCalendarTime;
    tarch::timing::Measurement _measureSolveTimestepCalendarTime;
    tarch::timing::Measurement _measureSolveTimestepAndValidateGridCalendarTime;
    tarch::timing::Measurement _measureSolveTimestepAndPlotCalendarTime;
    tarch::timing::Measurement _measureSolveTimestepAndPlotAndValidateGridCalendarTime;
    tarch::timing::Measurement _measureGatherCurrentSolutionCalendarTime;
    tarch::timing::Measurement _measureCleanupCalendarTime;

   
  public:
    RepositorySTDStack(
      peano::geometry::Geometry&                   geometry,
      const tarch::la::Vector<DIMENSIONS,double>&  domainSize,
      const tarch::la::Vector<DIMENSIONS,double>&  computationalDomainOffset
    );
    
    /**
     * Parallel Constructor
     *
     * Used in parallel mode only where the size of the domain is not known 
     * when the type of repository is determined.  
     */
    RepositorySTDStack(
      peano::geometry::Geometry&                   geometry
    );
    
    virtual ~RepositorySTDStack();

    virtual void restart(
      const tarch::la::Vector<DIMENSIONS,double>&  domainSize,
      const tarch::la::Vector<DIMENSIONS,double>&  domainOffset,
      int                                          domainLevel,
      const tarch::la::Vector<DIMENSIONS,int>&     positionOfCentralElementWithRespectToCoarserRemoteLevel
    );
         
    virtual void terminate();
        
    virtual peanoclaw::State& getState();
	
    virtual void iterate( bool reduceState );

    virtual void writeCheckpoint(peano::grid::Checkpoint<peanoclaw::Vertex, peanoclaw::Cell> * const checkpoint); 
    virtual void readCheckpoint( peano::grid::Checkpoint<peanoclaw::Vertex, peanoclaw::Cell> const * const checkpoint );
    virtual peano::grid::Checkpoint<peanoclaw::Vertex, peanoclaw::Cell>* createEmptyCheckpoint(); 

    virtual void switchToInitialiseGrid();    
    virtual void switchToInitialiseAndValidateGrid();    
    virtual void switchToPlot();    
    virtual void switchToQuery();    
    virtual void switchToRemesh();    
    virtual void switchToSolveTimestep();    
    virtual void switchToSolveTimestepAndValidateGrid();    
    virtual void switchToSolveTimestepAndPlot();    
    virtual void switchToSolveTimestepAndPlotAndValidateGrid();    
    virtual void switchToGatherCurrentSolution();    
    virtual void switchToCleanup();    

    virtual bool isActiveAdapterInitialiseGrid() const;
    virtual bool isActiveAdapterInitialiseAndValidateGrid() const;
    virtual bool isActiveAdapterPlot() const;
    virtual bool isActiveAdapterQuery() const;
    virtual bool isActiveAdapterRemesh() const;
    virtual bool isActiveAdapterSolveTimestep() const;
    virtual bool isActiveAdapterSolveTimestepAndValidateGrid() const;
    virtual bool isActiveAdapterSolveTimestepAndPlot() const;
    virtual bool isActiveAdapterSolveTimestepAndPlotAndValidateGrid() const;
    virtual bool isActiveAdapterGatherCurrentSolution() const;
    virtual bool isActiveAdapterCleanup() const;

   
    #ifdef Parallel
    virtual ContinueCommand continueToIterate();
    virtual void runGlobalStep();
    #endif

    virtual void setMaximumMemoryFootprintForTemporaryRegularGrids(double value);
    virtual void logIterationStatistics() const;
    virtual void clearIterationStatistics();
};


#endif
