#include "peanoclaw/repositories/Repository.h"
#include "peanoclaw/records/RepositoryState.h"

#include "peanoclaw/State.h"
#include "peanoclaw/Vertex.h"
#include "peanoclaw/Cell.h"

#include "peano/grid/Grid.h"

#include "peano/stacks/CellArrayStack.h"
#include "peano/stacks/CellSTDStack.h"

#include "peano/stacks/VertexArrayStack.h"
#include "peano/stacks/VertexSTDStack.h"

 #include "peanoclaw/adapters/InitialiseGrid.h" 
 #include "peanoclaw/adapters/InitialiseAndValidateGrid.h" 
 #include "peanoclaw/adapters/Plot.h" 
 #include "peanoclaw/adapters/PlotAndValidateGrid.h" 
 #include "peanoclaw/adapters/Remesh.h" 
 #include "peanoclaw/adapters/SolveTimestep.h" 
 #include "peanoclaw/adapters/SolveTimestepAndValidateGrid.h" 
 #include "peanoclaw/adapters/SolveTimestepAndPlot.h" 
 #include "peanoclaw/adapters/SolveTimestepAndPlotAndValidateGrid.h" 
 #include "peanoclaw/adapters/GatherCurrentSolution.h" 
 #include "peanoclaw/adapters/GatherCurrentSolutionAndValidateGrid.h" 
 #include "peanoclaw/adapters/Cleanup.h" 


namespace peano {
  namespace grid {
    template class Grid<peanoclaw::Vertex,peanoclaw::Cell,peanoclaw::State, peano::stacks::VertexArrayStack<peanoclaw::Vertex> ,peano::stacks::CellArrayStack<peanoclaw::Cell> ,peanoclaw::adapters::SolveTimestepAndPlot>;
    template class Grid<peanoclaw::Vertex,peanoclaw::Cell,peanoclaw::State, peano::stacks::VertexSTDStack<  peanoclaw::Vertex> ,peano::stacks::CellSTDStack<  peanoclaw::Cell> ,peanoclaw::adapters::SolveTimestepAndPlot>;
  }
}

#include "peano/grid/Grid.cpph"
