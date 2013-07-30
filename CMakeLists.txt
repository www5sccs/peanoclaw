find_package(PythonLibs)

set(PEANOCLAW_SOURCES
    src/peanoclaw/Area.cpp
    src/peanoclaw/Cell.cpp
    src/peanoclaw/State.cpp
    src/peanoclaw/Patch.cpp
    src/peanoclaw/PatchPlotter.cpp
    src/peanoclaw/Vertex.cpp
    src/peanoclaw/Numerics.cpp
    src/peanoclaw/NumericsFactory.cpp

    src/peanoclaw/pyclaw/PyClaw.cpp
    src/peanoclaw/pyclaw/PyClawState.cpp
    src/peanoclaw/pyclaw/pyclawBindings.cpp
    src/peanoclaw/pyclaw/FluxCorrectionCallbackWrapper.cpp
    src/peanoclaw/pyclaw/InterpolationCallbackWrapper.cpp
    src/peanoclaw/pyclaw/RestrictionCallbackWrapper.cpp

    src/peanoclaw/interSubgridCommunication/GhostLayerCompositor.cpp
    src/peanoclaw/interSubgridCommunication/DefaultRestriction.cpp
    src/peanoclaw/interSubgridCommunication/DefaultInterpolation.cpp
    src/peanoclaw/interSubgridCommunication/DefaultFluxCorrection.cpp
    src/peanoclaw/interSubgridCommunication/Extrapolation.cpp
    src/peanoclaw/interSubgridCommunication/GridLevelTransfer.cpp
    src/peanoclaw/interSubgridCommunication/aspects/AdjacentSubgrids.cpp

    src/peanoclaw/adapters/Cleanup.cpp
    src/peanoclaw/adapters/GatherCurrentSolution.cpp
    src/peanoclaw/adapters/InitialiseGrid.cpp
    src/peanoclaw/adapters/InitialiseAndValidateGrid.cpp
    src/peanoclaw/adapters/Plot.cpp
    src/peanoclaw/adapters/Remesh.cpp
    src/peanoclaw/adapters/SolveTimestep.cpp
    src/peanoclaw/adapters/SolveTimestepAndValidateGrid.cpp
    src/peanoclaw/adapters/SolveTimestepAndPlot.cpp
    src/peanoclaw/adapters/SolveTimestepAndPlotAndValidateGrid.cpp
    src/peanoclaw/configurations/PeanoClawConfigurationForSpacetreeGrid.cpp

    src/peanoclaw/mappings/Cleanup.cpp
    src/peanoclaw/mappings/GatherCurrentSolution.cpp
    src/peanoclaw/mappings/InitialiseGrid.cpp
    src/peanoclaw/mappings/Plot.cpp
    src/peanoclaw/mappings/Remesh.cpp
    src/peanoclaw/mappings/SolveTimestep.cpp
    src/peanoclaw/mappings/ValidateGrid.cpp

    src/peanoclaw/records/Cell.cpp
    src/peanoclaw/records/CellDescription.cpp
    src/peanoclaw/records/Data.cpp
    src/peanoclaw/records/RepositoryState.cpp
    src/peanoclaw/records/State.cpp
    src/peanoclaw/records/Vertex.cpp
    src/peanoclaw/records/VertexDescription.cpp
    src/peanoclaw/records/PatchDescription.cpp

    src/peanoclaw/repositories/RepositoryArrayStack.cpp
    src/peanoclaw/repositories/RepositoryFactory.cpp
    src/peanoclaw/repositories/RepositorySTDStack.cpp
 
    src/peanoclaw/runners/PeanoClawLibraryRunner.cpp

    #src/peanoclaw/tests/TestCase.cpp

    src/peanoclaw/statistics/LevelInformation.cpp
    src/peanoclaw/statistics/Probe.cpp
    src/peanoclaw/statistics/PatchDescriptionDatabase.cpp
    src/peanoclaw/statistics/ParallelGridValidator.cpp
    src/peanoclaw/statistics/SubgridStatistics.cpp

    src/peanoclaw/parallel/MasterWorkerAndForkJoinCommunicator.cpp
    src/peanoclaw/parallel/NeighbourCommunicator.cpp

    src/peanoclaw/native/NativeKernel.cpp   
    #src/peanoclaw/native/SWEKernel.cpp   
    #src/peanoclaw/native/SWE/blocks/SWE_Block.cpp
    #src/peanoclaw/native/SWE/blocks/SWE_WavePropagationBlock.cpp
    #src/peanoclaw/native/SWE_WavePropagationBlock_patch.cpp
)


include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/peanoclaw/native/SWE
        ${PYTHON_INCLUDE_DIRS}
        /home/hpc/pr63so/di72bap/mytools-amd/lib/python2.7/site-packages/numpy/core/include # numpy.get_include()
)

add_definitions(-DWAVE_PROPAGATION_SOLVER=1)

add_library (peanoclaw SHARED ${PEANOCLAW_SOURCES})

target_link_libraries (peanoclaw peano_shared peano_tarch_shared ${PYTHON_LIBRARIES} ${TBB_LINK_FLAGS} -lrt)
SET_TARGET_PROPERTIES(peanoclaw PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")

add_executable (peanoclaw_run src/peanoclaw/main.cpp)
target_link_libraries (peanoclaw_run peanoclaw)
