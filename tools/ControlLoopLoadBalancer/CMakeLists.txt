include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CONTROLLOOPLOADBALANCER_FILES
    ControlLoopLoadBalancer/OracleForOnePhaseControlLoopWrapper.cpp

    ControlLoopLoadBalancer/GridStateData.cpp
    ControlLoopLoadBalancer/WorkerData.cpp
   
    ControlLoopLoadBalancer/Reductions.cpp
    ControlLoopLoadBalancer/Strategy.cpp
    ControlLoopLoadBalancer/strategies/JoinDueToEraseStrategy.cpp
    ControlLoopLoadBalancer/strategies/ThresholdStrategy.cpp
    ControlLoopLoadBalancer/FilterStrategy.cpp

    ControlLoopLoadBalancer/ControlLoopLoadBalancer.cpp
    ControlLoopLoadBalancer/DefaultControlLoopLoadBalancer.cpp
)

add_library(peano_controllooploadbalancer_static STATIC
    ${CONTROLLOOPLOADBALANCER_FILES}
)

add_library(peano_controllooploadbalancer_shared SHARED
    ${CONTROLLOOPLOADBALANCER_FILES}
)

if (PEANO_OMP)
    SET_TARGET_PROPERTIES(peano_controllooploadbalancer_static PROPERTIES COMPILE_FLAGS "-fopenmp")
endif (PEANO_OMP)

if (PEANO_OMP)
    SET_TARGET_PROPERTIES(peano_controllooploadbalancer_shared PROPERTIES COMPILE_FLAGS "-fopenmp")
endif (PEANO_OMP)
