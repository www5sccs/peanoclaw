#include "peanoclaw/native/sweMain.h"

#include <cassert>
#include <cstdlib>
#include <string>
#include <iostream>

#if WAVE_PROPAGATION_SOLVER==0 || WAVE_PROPAGATION_SOLVER==1 || WAVE_PROPAGATION_SOLVER==2
  #ifndef CUDA
  #include "blocks/SWE_WavePropagationBlock.hh"
  typedef SWE_WavePropagationBlock Block;
  #else
  #include "blocks/cuda/SWE_WavePropagationBlockCuda.hh"
  typedef SWE_WavePropagationBlockCuda Block;
  #endif
#else
  #include "blocks/SWE_WaveAccumulationBlock.hh"
  typedef SWE_WaveAccumulationBlock Block;
#endif

#include "writer/VtkWriter.hh"

#include "scenarios/SWE_simple_scenarios.hh"
#include "tools/args.hh"
#include "tools/help.hh"
#include "tools/ProgressBar.hh"

void sweMain(
  peanoclaw::native::BreakingDam_SWEKernelScenario& scenario,
  tarch::la::Vector<DIMENSIONS,int> numberOfCells
) {
  tarch::la::Vector<DIMENSIONS,double> resolution
    = tarch::la::multiplyComponents(scenario.getDomainSize(), tarch::la::invertEntries(numberOfCells.convertScalar<double>()));
  Block l_wavePropgationBlock(numberOfCells[0],numberOfCells[1],resolution[0],resolution[1]);

  // initialize the wave propagation block
  l_wavePropgationBlock.initScenario(0, 0, scenario);

  // Init fancy progressbar
  tools::ProgressBar progressBar(scenario.getEndTime());

  // write the output at time zero
  tools::Logger::logger.printOutputTime((float) 0.);
  progressBar.update(0.);

  std::string l_baseName = "plot";
  std::string l_fileName = generateBaseFileName(l_baseName,0,0);
  //boundary size of the ghost layers
  io::BoundarySize l_boundarySize = {{1, 1, 1, 1}};
  // consturct a VtkWriter
  io::VtkWriter l_writer( l_fileName,
          l_wavePropgationBlock.getBathymetry(),
          l_boundarySize,
          numberOfCells[0], numberOfCells[1],
          resolution[0], resolution[1] );
  // Write zero time step
//  l_writer.writeTimeStep( l_wavePropgationBlock.getWaterHeight(),
//                          l_wavePropgationBlock.getDischarge_hu(),
//                          l_wavePropgationBlock.getDischarge_hv(),
//                          (float) 0.);


  /**
   * Simulation.
   */
  // print the start message and reset the wall clock time
  progressBar.clear();
  tools::Logger::logger.printStartMessage();
  tools::Logger::logger.initWallClockTime(time(NULL));

  //! simulation time.
  float l_t = 0.0;
  progressBar.update(l_t);

  unsigned int l_iterations = 0;

  // loop over checkpoints
  //for(int c=1; c<=l_numberOfCheckPoints; c++) {
  for(double time = 0; time <= scenario.getEndTime(); time += scenario.getGlobalTimestepSize()) {

    // do time steps until next checkpoint is reached
    while( l_t < time ) {
      // set values in ghost cells:
      l_wavePropgationBlock.setGhostLayer();

      // reset the cpu clock
      tools::Logger::logger.resetClockToCurrentTime("Cpu");

      // approximate the maximum time step
      // TODO: This calculation should be replaced by the usage of the wave speeds occuring during the flux computation
      // Remark: The code is executed on the CPU, therefore a "valid result" depends on the CPU-GPU-synchronization.
//      l_wavePropgationBlock.computeMaxTimestep();

      // compute numerical flux on each edge
      l_wavePropgationBlock.computeNumericalFluxes();

      //! maximum allowed time step width.
      float l_maxTimeStepWidth = l_wavePropgationBlock.getMaxTimestep();

      // update the cell values
      l_wavePropgationBlock.updateUnknowns(l_maxTimeStepWidth);

      // update the cpu time in the logger
      tools::Logger::logger.updateTime("Cpu");

      // update simulation time with time step width.
      l_t += l_maxTimeStepWidth;
      l_iterations++;

      // print the current simulation time
      progressBar.clear();
      tools::Logger::logger.printSimulationTime(l_t);
      progressBar.update(l_t);
    }

    // print current simulation time of the output
    progressBar.clear();
    tools::Logger::logger.printOutputTime(l_t);
    progressBar.update(l_t);

    // write output
//    l_writer.writeTimeStep( l_wavePropgationBlock.getWaterHeight(),
//                            l_wavePropgationBlock.getDischarge_hu(),
//                            l_wavePropgationBlock.getDischarge_hv(),
//                            l_t);

    if(tarch::la::equals(time, scenario.getEndTime())) {
      break;
    }
  }

  /**
   * Finalize.
   */
  // write the statistics message
  progressBar.clear();
  tools::Logger::logger.printStatisticsMessage();

  // print the cpu time
  tools::Logger::logger.printTime("Cpu", "CPU time");

  // print the wall clock time (includes plotting)
  tools::Logger::logger.printWallClockTime(time(NULL));

  // printer iteration counter
  tools::Logger::logger.printIterationsDone(l_iterations);
}

