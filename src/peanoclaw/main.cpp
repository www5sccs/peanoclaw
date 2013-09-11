#include "tarch/logging/Log.h"
#include "tarch/tests/TestCaseRegistry.h"
#include "tarch/logging/CommandLineLogger.h"
#include "tarch/parallel/Node.h"

#include "peano/peano.h"

#include "peanoclaw/runners/Runner.h"


tarch::logging::Log _log("");


int main(int argc, char** argv) {
  peano::fillLookupTables();

  int parallelSetup = peano::initParallelEnvironment(&argc,&argv);
  if ( parallelSetup!=0 ) {
    #ifdef Parallel
    // Please do not use the logging if MPI doesn't work properly.
    std::cerr << "mpi initialisation wasn't successful. Application shut down" << std::endl;
    #else
    _log.error("main()", "mpi initialisation wasn't successful. Application shut down");
    #endif
    return parallelSetup;
  }

  int sharedMemorySetup = peano::initSharedMemoryEnvironment();
  if (sharedMemorySetup!=0) {
    logError("main()", "shared memory initialisation wasn't successful. Application shut down");
    return sharedMemorySetup;
  }

  int programExitCode = 0;

  // @todo Please insert your code here and reset programExitCode
  //       if something goes wrong. 
  // ============================================================  

  // Configure the output
  tarch::logging::CommandLineLogger::getInstance().clearFilterList();
  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "info", false ) );
  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "debug", true ) );
//  tarch::logging::CommandLineLogger::getInstance().setLogFormat( ... please consult source code documentation );

  // Runs the unit tests
  tarch::tests::TestCaseRegistry::getInstance().getTestCaseCollection().run();  
  programExitCode = tarch::tests::TestCaseRegistry::getInstance().getTestCaseCollection().getNumberOfErrors();

  // Runs the integration tests
  //if (programExitCode==0) {
  //  tarch::tests::TestCaseRegistry::getInstance().getIntegrationTestCaseCollection().run();  
  //  programExitCode = tarch::tests::TestCaseRegistry::getInstance().getIntegrationTestCaseCollection().getNumberOfErrors();
  //}
  
  // dummy call to runner
  if (programExitCode==0) {
    tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "debug", -1, "peanoclaw", false ) );
    peanoclaw::runners::Runner runner;
    programExitCode = runner.run();
  }
  
  // ============================================================  

  if (programExitCode==0) {
    #ifdef Parallel
    if (tarch::parallel::Node::getInstance().isGlobalMaster()) {
      logInfo( "main()", "Peano terminates successfully" );
    }
    #else
    logInfo( "main()", "Peano terminates successfully" );
    #endif
  }
  else {
    logInfo( "main()", "quit with error code " << programExitCode );
  }

  peano::shutdownParallelEnvironment();
  peano::shutdownSharedMemoryEnvironment();

  return programExitCode;
}
