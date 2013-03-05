// This file is part of the Peano project. For conditions of distribution and 
// use, please see the copyright notice at www.peano-framework.org
#ifndef _PEANOCLAW_RUNNERS_RUNNER_H_ 
#define _PEANOCLAW_RUNNERS_RUNNER_H_ 


namespace peanoclaw {
  namespace runners {
    class Runner;
  }

  namespace repositories {
    class Repository;
  }
}



/**
 * Runner
 *
 */
class peanoclaw::runners::Runner {
  private:
    int runAsMaster(peanoclaw::repositories::Repository& repository);
    
    #ifdef Parallel
    int runAsClient(peanoclaw::repositories::Repository& repository);
    #endif
  public:
    Runner();
    virtual ~Runner();

    /**
     * Run
     */
    int run(); 
};

#endif
