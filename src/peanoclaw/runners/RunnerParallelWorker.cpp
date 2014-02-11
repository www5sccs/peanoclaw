#include "peanoclaw/runners/Runner.h"


#ifdef Parallel
#include "peano/utils/Globals.h"
#include "tarch/parallel/NodePool.h"
#include "peano/parallel/messages/ForkMessage.h"
#include "peanoclaw/repositories/Repository.h"


int peanoclaw::runners::Runner::runAsWorker(peanoclaw::repositories::Repository& repository) {
  int newMasterNode = tarch::parallel::NodePool::getInstance().waitForJob(); 
  while ( newMasterNode != tarch::parallel::NodePool::JobRequestMessageAnswerValues::Terminate ) {
    if ( newMasterNode >= tarch::parallel::NodePool::JobRequestMessageAnswerValues::NewMaster ) {
      peano::parallel::messages::ForkMessage forkMessage;
      forkMessage.receive(tarch::parallel::NodePool::getInstance().getMasterRank(),tarch::parallel::NodePool::getInstance().getTagForForkMessages(), true, ReceiveIterationControlMessagesBlocking);

      repository.restart(
        forkMessage.getH(),
        forkMessage.getDomainOffset(),
        forkMessage.getLevel(),
        forkMessage.getPositionOfFineGridCellRelativeToCoarseGridCell()
      );
  
      bool continueToIterate = true;
      while (continueToIterate) {
        switch (repository.continueToIterate()) {
          case peanoclaw::repositories::Repository::Continue:
            repository.iterate();  
            break;
          case peanoclaw::repositories::Repository::Terminate:
            continueToIterate = false;  
            break;
          case peanoclaw::repositories::Repository::RunGlobalStep:
            runGlobalStep();  
            break;
        }
      }
    
      // insert your postprocessing here
      // -------------------------------  

      // -------------------------------

      repository.terminate();
    }
    else if ( newMasterNode == tarch::parallel::NodePool::JobRequestMessageAnswerValues::RunAllNodes ) {
      runGlobalStep();  
    }
    newMasterNode = tarch::parallel::NodePool::getInstance().waitForJob(); 
  }
  return 0;
}


void peanoclaw::runners::Runner::runGlobalStep() {
    // insert yourcode here
    // -------------------------------

    // -------------------------------
}
#endif
