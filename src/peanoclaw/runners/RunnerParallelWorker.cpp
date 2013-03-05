#include "peanoclaw/runners/Runner.h"


#ifdef Parallel
#include "peano/utils/Globals.h"
#include "tarch/parallel/NodePool.h"
#include "peano/parallel/messages/ForkMessage.h"
#include "repositories/Repository.h"


int peanoclaw::runners::Runner::runAsClient(peanoclaw::repositories::Repository& repository) {
  while ( tarch::parallel::NodePool::getInstance().waitForJob() >= tarch::parallel::NodePool::JobRequestMessageAnswerValues::NewMaster ) {
    peano::parallel::messages::ForkMessage forkMessage;
    forkMessage.receive(tarch::parallel::NodePool::getInstance().getMasterRank(),tarch::parallel::NodePool::getInstance().getTagForForkMessages(), true);

    repository.restart(
      forkMessage.getH(),
      forkMessage.getDomainOffset(),
      forkMessage.getLevel()
    );
  
    while (repository.continueToIterate()) {
      repository.iterate();
    }

    // insert your postprocessing here
    // -------------------------------

    // -------------------------------

    repository.terminate();
  }
  return 0;
}
#endif
