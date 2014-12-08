/*
 * FluxCorrectionCallbackWrapper.h
 *
 *  Created on: Mar 7, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_PYCLAW_FLUXCORRECTIONCALLBACKWRAPPER_H_
#define PEANOCLAW_PYCLAW_FLUXCORRECTIONCALLBACKWRAPPER_H_

#include "peanoclaw/pyclaw/PyClawCallbacks.h"
#include "peanoclaw/interSubgridCommunication/FluxCorrection.h"

namespace peanoclaw {
  namespace pyclaw {
    class FluxCorrectionCallbackWrapper;
  }
}

class peanoclaw::pyclaw::FluxCorrectionCallbackWrapper : public peanoclaw::interSubgridCommunication::FluxCorrection {

  private:
    InterPatchCommunicationCallback _fluxCorrectionCallback;

  public:
    FluxCorrectionCallbackWrapper(
      InterPatchCommunicationCallback fluxCorrectionCallback
    );

    virtual ~FluxCorrectionCallbackWrapper();

    /**
     * @see peanoclaw::interSubgridCommunication::FluxCorrection
     */
    void applyCorrection(
      const Patch& finePatch,
      Patch& coarsePatch,
      int dimension,
      int direction
    ) const;

    void computeFluxes(Patch& subgrid) const;
};

#endif /* PEANOCLAW_PYCLAW_FLUXCORRECTIONCALLBACKWRAPPER_H_ */
