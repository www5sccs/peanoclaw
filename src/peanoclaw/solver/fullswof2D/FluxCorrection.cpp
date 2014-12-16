#include "peanoclaw/solver/fullswof2D/FluxCorrection.h"

#include "peanoclaw/Patch.h"

#include "peano/utils/Loop.h"

#include "peanoclaw/native/FullSWOF2D.h"

#if defined(PEANOCLAW_FULLSWOF2D) && !defined(SCHEME_HPP)
#include "scheme.hpp"
#endif

tarch::logging::Log peanoclaw::solver::fullswof2D::FluxCorrection::_log( "peanoclaw::solver::fullswof2D::FluxCorrection" );

#ifdef PEANOCLAW_FULLSWOF2D
void peanoclaw::solver::fullswof2D::FluxCorrection::setFullSWOF2D(peanoclaw::native::FullSWOF2D* fullswof2D)
{
  _fullswof2D = fullswof2D;
}
#endif

void peanoclaw::solver::fullswof2D::FluxCorrection::computeFluxes(Patch& subgrid) const {
  #ifdef PEANOCLAW_FULLSWOF2D
  switch(subgrid.getUnknownsPerSubcell()) {
    case 1:
      peanoclaw::solver::fullswof2D::FluxCorrectionTemplate<1> fluxCorrection1;
      fluxCorrection1.computeFluxes(subgrid, *_fullswof2D);
      break;
    case 2:
      peanoclaw::solver::fullswof2D::FluxCorrectionTemplate<2> fluxCorrection2;
      fluxCorrection2.computeFluxes(subgrid, *_fullswof2D);
      break;
    case 3:
      peanoclaw::solver::fullswof2D::FluxCorrectionTemplate<3> fluxCorrection3;
      fluxCorrection3.computeFluxes(subgrid, *_fullswof2D);
      break;
    case 4:
      peanoclaw::solver::fullswof2D::FluxCorrectionTemplate<4> fluxCorrection4;
      fluxCorrection4.computeFluxes(subgrid, *_fullswof2D);
      break;
    case 5:
      peanoclaw::solver::fullswof2D::FluxCorrectionTemplate<5> fluxCorrection5;
      fluxCorrection5.computeFluxes(subgrid, *_fullswof2D);
      break;
    case 6:
      peanoclaw::solver::fullswof2D::FluxCorrectionTemplate<6> fluxCorrection6;
      fluxCorrection6.computeFluxes(subgrid, *_fullswof2D);
      break;
    case 7:
      peanoclaw::solver::fullswof2D::FluxCorrectionTemplate<7> fluxCorrection7;
      fluxCorrection7.computeFluxes(subgrid, *_fullswof2D);
      break;
    case 8:
      peanoclaw::solver::fullswof2D::FluxCorrectionTemplate<8> fluxCorrection8;
      fluxCorrection8.computeFluxes(subgrid, *_fullswof2D);
      break;
    case 9:
      peanoclaw::solver::fullswof2D::FluxCorrectionTemplate<9> fluxCorrection9;
      fluxCorrection9.computeFluxes(subgrid, *_fullswof2D);
      break;
    case 10:
      peanoclaw::solver::fullswof2D::FluxCorrectionTemplate<10> fluxCorrection10;
      fluxCorrection10.computeFluxes(subgrid, *_fullswof2D);
      break;
  }
  #endif
}

void peanoclaw::solver::fullswof2D::FluxCorrection::applyCorrection(
    Patch& sourceSubgrid,
    Patch& destinationSubgrid,
    int dimension,
    int direction
) const {
  switch(sourceSubgrid.getUnknownsPerSubcell()) {
      case 1:
        peanoclaw::solver::fullswof2D::FluxCorrectionTemplate<1> fluxCorrection1;
        fluxCorrection1.applyCorrection(sourceSubgrid, destinationSubgrid, dimension, direction);
        break;
      case 2:
        peanoclaw::solver::fullswof2D::FluxCorrectionTemplate<2> fluxCorrection2;
        fluxCorrection2.applyCorrection(sourceSubgrid, destinationSubgrid, dimension, direction);
        break;
      case 3:
        peanoclaw::solver::fullswof2D::FluxCorrectionTemplate<3> fluxCorrection3;
        fluxCorrection3.applyCorrection(sourceSubgrid, destinationSubgrid, dimension, direction);
        break;
      case 4:
        peanoclaw::solver::fullswof2D::FluxCorrectionTemplate<4> fluxCorrection4;
        fluxCorrection4.applyCorrection(sourceSubgrid, destinationSubgrid, dimension, direction);
        break;
      case 5:
        peanoclaw::solver::fullswof2D::FluxCorrectionTemplate<5> fluxCorrection5;
        fluxCorrection5.applyCorrection(sourceSubgrid, destinationSubgrid, dimension, direction);
        break;
      case 6:
        peanoclaw::solver::fullswof2D::FluxCorrectionTemplate<6> fluxCorrection6;
        fluxCorrection6.applyCorrection(sourceSubgrid, destinationSubgrid, dimension, direction);
        break;
      case 7:
        peanoclaw::solver::fullswof2D::FluxCorrectionTemplate<7> fluxCorrection7;
        fluxCorrection7.applyCorrection(sourceSubgrid, destinationSubgrid, dimension, direction);
        break;
      case 8:
        peanoclaw::solver::fullswof2D::FluxCorrectionTemplate<8> fluxCorrection8;
        fluxCorrection8.applyCorrection(sourceSubgrid, destinationSubgrid, dimension, direction);
        break;
      case 9:
        peanoclaw::solver::fullswof2D::FluxCorrectionTemplate<9> fluxCorrection9;
        fluxCorrection9.applyCorrection(sourceSubgrid, destinationSubgrid, dimension, direction);
        break;
      case 10:
        peanoclaw::solver::fullswof2D::FluxCorrectionTemplate<10> fluxCorrection10;
        fluxCorrection10.applyCorrection(sourceSubgrid, destinationSubgrid, dimension, direction);
        break;
    }
}


