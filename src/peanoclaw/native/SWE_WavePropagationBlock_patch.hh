/**
 * @file
 * This file is part of SWE.
 *
 * @author Alexander Breuer (breuera AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Dipl.-Math._Alexander_Breuer)
 * @author Sebastian Rettenberger (rettenbs AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Sebastian_Rettenberger,_M.Sc.)
 *
 * @section LICENSE
 *
 * SWE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SWE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SWE.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * @section DESCRIPTION
 *
 * SWE_Block, which uses solvers in the wave propagation formulation.
 */

#ifndef SWEWAVEPROPAGATIONBLOCK_PATCH_HH_
#define SWEWAVEPROPAGATIONBLOCK_PATCH_HH_

#include "peanoclaw/Patch.h"

#if WAVE_PROPAGATION_SOLVER==1 || WAVE_PROPAGATION_SOLVER==2 || WAVE_PROPAGATION_SOLVER==3
#include "blocks/SWE_WavePropagationBlock.hh"
#else
#include "blocks/SWE_WaveAccumulationBlock.hh"
#endif

class SWE_WavePropagationBlock_patch
#if WAVE_PROPAGATION_SOLVER==1 || WAVE_PROPAGATION_SOLVER==2 || WAVE_PROPAGATION_SOLVER==3
  : public  SWE_WavePropagationBlock {
#else
  : public  SWE_WaveAccumulationBlock {
#endif
  public:
    //constructor of a SWE_WavePropagationBlock.
    SWE_WavePropagationBlock_patch(peanoclaw::Patch& patch);

    /**
     * Destructor of a SWE_WavePropagationBlock.
     *
     * In the case of a hybrid solver (NDEBUG not defined) information about the used solvers will be printed.
     */
    virtual ~SWE_WavePropagationBlock_patch();

    /**
     * Sets the arrays in the solver.
     */
    void setArrays(
      const peanoclaw::Patch& subgrid,
      float* h,
      float* hu,
      float* hv,
      float* b
    );

  private:
    peanoclaw::Patch& _patch;
};

#endif /* SWEWAVEPROPAGATIONBLOCK_PATCH_HH_ */
