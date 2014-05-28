#include "SWE_WavePropagationBlock_patch.hh"

#include <cassert>
#include <string>
#include <limits>

SWE_WavePropagationBlock_patch::SWE_WavePropagationBlock_patch(peanoclaw::Patch& patch)
  :
#if WAVE_PROPAGATION_SOLVER==1 || WAVE_PROPAGATION_SOLVER==2 || WAVE_PROPAGATION_SOLVER==3
    SWE_WavePropagationBlock(
#else
    SWE_WaveAccumulationBlock(
#endif
          patch.getSubdivisionFactor()(0),
          patch.getSubdivisionFactor()(1),
          patch.getSubcellSize()(0),
          patch.getSubcellSize()(1)
      ),
    _patch(patch)
{
    setBoundaryType(BND_LEFT, CONNECT);
    setBoundaryType(BND_RIGHT, CONNECT);
    setBoundaryType(BND_BOTTOM, CONNECT);
    setBoundaryType(BND_TOP, CONNECT);
    setBoundaryConditions();

    tarch::la::Vector<DIMENSIONS,int> subcellIndex;

    tarch::la::Vector<DIMENSIONS,int> subdivisionFactor = _patch.getSubdivisionFactor();
    peanoclaw::grid::SubgridAccessor& accessor = _patch.getAccessor();

    for (int x = -1; x < subdivisionFactor(0)+1; x++) {
        for (int y = -1; y < subdivisionFactor(1)+1; y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;
            SWE_Block::h[x+1][y+1] = accessor.getValueUOld(subcellIndex, 0);
        }
    }
     
    for (int x = -1; x < subdivisionFactor(0)+1; x++) {
        for (int y = -1; y < subdivisionFactor(1)+1; y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;
            SWE_Block::hu[x+1][y+1] = accessor.getValueUOld(subcellIndex, 1);
        }
    }
  
    for (int x = -1; x < subdivisionFactor(0)+1; x++) {
        for (int y = -1; y < subdivisionFactor(1)+1; y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;
            SWE_Block::hv[x+1][y+1] = accessor.getValueUOld(subcellIndex, 2);
        }
    }
 
    for (int x = -1; x < subdivisionFactor(0)+1; x++) {
        for (int y = -1; y < subdivisionFactor(1)+1; y++) {
          SWE_Block::b[x+1][y+1] = 0.0;
        }
    }
}

SWE_WavePropagationBlock_patch::~SWE_WavePropagationBlock_patch() 
{
    tarch::la::Vector<DIMENSIONS,int> subdivisionFactor = _patch.getSubdivisionFactor();
    peanoclaw::grid::SubgridAccessor& accessor = _patch.getAccessor();

    tarch::la::Vector<DIMENSIONS,int> subcellIndex;
    for (int x = 0; x < subdivisionFactor(0); x++) {
        for (int y = 0; y < subdivisionFactor(1); y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;
            accessor.setValueUNew(subcellIndex, 0, SWE_Block::h[x+1][y+1]);

        }
    }

    for (int x = 0; x < subdivisionFactor(0); x++) {
        for (int y = 0; y < subdivisionFactor(1); y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;
            accessor.setValueUNew(subcellIndex, 1, SWE_Block::hu[x+1][y+1]);

        }
    }

    for (int x = 0; x < subdivisionFactor(0); x++) {
        for (int y = 0; y < subdivisionFactor(1); y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;
            accessor.setValueUNew(subcellIndex, 2, SWE_Block::hv[x+1][y+1]);
        }
    }
}
